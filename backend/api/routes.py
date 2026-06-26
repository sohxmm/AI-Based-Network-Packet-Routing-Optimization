from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Callable, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.state import get_simulator
from ml.congestion_lstm import CongestionPredictor
from router.aco import AntColonyRouter
from router.bellman_ford import find_route as bellman_ford_route
from router.dijkstra import find_route as dijkstra_route
from router.rl_agent import RLRouter
from simulator.data_models import NetworkState, RoutingDecision

router = APIRouter()

AlgorithmName = Literal["dijkstra", "bellman_ford", "aco", "rl"]
RouteFinder = Callable[[NetworkState, str, str], RoutingDecision]
DEFAULT_ALGORITHMS: list[AlgorithmName] = ["dijkstra", "bellman_ford", "aco", "rl"]
congestion_predictor = CongestionPredictor()
forecast_history: list[list[float]] = []


class RouteRequest(BaseModel):
    """Request body for single-algorithm route calculation."""

    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R5"])
    algorithm: AlgorithmName = "dijkstra"


class RouteCompareRequest(BaseModel):
    """Request body for comparing several routing algorithms."""

    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R5"])
    algorithms: list[AlgorithmName] | None = None


class LinkRequest(BaseModel):
    """Request body for simulator link failure controls."""

    source: str = Field(..., min_length=1, examples=["R1"])
    target: str = Field(..., min_length=1, examples=["R2"])


@router.get("/network/state")
def get_network_state() -> dict[str, object]:
    """Return the current simulator state without advancing the simulation."""
    return _state_to_dict(get_simulator().get_state())


@router.get("/network/topology")
def get_network_topology() -> dict[str, object]:
    """Return static topology-oriented data for graph visualizations."""
    state = get_simulator().get_state()
    return {
        "nodes": [{"id": node, "label": node} for node in state.nodes],
        "links": [
            {
                "source": link.source,
                "target": link.target,
                "base_latency": link.base_latency,
                "bandwidth": link.bandwidth,
            }
            for link in state.links
        ],
        "step_count": state.step_count,
    }


@router.post("/sim/step")
def step_simulation() -> dict[str, object]:
    """Advance the singleton simulator by one step and return the new state."""
    return _state_to_dict(get_simulator().step())


@router.post("/sim/inject-failure")
def inject_failure(request: LinkRequest) -> dict[str, object]:
    """Temporarily remove a link from the active simulator topology."""
    simulator = get_simulator()
    try:
        simulator.inject_failure(request.source, request.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _state_to_dict(simulator.get_state())


@router.post("/sim/restore-link")
def restore_link(request: LinkRequest) -> dict[str, object]:
    """Restore a previously failed link in the active simulator topology."""
    simulator = get_simulator()
    try:
        simulator.restore_link(request.source, request.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _state_to_dict(simulator.get_state())


@router.post("/network/route")
def route_packet(request: RouteRequest) -> dict[str, object]:
    """Calculate a route through the current network using one algorithm."""
    state = get_simulator().get_state()
    _validate_nodes(state, request.source, request.destination)
    decision = _run_algorithm(request.algorithm, state, request.source, request.destination)
    return _decision_to_dict(decision)


@router.post("/network/route/compare")
def compare_routes(request: RouteCompareRequest) -> dict[str, object]:
    """Compare routing decisions across multiple algorithms."""
    state = get_simulator().get_state()
    _validate_nodes(state, request.source, request.destination)
    algorithms = request.algorithms or DEFAULT_ALGORITHMS
    decisions = [
        _decision_to_dict(_run_algorithm(algorithm, state, request.source, request.destination))
        for algorithm in algorithms
    ]
    return {
        "source": request.source,
        "destination": request.destination,
        "step_count": state.step_count,
        "results": decisions,
    }


@router.get("/metrics/summary")
def get_metrics_summary() -> dict[str, object]:
    """Return live summary metrics derived from the current simulator state."""
    state = get_simulator().get_state()
    links = state.links
    avg_utilization = (
        sum(link.utilization for link in links) / len(links)
        if links
        else 0.0
    )
    congestion_events = sum(1 for link in links if link.utilization >= 0.7)
    packet_delivery_rate = _estimate_packet_delivery_rate(state)
    sample_decisions = _sample_algorithm_decisions(state, "dijkstra")
    latencies = [
        decision.total_latency
        for decision in sample_decisions
        if decision.success and isfinite(decision.total_latency)
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        "step_count": state.step_count,
        "avg_latency": avg_latency,
        "avg_utilization": avg_utilization,
        "packet_delivery_rate": packet_delivery_rate,
        "congestion_events": congestion_events,
        "active_algorithm": "dijkstra",
        "rl_trained": RLRouter().is_trained,
    }


@router.get("/metrics/algorithm-comparison")
def get_algorithm_comparison() -> dict[str, object]:
    """Compare algorithms on deterministic sample routes from the live state."""
    state = get_simulator().get_state()
    return {
        "step_count": state.step_count,
        "results": [
            _algorithm_metric_row(state, algorithm)
            for algorithm in DEFAULT_ALGORITHMS
        ],
    }


@router.get("/network/congestion-forecast")
def get_congestion_forecast(steps: int = 3) -> dict[str, object]:
    """Return short-horizon link utilization forecasts for the live network."""
    state = get_simulator().get_state()
    current_snapshot = [link.utilization for link in state.links]
    forecast_history.append(current_snapshot)
    del forecast_history[:-congestion_predictor.seq_len]

    predictions = []
    rolling_window = list(forecast_history)
    for step_index in range(max(1, min(steps, 10))):
        next_snapshot = congestion_predictor.predict_next(rolling_window)
        predictions.append(
            {
                "step_ahead": step_index + 1,
                "links": [
                    {
                        "source": link.source,
                        "target": link.target,
                        "predicted_utilization": next_snapshot[index],
                    }
                    for index, link in enumerate(state.links)
                ],
            }
        )
        rolling_window.append(next_snapshot)

    return {
        "step_count": state.step_count,
        "model_trained": congestion_predictor.model is not None,
        "predictions": predictions,
    }


def _run_algorithm(
    algorithm: AlgorithmName,
    state: NetworkState,
    source: str,
    destination: str,
) -> RoutingDecision:
    """Dispatch route calculation to the selected algorithm implementation."""
    route_finders: dict[AlgorithmName, RouteFinder] = {
        "dijkstra": dijkstra_route,
        "bellman_ford": bellman_ford_route,
        "aco": AntColonyRouter().find_path,
        "rl": RLRouter().predict,
    }
    return route_finders[algorithm](state, source, destination)


def _sample_algorithm_decisions(
    state: NetworkState,
    algorithm: AlgorithmName,
) -> list[RoutingDecision]:
    """Run one algorithm on stable sample source/destination pairs."""
    pairs = _sample_node_pairs(state.nodes)
    return [
        _run_algorithm(algorithm, state, source, destination)
        for source, destination in pairs
    ]


def _sample_node_pairs(nodes: list[str]) -> list[tuple[str, str]]:
    """Create deterministic route samples without relying on database history."""
    if len(nodes) < 2:
        return []

    sample_count = min(5, len(nodes))
    return [
        (nodes[index], nodes[(index + max(1, len(nodes) // 2)) % len(nodes)])
        for index in range(sample_count)
    ]


def _algorithm_metric_row(
    state: NetworkState,
    algorithm: AlgorithmName,
) -> dict[str, object]:
    """Build one live comparison row for an algorithm."""
    decisions = _sample_algorithm_decisions(state, algorithm)
    successes = [decision for decision in decisions if decision.success]
    latencies = [
        decision.total_latency
        for decision in successes
        if isfinite(decision.total_latency)
    ]
    return {
        "algorithm": algorithm,
        "avg_latency": sum(latencies) / len(latencies) if latencies else None,
        "success_rate": len(successes) / len(decisions) if decisions else 0.0,
        "num_decisions": len(decisions),
    }


def _estimate_packet_delivery_rate(state: NetworkState) -> float:
    """Estimate delivery rate from current link packet-loss probabilities."""
    if not state.links:
        return 1.0

    avg_loss = sum(link.packet_loss_rate for link in state.links) / len(state.links)
    return max(0.0, min(1.0, 1.0 - avg_loss))


def _validate_nodes(state: NetworkState, source: str, destination: str) -> None:
    """Raise a clear HTTP error when route endpoints receive unknown routers."""
    missing = [node for node in (source, destination) if node not in state.nodes]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unknown router node.",
                "missing_nodes": missing,
                "available_nodes": state.nodes,
            },
        )


def _state_to_dict(state: NetworkState) -> dict[str, object]:
    """Serialize a NetworkState dataclass for HTTP responses."""
    return {
        "nodes": state.nodes,
        "links": [asdict(link) for link in state.links],
        "timestamp": state.timestamp,
        "step_count": state.step_count,
    }


def _decision_to_dict(decision: RoutingDecision) -> dict[str, object]:
    """Serialize a routing decision with JSON-safe failed-route latency."""
    return {
        "source": decision.source,
        "destination": decision.destination,
        "path": decision.path,
        "algorithm": decision.algorithm,
        "total_latency": decision.total_latency if isfinite(decision.total_latency) else None,
        "avg_utilization": decision.avg_utilization,
        "success": decision.success,
    }
