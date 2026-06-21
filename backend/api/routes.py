from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Callable, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.state import get_simulator
from router.aco import AntColonyRouter
from router.bellman_ford import find_route as bellman_ford_route
from router.dijkstra import find_route as dijkstra_route
from router.rl_agent import RLRouter
from simulator.data_models import NetworkState, RoutingDecision

router = APIRouter()

AlgorithmName = Literal["dijkstra", "bellman_ford", "aco", "rl"]
RouteFinder = Callable[[NetworkState, str, str], RoutingDecision]


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
    algorithms = request.algorithms or ["dijkstra", "bellman_ford", "aco", "rl"]
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
