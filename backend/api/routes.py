from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Callable, Literal
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from api.state import get_simulator
from ml.congestion_lstm import CongestionPredictor
from router.aco import AntColonyRouter
from router.bellman_ford import find_route as bellman_ford_route
from router.dijkstra import find_route as dijkstra_route
from router.rl_agent import RLRouter
from simulator.data_models import NetworkState, RoutingDecision

from db.database import get_db, AsyncSessionLocal
from db.models import RoutingEvent, NetworkSnapshot
from api.websocket import manager

router = APIRouter()

AlgorithmName = Literal["dijkstra", "bellman_ford", "aco", "rl"]
RouteFinder = Callable[[NetworkState, str, str], RoutingDecision]
DEFAULT_ALGORITHMS: list[AlgorithmName] = ["dijkstra", "bellman_ford", "aco", "rl"]
congestion_predictor = CongestionPredictor()
forecast_history: list[list[float]] = []


async def handle_simulator_step(state: NetworkState) -> None:
    """Save network snapshot to database and broadcast state update to all WebSocket clients."""
    # 1. Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "state_update",
        "payload": _state_to_dict(state)
    })

    # 2. Save snapshot to database
    links = state.links
    avg_utilization = (
        sum(link.utilization for link in links) / len(links)
        if links
        else 0.0
    )
    congested_links = sum(1 for link in links if link.utilization >= 0.7)
    
    snapshot = NetworkSnapshot(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        state_json=_state_to_dict(state),
        avg_utilization=avg_utilization,
        congested_links=congested_links,
        step_count=state.step_count
    )
    
    try:
        async with AsyncSessionLocal() as session:
            session.add(snapshot)
            await session.commit()
    except Exception as exc:
        print(f"[DB] Error saving network snapshot: {exc}")


async def save_routing_event(decision: RoutingDecision, step_count: int) -> None:
    """Save routing decision to database."""
    # Ensure total_latency is a valid float/None in DB (convert inf/nan to None)
    latency = decision.total_latency
    if latency is not None and not isfinite(latency):
        latency = None
        
    event = RoutingEvent(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        source=decision.source,
        destination=decision.destination,
        algorithm=decision.algorithm,
        path=decision.path,
        total_latency=latency,
        success=decision.success,
        step_count=step_count,
    )
    try:
        async with AsyncSessionLocal() as session:
            session.add(event)
            await session.commit()
    except Exception as exc:
        print(f"[DB] Error saving routing event: {exc}")


async def broadcast_routing_event(decision: RoutingDecision) -> None:
    """Broadcast routing event to all connected WebSocket clients."""
    await manager.broadcast({
        "type": "routing_event",
        "payload": _decision_to_dict(decision)
    })


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
async def step_simulation() -> dict[str, object]:
    """Advance the singleton simulator by one step and return the new state."""
    state = get_simulator().step()
    await handle_simulator_step(state)
    return _state_to_dict(state)


@router.post("/sim/reset")
async def reset_simulation() -> dict[str, object]:
    """Reset the singleton simulator to its initial state."""
    state = get_simulator().reset()
    await handle_simulator_step(state)
    return _state_to_dict(state)


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
async def route_packet(
    request: RouteRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Calculate a route through the current network using one algorithm."""
    state = get_simulator().get_state()
    _validate_nodes(state, request.source, request.destination)
    decision = _run_algorithm(request.algorithm, state, request.source, request.destination)
    
    await save_routing_event(decision, state.step_count)
    await broadcast_routing_event(decision)
    
    if not decision.success:
        raise HTTPException(
            status_code=400,
            detail=f"No path exists between {request.source} and {request.destination} using {request.algorithm}."
        )
        
    return _decision_to_dict(decision)


@router.post("/network/route/compare")
async def compare_routes(
    request: RouteCompareRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Compare routing decisions across multiple algorithms."""
    state = get_simulator().get_state()
    _validate_nodes(state, request.source, request.destination)
    algorithms = request.algorithms or DEFAULT_ALGORITHMS
    
    decisions = []
    has_any_success = False
    
    for algorithm in algorithms:
        decision = _run_algorithm(algorithm, state, request.source, request.destination)
        await save_routing_event(decision, state.step_count)
        await broadcast_routing_event(decision)
        
        if decision.success:
            has_any_success = True
            
        decisions.append(_decision_to_dict(decision))
        
    if not has_any_success:
        raise HTTPException(
            status_code=400,
            detail=f"No path exists between {request.source} and {request.destination} using any of the selected algorithms."
        )
        
    return {
        "source": request.source,
        "destination": request.destination,
        "step_count": state.step_count,
        "results": decisions,
    }


@router.get("/metrics/summary")
async def get_metrics_summary(
    algorithm: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Return summary metrics derived from the database for the last 100 decisions."""
    if algorithm and algorithm not in ["dijkstra", "bellman_ford", "aco", "rl"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm name: {algorithm}. Supported: ['dijkstra', 'bellman_ford', 'aco', 'rl']"
        )
        
    stmt = select(RoutingEvent)
    if algorithm:
        stmt = stmt.where(RoutingEvent.algorithm == algorithm)
    stmt = stmt.order_by(desc(RoutingEvent.timestamp)).limit(100)
    
    result = await db.execute(stmt)
    events = result.scalars().all()
    
    if events:
        successful_events = [e for e in events if e.success and e.total_latency is not None]
        avg_latency = (
            sum(e.total_latency for e in successful_events) / len(successful_events)
            if successful_events
            else 0.0
        )
        packet_delivery_rate = sum(1 for e in events if e.success) / len(events)
        congestion_events = sum(1 for e in events if e.avg_utilization >= 0.7)
    else:
        state = get_simulator().get_state()
        avg_latency = 0.0
        packet_delivery_rate = _estimate_packet_delivery_rate(state)
        congestion_events = sum(1 for link in state.links if link.utilization >= 0.7)
        
    state = get_simulator().get_state()
    return {
        "step_count": state.step_count,
        "avg_latency": avg_latency,
        "avg_utilization": sum(link.utilization for link in state.links) / len(state.links) if state.links else 0.0,
        "packet_delivery_rate": packet_delivery_rate,
        "congestion_events": congestion_events,
        "active_algorithm": algorithm or "dijkstra",
        "rl_trained": RLRouter().is_trained,
    }


@router.get("/metrics/history")
async def get_metrics_history(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
    """Return the last N routing events from the database."""
    stmt = select(RoutingEvent).order_by(desc(RoutingEvent.timestamp)).limit(limit)
    result = await db.execute(stmt)
    events = result.scalars().all()
    return [
        {
            "id": event.id,
            "timestamp": event.timestamp,
            "source": event.source,
            "destination": event.destination,
            "algorithm": event.algorithm,
            "path": event.path,
            "total_latency": event.total_latency,
            "success": event.success,
            "step_count": event.step_count,
        }
        for event in events
    ]


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
    if algorithm not in route_finders:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm name: {algorithm}. Supported: {list(route_finders.keys())}"
        )
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
