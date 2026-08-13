from __future__ import annotations

import copy
import uuid
from dataclasses import asdict
from math import isfinite
from typing import Callable, Literal
from datetime import datetime

from fastapi import HTTPException
from pydantic import BaseModel, Field

from api.state import get_simulator, get_aco_router, get_rl_router, get_gnn_router, get_multi_agent_router
from ml.congestion_lstm import CongestionPredictor
from router.bellman_ford import find_route as bellman_ford_route
from router.dijkstra import find_route as dijkstra_route
from simulator.data_models import LinkState, NetworkState, RoutingDecision
from db.database import AsyncSessionLocal
from db.models import RoutingEvent, NetworkSnapshot
from api.websocket import manager

AlgorithmName = Literal["dijkstra", "bellman_ford", "aco", "rl", "gnn", "multi_agent"]
RouteFinder = Callable[[NetworkState, str, str], RoutingDecision]
DEFAULT_ALGORITHMS: list[AlgorithmName] = ["dijkstra", "bellman_ford", "aco", "rl", "gnn", "multi_agent"]
congestion_predictor = CongestionPredictor()

try:
    from pathlib import Path as _Path
    _LSTM_MODEL = _Path(__file__).resolve().parents[2] / "ml" / "models" / "congestion_lstm.pt"
    if _LSTM_MODEL.exists():
        congestion_predictor.load(str(_LSTM_MODEL))
        print("[Routes] Loaded congestion LSTM model")
except Exception as exc:
    print(f"[Routes] Could not load congestion LSTM model: {exc}")

forecast_history: list[list[float]] = []

class RouteRequest(BaseModel):
    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R5"])
    algorithm: AlgorithmName = "dijkstra"
    use_forecast: bool = False

class RouteCompareRequest(BaseModel):
    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R5"])
    algorithms: list[AlgorithmName] | None = None
    use_forecast: bool = False

class LinkRequest(BaseModel):
    source: str = Field(..., min_length=1, examples=["R1"])
    target: str = Field(..., min_length=1, examples=["R2"])

def _run_algorithm(
    algorithm: AlgorithmName,
    state: NetworkState,
    source: str,
    destination: str,
) -> RoutingDecision:
    route_finders: dict[AlgorithmName, RouteFinder] = {
        "dijkstra": dijkstra_route,
        "bellman_ford": bellman_ford_route,
        "aco": get_aco_router().find_path,
        "rl": get_rl_router().predict,
        "gnn": get_gnn_router().predict,
        "multi_agent": get_multi_agent_router().find_route,
    }
    if algorithm not in route_finders:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm name: {algorithm}. Supported: {list(route_finders.keys())}"
        )
    return route_finders[algorithm](state, source, destination)

def _build_forecast_state(state: NetworkState) -> NetworkState | None:
    current_snapshot = [link.utilization for link in state.links]
    forecast_history.append(current_snapshot)
    del forecast_history[:-congestion_predictor.seq_len]

    if len(forecast_history) < congestion_predictor.seq_len or congestion_predictor.model is None:
        return None

    predicted_utils = congestion_predictor.predict_next(forecast_history)
    new_links = []
    for i, link in enumerate(state.links):
        pred_util = predicted_utils[i] if i < len(predicted_utils) else link.utilization
        pred_util = max(0.0, min(1.0, pred_util))
        new_links.append(LinkState(
            source=link.source,
            target=link.target,
            base_latency=link.base_latency,
            bandwidth=link.bandwidth,
            utilization=pred_util,
            queue_size=int(pred_util * 100),
            packet_loss_rate=max(0.0, pred_util - 0.7) * 0.2,
        ))

    return NetworkState(
        nodes=list(state.nodes),
        links=new_links,
        timestamp=state.timestamp,
        step_count=state.step_count,
    )

def _sample_algorithm_decisions(state: NetworkState, algorithm: AlgorithmName) -> list[RoutingDecision]:
    pairs = _sample_node_pairs(state.nodes)
    return [_run_algorithm(algorithm, state, src, dest) for src, dest in pairs]

def _sample_node_pairs(nodes: list[str]) -> list[tuple[str, str]]:
    if len(nodes) < 2: return []
    sample_count = min(5, len(nodes))
    return [(nodes[index], nodes[(index + max(1, len(nodes) // 2)) % len(nodes)]) for index in range(sample_count)]

def _algorithm_metric_row(state: NetworkState, algorithm: AlgorithmName) -> dict[str, object]:
    decisions = _sample_algorithm_decisions(state, algorithm)
    successes = [d for d in decisions if d.success]
    latencies = [d.total_latency for d in successes if isfinite(d.total_latency)]
    return {
        "algorithm": algorithm,
        "avg_latency": sum(latencies) / len(latencies) if latencies else None,
        "success_rate": len(successes) / len(decisions) if decisions else 0.0,
        "num_decisions": len(decisions),
    }

def _estimate_packet_delivery_rate(state: NetworkState) -> float:
    if not state.links: return 1.0
    avg_loss = sum(link.packet_loss_rate for link in state.links) / len(state.links)
    return max(0.0, min(1.0, 1.0 - avg_loss))

def _validate_nodes(state: NetworkState, source: str, destination: str) -> None:
    missing = [node for node in (source, destination) if node not in state.nodes]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={"message": "Unknown router node.", "missing_nodes": missing, "available_nodes": state.nodes},
        )

def _state_to_dict(state: NetworkState) -> dict[str, object]:
    return {
        "nodes": state.nodes,
        "links": [asdict(link) for link in state.links],
        "timestamp": state.timestamp,
        "step_count": state.step_count,
    }

def _decision_to_dict(decision: RoutingDecision) -> dict[str, object]:
    return {
        "source": decision.source,
        "destination": decision.destination,
        "path": decision.path,
        "algorithm": decision.algorithm,
        "total_latency": decision.total_latency if isfinite(decision.total_latency) else None,
        "avg_utilization": decision.avg_utilization,
        "success": decision.success,
        "is_fallback": getattr(decision, "is_fallback", False),
    }

async def save_routing_event(decision: RoutingDecision, step_count: int) -> None:
    latency = decision.total_latency
    if latency is not None and not isfinite(latency): latency = None
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
    await manager.broadcast({"type": "routing_event", "payload": _decision_to_dict(decision)})
