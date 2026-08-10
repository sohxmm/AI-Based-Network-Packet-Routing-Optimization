from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from api.state import get_simulator, get_rl_router, get_gnn_router
from db.database import get_db
from db.models import RoutingEvent
from .common import _estimate_packet_delivery_rate, _algorithm_metric_row, DEFAULT_ALGORITHMS

router = APIRouter()

@router.get("/metrics/summary")
async def get_metrics_summary(
    algorithm: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    if algorithm and algorithm not in ["dijkstra", "bellman_ford", "aco", "rl", "gnn", "multi_agent"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm name: {algorithm}. Supported: ['dijkstra', 'bellman_ford', 'aco', 'rl', 'gnn']"
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
        congestion_events = sum(1 for e in events if getattr(e, "avg_utilization", 0.0) >= 0.7)
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
        "rl_trained": get_rl_router().is_trained,
        "gnn_trained": get_gnn_router().is_trained,
    }

@router.get("/metrics/history")
async def get_metrics_history(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
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
    state = get_simulator().get_state()
    return {
        "step_count": state.step_count,
        "results": [
            _algorithm_metric_row(state, algorithm)
            for algorithm in DEFAULT_ALGORITHMS
        ],
    }

@router.get("/health/gnn")
def health_check_gnn() -> dict[str, object]:
    gnn = get_gnn_router()
    is_loaded = gnn.is_trained
    return {
        "status": "ok" if is_loaded else "model_not_found",
        "gnn_trained": is_loaded,
        "message": "GNN model is ready" if is_loaded else "Run 'python -m ml.train_gnn' to train"
    }
