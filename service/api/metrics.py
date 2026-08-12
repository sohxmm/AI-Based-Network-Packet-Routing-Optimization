"""Metrics, history and model-health endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ml.model_registry import REGISTRY, inventory, path_for
from routing import ALGORITHM_NAMES, LEARNED_ALGORITHMS
from service.api.dispatch import current_state, run_algorithm
from service.db.database import get_db
from service.db.models import AlgorithmMetric, RoutingEvent
from service.state import get_forecaster, get_router

logger = logging.getLogger(__name__)
router = APIRouter()

#: Utilization at or above which a link counts as congested.
CONGESTION_THRESHOLD = 0.7
#: Hard cap on history queries. Previously unbounded, so ?limit=100000000 was
#: a trivial denial-of-service vector.
MAX_HISTORY_LIMIT = 1000


@router.get("/metrics/summary")
async def get_metrics_summary(
    algorithm: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Headline metrics over recent routing events, with a live-state fallback."""
    if algorithm and algorithm not in ALGORITHM_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown algorithm {algorithm!r}. "
                f"Supported: {', '.join(ALGORITHM_NAMES)}."
            ),
        )

    statement = select(RoutingEvent)
    if algorithm:
        statement = statement.where(RoutingEvent.algorithm == algorithm)
    statement = statement.order_by(desc(RoutingEvent.timestamp)).limit(100)

    try:
        events = (await db.execute(statement)).scalars().all()
    except Exception as exc:  # noqa: BLE001 - degrade rather than 500
        logger.warning("Metrics query failed, falling back to live state: %s", exc)
        events = []

    state = current_state()

    if events:
        succeeded = [e for e in events if e.success and e.total_latency is not None]
        avg_latency = (
            sum(e.total_latency for e in succeeded) / len(succeeded) if succeeded else 0.0
        )
        delivery_rate = sum(1 for e in events if e.success) / len(events)
        # avg_utilization is now a real column. It did not exist before, so the
        # getattr default always applied and this count was permanently zero.
        congestion_events = sum(
            1
            for e in events
            if e.avg_utilization is not None and e.avg_utilization >= CONGESTION_THRESHOLD
        )
        fallback_rate = sum(1 for e in events if e.is_fallback) / len(events)
    else:
        losses = [link.packet_loss_rate for link in state.links]
        avg_latency = 0.0
        delivery_rate = max(0.0, 1.0 - (sum(losses) / len(losses) if losses else 0.0))
        congestion_events = sum(
            1 for link in state.links if link.utilization >= CONGESTION_THRESHOLD
        )
        fallback_rate = 0.0

    return {
        "step_count": state.step_count,
        "avg_latency": avg_latency,
        "avg_utilization": (
            sum(link.utilization for link in state.links) / len(state.links)
            if state.links
            else 0.0
        ),
        "packet_delivery_rate": delivery_rate,
        "congestion_events": congestion_events,
        "fallback_rate": fallback_rate,
        "active_algorithm": algorithm or "dijkstra",
        "models": {name: get_router(name).is_trained for name in LEARNED_ALGORITHMS},
        "forecaster_trained": get_forecaster().is_trained,
    }


@router.get("/metrics/history")
async def get_metrics_history(
    limit: int = Query(100, ge=1, le=MAX_HISTORY_LIMIT),
    algorithm: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
    """Recent routing events, newest first."""
    statement = select(RoutingEvent)
    if algorithm:
        statement = statement.where(RoutingEvent.algorithm == algorithm)
    statement = statement.order_by(desc(RoutingEvent.timestamp)).limit(limit)

    try:
        events = (await db.execute(statement)).scalars().all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("History query failed: %s", exc)
        return []

    return [
        {
            "id": event.id,
            "timestamp": event.timestamp,
            "source": event.source,
            "destination": event.destination,
            "algorithm": event.algorithm,
            "traffic_class": event.traffic_class,
            "path": event.path,
            "total_latency": event.total_latency,
            "avg_utilization": event.avg_utilization,
            "success": event.success,
            "is_fallback": event.is_fallback,
            "qos_feasible": event.qos_feasible,
            "step_count": event.step_count,
        }
        for event in events
    ]


@router.get("/metrics/benchmark-history")
async def get_benchmark_history(
    limit: int = Query(200, ge=1, le=MAX_HISTORY_LIMIT),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
    """Persisted benchmark runs, so results can be tracked over time.

    These rows used to be constructed and then discarded behind a commented-out
    commit. Populate the table with ``python -m experiments.runner --persist``.
    """
    statement = (
        select(AlgorithmMetric).order_by(desc(AlgorithmMetric.timestamp)).limit(limit)
    )
    try:
        rows = (await db.execute(statement)).scalars().all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Benchmark history query failed: %s", exc)
        return []

    return [
        {
            "id": row.id,
            "timestamp": row.timestamp,
            "algorithm": row.algorithm,
            "scenario": row.scenario,
            "avg_latency": row.avg_latency,
            "success_rate": row.success_rate,
            "num_decisions": row.num_decisions,
        }
        for row in rows
    ]


@router.get("/metrics/algorithm-comparison")
def get_algorithm_comparison(traffic_class: str = "best_effort") -> dict[str, object]:
    """A quick live comparison across algorithms on a fixed set of demands."""
    state = current_state()
    nodes = state.nodes
    if len(nodes) < 2:
        return {"step_count": state.step_count, "results": []}

    stride = max(1, len(nodes) // 2)
    pairs = [
        (nodes[i], nodes[(i + stride) % len(nodes)])
        for i in range(min(5, len(nodes)))
    ]

    results = []
    for algorithm in ALGORITHM_NAMES:
        decisions = [
            run_algorithm(algorithm, state, src, dst, traffic_class) for src, dst in pairs
        ]
        successes = [d for d in decisions if d.success]
        latencies = [d.total_latency for d in successes if d.total_latency != float("inf")]
        results.append(
            {
                "algorithm": algorithm,
                "avg_latency": sum(latencies) / len(latencies) if latencies else None,
                "success_rate": len(successes) / len(decisions) if decisions else 0.0,
                "fallback_rate": (
                    sum(1 for d in decisions if d.is_fallback) / len(decisions)
                    if decisions
                    else 0.0
                ),
                "num_decisions": len(decisions),
            }
        )

    return {
        "step_count": state.step_count,
        "traffic_class": traffic_class,
        "results": results,
    }


@router.get("/health/models")
def health_check_models() -> dict[str, object]:
    """Which model artifacts exist on disk and which actually loaded.

    This endpoint exists because the project's worst bug was a filename
    mismatch that made three of four AI features silently serve heuristics. A
    reviewer should be able to answer "is the AI actually running?" with one
    HTTP request rather than by reading logs.
    """
    loaded = {name: get_router(name).is_trained for name in LEARNED_ALGORITHMS}
    loaded["lstm"] = get_forecaster().is_trained

    models = inventory()
    for key in models:
        models[key]["loaded_in_memory"] = loaded.get(key)

    all_ok = all(
        entry["file_present"] or not entry["expected_in_repo"] for entry in models.values()
    )
    return {
        "status": "ok" if all_ok else "degraded",
        "models": models,
        "message": (
            "All expected model artifacts are present."
            if all_ok
            else "Some expected artifacts are missing; affected routers are "
            "serving heuristic fallbacks. Run `make train`."
        ),
    }


@router.get("/health/gnn")
def health_check_gnn() -> dict[str, object]:
    """Backward-compatible alias retained for the existing test suite."""
    is_loaded = get_router("gnn").is_trained
    return {
        "status": "ok" if is_loaded else "model_not_found",
        "gnn_trained": is_loaded,
        "message": (
            "GNN model is ready"
            if is_loaded
            else f"Run '{REGISTRY['gnn'].train_command}' to train "
            f"(expected at {path_for('gnn').name})"
        ),
    }


__all__ = ["router"]
