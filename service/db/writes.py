"""Database writes, batched.

``compare_routes`` used to loop over up to six algorithms and, for each one,
open a brand new session and emit a separate WebSocket frame: six sessions and
six broadcasts per request. The N+1 here was on the *write* side.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from math import isfinite

from core.models import RoutingDecision
from service.db.database import AsyncSessionLocal
from service.db.models import NetworkSnapshot, RoutingEvent

logger = logging.getLogger(__name__)


def build_routing_event(
    decision: RoutingDecision, step_count: int, traffic_class: str = "best_effort"
) -> RoutingEvent:
    """Map a decision onto a database row."""
    latency = decision.total_latency
    if latency is not None and not isfinite(latency):
        latency = None

    qos = decision.diagnostics.get("qos") if decision.diagnostics else None
    return RoutingEvent(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC),
        source=decision.source,
        destination=decision.destination,
        algorithm=decision.algorithm,
        traffic_class=traffic_class,
        path=decision.path,
        total_latency=latency,
        avg_utilization=decision.avg_utilization,
        success=decision.success,
        is_fallback=decision.is_fallback,
        qos_feasible=bool(qos.get("feasible")) if isinstance(qos, dict) else None,
        step_count=step_count,
    )


async def save_routing_events(events: list[RoutingEvent]) -> bool:
    """Persist a batch of routing events in a single transaction."""
    if not events:
        return True
    try:
        async with AsyncSessionLocal() as session:
            session.add_all(events)
            await session.commit()
        return True
    except Exception as exc:  # noqa: BLE001 - the app must survive a DB outage
        logger.warning("Failed to persist %d routing events: %s", len(events), exc)
        return False


async def save_snapshot(snapshot: NetworkSnapshot) -> bool:
    """Persist one network snapshot."""
    try:
        async with AsyncSessionLocal() as session:
            session.add(snapshot)
            await session.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist network snapshot: %s", exc)
        return False


__all__ = ["build_routing_event", "save_routing_events", "save_snapshot"]
