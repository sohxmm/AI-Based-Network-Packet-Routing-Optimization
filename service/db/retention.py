"""Snapshot retention.

``network_snapshots`` grew at roughly 860 MB/day with no retention, no
partitioning and no compression. Combined with writing every tick from inside
the 1 Hz simulator loop, that was an operational hazard rather than a feature.
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, func, select

from service.db.database import AsyncSessionLocal
from service.db.models import NetworkSnapshot

logger = logging.getLogger(__name__)

#: About 28 hours of history at one snapshot per ten simulator ticks.
MAX_SNAPSHOTS = 10_000


async def prune_snapshots(max_rows: int = MAX_SNAPSHOTS) -> int:
    """Keep only the newest *max_rows* snapshots. Returns rows deleted."""
    try:
        async with AsyncSessionLocal() as session:
            total = (
                await session.execute(select(func.count()).select_from(NetworkSnapshot))
            ).scalar_one()
            if total <= max_rows:
                return 0

            cutoff = (
                await session.execute(
                    select(NetworkSnapshot.timestamp)
                    .order_by(NetworkSnapshot.timestamp.desc())
                    .offset(max_rows)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if cutoff is None:
                return 0

            result = await session.execute(
                delete(NetworkSnapshot).where(NetworkSnapshot.timestamp < cutoff)
            )
            await session.commit()
            deleted = int(result.rowcount or 0)
            if deleted:
                logger.info("Pruned %d old network snapshots", deleted)
            return deleted
    except Exception as exc:  # noqa: BLE001 - retention must never take the app down
        logger.warning("Snapshot pruning failed: %s", exc)
        return 0


__all__ = ["MAX_SNAPSHOTS", "prune_snapshots"]
