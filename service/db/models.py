"""SQLAlchemy models.

Two changes from the audited schema:

* ``PacketLog`` is gone. It had zero references anywhere in the codebase —
  infrastructure for its own sake, which reads as padding.
* ``RoutingEvent`` gains ``avg_utilization``. The metrics endpoint computed
  ``congestion_events`` from ``getattr(e, "avg_utilization", 0.0)``, but the
  column did not exist, so the default always applied and the metric was
  silently, permanently zero.

``AlgorithmMetric`` is kept and *activated*: the benchmark used to build these
rows and then throw them away behind a commented-out commit, which is dead code
that looks live. It is now written when ``--persist`` is passed and readable via
``GET /metrics/benchmark-history``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow() -> datetime:
    """Timezone-aware UTC now. ``datetime.utcnow()`` is deprecated."""
    return datetime.now(timezone.utc)


class RoutingEvent(Base):
    """One routing decision made by the system."""

    __tablename__ = "routing_events"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, index=True)
    source = Column(String, index=True)
    destination = Column(String, index=True)
    algorithm = Column(String, index=True)
    traffic_class = Column(String, index=True)
    path = Column(JSON)
    total_latency = Column(Float)
    avg_utilization = Column(Float)
    success = Column(Boolean)
    is_fallback = Column(Boolean, default=False)
    qos_feasible = Column(Boolean)
    step_count = Column(Integer, index=True)


class NetworkSnapshot(Base):
    """A full network state at one instant.

    Written every Nth tick rather than every tick. At 1 Hz with ~10 KB rows this
    table grew about 860 MB/day, unbounded, and the write was awaited inside the
    simulator loop so a slow database stalled the simulation itself.
    """

    __tablename__ = "network_snapshots"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, index=True)
    state_json = Column(JSON)
    avg_utilization = Column(Float)
    congested_links = Column(Integer)
    step_count = Column(Integer, index=True)


class AlgorithmMetric(Base):
    """Aggregate benchmark metrics for one algorithm in one scenario."""

    __tablename__ = "algorithm_metrics"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=_utcnow, index=True)
    algorithm = Column(String, index=True)
    scenario = Column(String, index=True)
    window_start_step = Column(Integer)
    window_end_step = Column(Integer)
    avg_latency = Column(Float)
    success_rate = Column(Float)
    num_decisions = Column(Integer)


__all__ = ["AlgorithmMetric", "Base", "NetworkSnapshot", "RoutingEvent"]
