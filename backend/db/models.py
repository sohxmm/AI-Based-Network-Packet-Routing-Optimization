# TODO: implement
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models in this project."""


class RoutingEvent(Base):
    """
    One row per routing decision made by any algorithm (Dijkstra, Bellman-Ford,
    ACO, or the RL agent). Stores every field from RoutingDecision plus a
    timestamp and derived hop_count, so this table can later be used as
    training/evaluation history for the RL agent and for dashboard metrics
    (avg latency, packet delivery ratio, congestion events).
    """

    __tablename__ = "routing_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

    source: Mapped[str] = mapped_column(String, nullable=False)
    destination: Mapped[str] = mapped_column(String, nullable=False)
    algorithm: Mapped[str] = mapped_column(String, nullable=False, index=True)

    path: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    hop_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    total_latency: Mapped[float] = mapped_column(Float, nullable=False)
    avg_utilization: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return (
            f"<RoutingEvent {self.algorithm} {self.source}->{self.destination} "
            f"success={self.success}>"
        )


class NetworkSnapshot(Base):
    """
    Periodic save of the entire NetworkState (nodes, links, utilization,
    queue sizes, etc.) as a single JSONB blob. This is the ground-truth
    time-series data the LSTM congestion predictor trains on, and is also
    useful for replaying or auditing network history.
    """

    __tablename__ = "network_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    state_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    def __repr__(self) -> str:
        return f"<NetworkSnapshot step={self.step_count} at={self.timestamp}>"