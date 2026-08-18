"""Initial schema.

Creates the three tables the application actually uses. ``PacketLog`` is
deliberately absent: it had zero references anywhere in the codebase, which is
schema for its own sake.

Revision ID: 0001
Revises:
Create Date: 2026-08-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "routing_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), index=True),
        sa.Column("source", sa.String(), index=True),
        sa.Column("destination", sa.String(), index=True),
        sa.Column("algorithm", sa.String(), index=True),
        sa.Column("traffic_class", sa.String(), index=True),
        sa.Column("path", sa.JSON()),
        sa.Column("total_latency", sa.Float()),
        # Added because metrics.py computed congestion_events from a column
        # that did not exist, so the default always applied and the metric was
        # permanently zero.
        sa.Column("avg_utilization", sa.Float()),
        sa.Column("success", sa.Boolean()),
        sa.Column("is_fallback", sa.Boolean(), server_default=sa.false()),
        sa.Column("qos_feasible", sa.Boolean()),
        sa.Column("step_count", sa.Integer(), index=True),
    )

    op.create_table(
        "network_snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), index=True),
        sa.Column("state_json", sa.JSON()),
        sa.Column("avg_utilization", sa.Float()),
        sa.Column("congested_links", sa.Integer()),
        sa.Column("step_count", sa.Integer(), index=True),
    )

    op.create_table(
        "algorithm_metrics",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), index=True),
        sa.Column("algorithm", sa.String(), index=True),
        sa.Column("scenario", sa.String(), index=True),
        sa.Column("window_start_step", sa.Integer()),
        sa.Column("window_end_step", sa.Integer()),
        sa.Column("avg_latency", sa.Float()),
        sa.Column("success_rate", sa.Float()),
        sa.Column("num_decisions", sa.Integer()),
    )


def downgrade() -> None:
    op.drop_table("algorithm_metrics")
    op.drop_table("network_snapshots")
    op.drop_table("routing_events")
