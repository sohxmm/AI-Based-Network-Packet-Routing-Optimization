"""Async SQLAlchemy engine and session factory.

Modernised: ``async_sessionmaker`` instead of the legacy
``sessionmaker(class_=AsyncSession)``, and ``pool_pre_ping=True`` because in
Docker a database restart leaves stale pooled connections and the first request
after an idle period fails without it.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://routinguser:routingpass@localhost:5433/routing_db",
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db():
    """FastAPI dependency yielding a database session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create any missing tables.

    ``create_all`` can only CREATE, never ALTER, which is why schema changes
    silently failed to reach existing deployments. Alembic in
    ``service/migrations/`` is the real migration path; this remains as a
    convenience for tests and throwaway databases.
    """
    from service.db.models import Base

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")


__all__ = ["AsyncSessionLocal", "DATABASE_URL", "engine", "get_db", "init_db"]
