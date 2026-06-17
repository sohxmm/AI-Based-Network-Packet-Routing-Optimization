import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Read database URL from environment or use default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://routinguser:routingpass@localhost:5432/routing_db"
)

# Create async engine (allows concurrent DB operations)
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set True to debug SQL queries
    future=True,
)

# Session factory for creating DB connections
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """Dependency for FastAPI routes: provides a DB session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Create all tables in database on startup."""
    async with engine.begin() as conn:
        from db.models import Base
        await conn.run_sync(Base.metadata.create_all)