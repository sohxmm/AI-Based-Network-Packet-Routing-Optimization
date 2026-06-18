"""
Create all database tables defined in db/models.py.

Run this once after the PostgreSQL container is up (docker compose up -d)
and before starting the FastAPI app for the first time, or whenever a new
table/column is added to models.py during development:

    python -m db.init_db
"""

import asyncio

from db.database import engine
from db.models import Base


async def init_db() -> None:
    """Create every table defined on Base's metadata if it doesn't exist yet."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("All tables created (or already existed).")


if __name__ == "__main__":
    asyncio.run(init_db())
