# TODO: implement
import os

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
#print(f"DEBUG: DATABASE_URL = {repr(DATABASE_URL)}") command to debug : python -m db.init_db 2>&1 | Tee-Object -FilePath debug_output.txt

if DATABASE_URL is None:
    raise RuntimeError(
        "DATABASE_URL is not set. Make sure a .env file exists in the project "
        "root (copy .env.example to .env) and that it defines DATABASE_URL."
    )

engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """
    FastAPI dependency that yields a database session for the duration of
    a single request, and always closes it afterward -- even if the request
    raises an exception. Used in route handlers like:

        @app.get("/network/state")
        async def get_state(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        yield session