from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.network import router as network_router
from api.routers.simulator import router as simulator_router, handle_simulator_step
from api.routers.metrics import router as metrics_router

from api.benchmark_api import router as benchmark_router
from api.experiment_api import router as experiment_router
from api.state import get_simulator
from api.websocket import router as websocket_router
from db.database import init_db


async def advance_simulator_forever() -> None:
    """Advance the network simulator once per second until the app shuts down."""
    try:
        while True:
            state = get_simulator().step()
            await handle_simulator_step(state)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the background simulator loop with the FastAPI app."""
    # Initialize the database tables on startup
    print("[DB] Initializing database tables...")
    try:
        await init_db()
        print("[DB] Database tables initialized successfully.")
    except Exception as exc:
        print(f"[DB] Database initialization failed: {exc}")

    simulator_task = asyncio.create_task(advance_simulator_forever())
    app.state.simulator_task = simulator_task

    try:
        yield
    finally:
        simulator_task.cancel()
        try:
            await simulator_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="AI-Based Network Packet Routing Optimization",
    lifespan=lifespan,
)
import os
_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(network_router)
app.include_router(simulator_router)
app.include_router(metrics_router)
app.include_router(benchmark_router)
app.include_router(experiment_router)
app.include_router(websocket_router)



@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a minimal health response for uptime checks."""
    return {"status": "ok"}
