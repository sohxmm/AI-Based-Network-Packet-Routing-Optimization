from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from api.state import get_simulator
from api.websocket import router as websocket_router


async def advance_simulator_forever() -> None:
    """Advance the network simulator once per second until the app shuts down."""
    try:
        while True:
            get_simulator().step()
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the background simulator loop with the FastAPI app."""
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.include_router(websocket_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a minimal health response for uptime checks."""
    return {"status": "ok"}
