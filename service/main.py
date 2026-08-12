"""FastAPI application entrypoint.

The background simulator loop is the part worth reading. It used to catch only
``CancelledError``, so any other exception killed the task while the app stayed
up, ``/health`` kept returning ``{"status": "ok"}``, and the dashboard froze with
no error anywhere. The loop now survives failures, logs them with a stack trace,
backs off after repeated failures, and — critically — ``/health`` reports the age
of the last successful tick, so a dead loop is externally detectable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.api.benchmark import router as benchmark_router
from service.api.experiments import router as experiments_router
from service.api.metrics import router as metrics_router
from service.api.network import router as network_router
from service.api.simulator import handle_simulator_step, router as simulator_router
from service.api.websocket import router as websocket_router
from service.db.database import init_db
from service.db.retention import prune_snapshots
from service.logging_config import configure_logging
from service.state import get_source

logger = logging.getLogger("service.main")

TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1.0"))
RETENTION_INTERVAL_SECONDS = 600
#: After this many consecutive failures the loop slows down instead of
#: hammering a broken dependency once a second.
FAILURE_BACKOFF_THRESHOLD = 10
#: /health reports "degraded" once the last successful tick is older than this.
LIVENESS_WINDOW_SECONDS = 10.0

_last_tick = {"t": 0.0, "consecutive_failures": 0}


async def advance_simulator_forever() -> None:
    """Advance the network source once per tick, surviving transient failures."""
    while True:
        try:
            state = get_source().step()
            await handle_simulator_step(state)
            _last_tick["t"] = time.time()
            _last_tick["consecutive_failures"] = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            _last_tick["consecutive_failures"] += 1
            logger.exception(
                "Simulator tick failed (%d consecutive)",
                _last_tick["consecutive_failures"],
            )
            if _last_tick["consecutive_failures"] >= FAILURE_BACKOFF_THRESHOLD:
                logger.critical(
                    "Simulator loop has failed %d times in a row; backing off to 10s.",
                    _last_tick["consecutive_failures"],
                )
                await asyncio.sleep(10)
        await asyncio.sleep(TICK_SECONDS)


async def prune_snapshots_forever() -> None:
    """Trim the snapshot table periodically so it cannot grow without bound."""
    while True:
        await asyncio.sleep(RETENTION_INTERVAL_SECONDS)
        try:
            await prune_snapshots()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Snapshot retention pass failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the background tasks alongside the application."""
    configure_logging()
    logger.info("Starting AI-Based Network Packet Routing Optimization")

    try:
        await init_db()
    except Exception as exc:  # noqa: BLE001 - the app is useful without a DB
        logger.warning(
            "Database unavailable (%s). The API will run; history and metrics "
            "endpoints will degrade to live-state estimates.",
            exc,
        )

    logger.info("Network source: %s", get_source().describe())

    tasks = [
        asyncio.create_task(advance_simulator_forever()),
        asyncio.create_task(prune_snapshots_forever()),
    ]
    app.state.background_tasks = tasks

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Shutdown complete")


app = FastAPI(
    title="AI-Based Network Packet Routing Optimization",
    description=(
        "A controlled benchmarking platform for congestion-aware routing. "
        "Five routing strategies measured head-to-head under reproducible "
        "scenarios, with QoS constraints, statistical significance testing, "
        "fallback tracking and automated degeneracy detection."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

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
app.include_router(experiments_router)
app.include_router(websocket_router)


@app.get("/health")
def health_check() -> dict[str, object]:
    """Liveness, including whether the simulator loop is actually ticking.

    Returning a bare ``{"status": "ok"}`` was actively misleading: a dead
    background loop left the dashboard frozen while health checks stayed green.
    """
    age = time.time() - _last_tick["t"] if _last_tick["t"] else None
    healthy = age is not None and age < LIVENESS_WINDOW_SECONDS
    return {
        "status": "ok" if healthy else "degraded",
        "simulator_last_tick_age_s": round(age, 2) if age is not None else None,
        "consecutive_tick_failures": _last_tick["consecutive_failures"],
    }


__all__ = ["app"]
