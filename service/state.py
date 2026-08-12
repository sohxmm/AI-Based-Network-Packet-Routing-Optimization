"""Singleton services shared by the REST and WebSocket handlers.

The singleton pattern here is *correctly motivated* and worth keeping: ACO
pheromone tables and loaded torch weights genuinely must persist across
requests, and rebuilding a router per request would both discard learned state
and reload model weights from disk on every call.

Two things changed. The routers are now built through
:func:`routing.build_router_set`, so the benchmark harness can construct its own
isolated set instead of reaching in and mutating these — running a sandbox
experiment used to permanently shift the live dashboard's pheromone table.
And the network source is swappable at runtime, so the same dashboard can be
driven by the simulator, by a recorded trace, or by live measurements of the
operator's own network.
"""

from __future__ import annotations

import logging
import os

from core.sources import (
    LiveProbeSource,
    NetworkSource,
    ProbeTarget,
    SimulatedSource,
    TraceReplaySource,
)
from ml.model_registry import log_model_inventory
from routing import build_router_set
from routing.failover import FailoverMonitor
from routing.learned.forecaster import CongestionPredictor

logger = logging.getLogger(__name__)

#: Live probing is opt-in. It measures real hosts, so it must never start by
#: accident just because someone deployed the container.
LIVE_PROBE_ENABLED = os.getenv("LIVE_PROBE_ENABLED", "0") == "1"


class AppState:
    """Hold the singleton services the API depends on."""

    def __init__(self) -> None:
        log_model_inventory()

        self.source: NetworkSource = SimulatedSource(num_nodes=25, seed=42)
        self.routers = build_router_set(seed=42, load_models=True)

        self.forecaster = CongestionPredictor()
        self.forecaster.load()

        # The failover monitor watches flows using the best available router.
        self.failover = FailoverMonitor(self.routers["gnn"])

    # -- network source ---------------------------------------------------

    def set_source(self, source: NetworkSource) -> NetworkSource:
        """Swap the live network source and reset dependent state."""
        self.source = source
        self.forecaster.history.clear()
        self.failover.clear()
        logger.info("Network source switched to %s", source.describe())
        return source

    def use_simulator(self, num_nodes: int = 25, seed: int = 42) -> NetworkSource:
        return self.set_source(SimulatedSource(num_nodes=num_nodes, seed=seed))

    def use_trace(self, path: str) -> NetworkSource:
        return self.set_source(TraceReplaySource(path))

    def use_live(self, hosts: list[str] | None = None) -> NetworkSource:
        """Switch to live probing of the operator's own network."""
        if not LIVE_PROBE_ENABLED:
            raise PermissionError(
                "Live probing is disabled. Set LIVE_PROBE_ENABLED=1 to enable it, "
                "and only point it at networks you are authorised to measure."
            )
        targets = (
            [ProbeTarget(host=h, label=_label_for(h)) for h in hosts] if hosts else None
        )
        return self.set_source(LiveProbeSource(targets=targets))


def _label_for(host: str) -> str:
    """Turn a hostname or IP into a short, graph-safe node label."""
    cleaned = host.replace(".", "_").replace(":", "_").upper()
    return cleaned[:16]


app_state = AppState()


# -- accessors ------------------------------------------------------------
def get_source() -> NetworkSource:
    """Return the active network source."""
    return app_state.source


def get_simulator():
    """Return the underlying simulator, when the active source has one.

    Returns ``None`` for trace and live sources, which cannot be stepped,
    reset or have failures injected. Callers must handle that.
    """
    return getattr(app_state.source, "simulator", None)


def get_router(name: str):
    """Return the singleton router registered under *name*."""
    return app_state.routers[name]


def get_routers() -> dict:
    return app_state.routers


def get_forecaster() -> CongestionPredictor:
    return app_state.forecaster


def get_failover() -> FailoverMonitor:
    return app_state.failover


__all__ = [
    "LIVE_PROBE_ENABLED",
    "AppState",
    "app_state",
    "get_failover",
    "get_forecaster",
    "get_router",
    "get_routers",
    "get_simulator",
    "get_source",
]
