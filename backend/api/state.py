"""Shared application state for simulator-backed API endpoints."""

from simulator.network_sim import NetworkSimulator


class AppState:
    """Hold singleton services used by REST and future WebSocket handlers."""

    def __init__(self) -> None:
        self.simulator = NetworkSimulator()


app_state = AppState()


def get_simulator() -> NetworkSimulator:
    """Return the singleton network simulator instance."""
    return app_state.simulator
