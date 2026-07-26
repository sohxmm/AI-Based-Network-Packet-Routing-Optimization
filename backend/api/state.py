"""Shared application state for simulator-backed API endpoints."""

from simulator.network_sim import NetworkSimulator
from router.aco import AntColonyRouter
from router.gnn_router import GNNRouter
from router.rl_agent import RLRouter


class AppState:
    """Hold singleton services used by REST and future WebSocket handlers."""

    def __init__(self) -> None:
        self.simulator = NetworkSimulator(num_nodes=25, seed=42)

        # Router singletons — loaded once, reused across all requests
        self.aco_router = AntColonyRouter()
        self.rl_router = RLRouter()
        self.rl_router.try_load_model()
        self.gnn_router = GNNRouter()
        self.gnn_router.try_load_model()



app_state = AppState()


def get_simulator() -> NetworkSimulator:
    """Return the singleton network simulator instance."""
    return app_state.simulator


def get_aco_router() -> AntColonyRouter:
    """Return the singleton ACO router instance (preserves pheromone state)."""
    return app_state.aco_router


def get_rl_router() -> RLRouter:
    """Return the singleton RL router instance (model loaded once at startup)."""
    return app_state.rl_router


def get_gnn_router() -> GNNRouter:
    """Return the singleton GNN router instance (model loaded once at startup)."""
    return app_state.gnn_router
