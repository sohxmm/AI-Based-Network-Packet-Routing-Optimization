"""The one place algorithm names are mapped to implementations.

The API dispatcher, the benchmark harness and the experiment sandbox each used
to keep their own hand-written list. They drifted: the metrics endpoint
validated against six algorithms while its own error message named five, and the
benchmark's list included two predictive variants that had never executed.
"""

from __future__ import annotations

from routing.base import Router
from routing.classical.bellman_ford import BellmanFordRouter
from routing.classical.constrained import ConstrainedRouter
from routing.classical.dijkstra import DijkstraRouter
from routing.heuristic.aco import AntColonyRouter
from routing.learned.gnn import GNNRouter
from routing.learned.multi_agent import MultiAgentRouter
from routing.learned.rl import RLRouter
from routing.random_baseline import RandomBaselineRouter

#: Every algorithm the system can run, in presentation order.
ALGORITHM_NAMES: list[str] = [
    "dijkstra",
    "bellman_ford",
    "constrained",
    "aco",
    "gnn",
    "rl",
    "multi_agent",
    "random_baseline",
]

#: Algorithms that serve a trained artifact and can therefore fall back.
LEARNED_ALGORITHMS: list[str] = ["gnn", "rl", "multi_agent"]

#: Excluded from the degeneracy guardrail. Bellman-Ford is mathematically
#: required to match Dijkstra, and the constrained baseline is *supposed* to
#: agree with it whenever no constraint binds.
DEGENERACY_EXEMPT: set[str] = {"dijkstra", "bellman_ford", "constrained"}


def build_router_set(seed: int = 42, load_models: bool = True) -> dict[str, Router]:
    """Create a fresh, isolated set of routers.

    Every caller that runs an experiment must build its own set. The benchmark
    previously acquired the API's singletons at module import, so running a
    sandbox experiment permanently mutated the live dashboard's ACO pheromone
    table and made results depend on call order.
    """
    aco = AntColonyRouter(seed=seed)
    rl = RLRouter(seed=seed)
    gnn = GNNRouter(seed=seed)
    marl = MultiAgentRouter(seed=seed)

    if load_models:
        rl.try_load_model()
        gnn.try_load_model()
        marl.try_load_models()

    return {
        "dijkstra": DijkstraRouter(),
        "bellman_ford": BellmanFordRouter(),
        "constrained": ConstrainedRouter(),
        "aco": aco,
        "gnn": gnn,
        "rl": rl,
        "multi_agent": marl,
        "random_baseline": RandomBaselineRouter(seed=seed),
    }


def describe_algorithms() -> list[dict[str, object]]:
    """Metadata for the dashboard's algorithm picker."""
    routers = build_router_set(load_models=False)
    return [
        {
            "name": name,
            "label": routers[name].label,
            "description": routers[name].description,
            "learned": name in LEARNED_ALGORITHMS,
        }
        for name in ALGORITHM_NAMES
    ]


__all__ = [
    "ALGORITHM_NAMES",
    "DEGENERACY_EXEMPT",
    "LEARNED_ALGORITHMS",
    "build_router_set",
    "describe_algorithms",
]
