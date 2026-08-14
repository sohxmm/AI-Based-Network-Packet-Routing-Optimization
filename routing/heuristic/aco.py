"""
Ant Colony Optimization models route discovery as a colony of ants exploring
paths. Better paths receive more pheromone, making future ants more likely to
choose them. Pheromone evaporation prevents the algorithm from getting stuck on
old choices when network traffic changes.

alpha controls how strongly ants follow pheromones. beta controls how strongly
ants prefer low-cost links. ACO is useful for dynamic multi-path routing because
it keeps exploring alternatives while reinforcing efficient routes.

Unlike the learned routers, ACO has no trained artifact: it searches from
scratch on every call, guided by a pheromone table that persists across calls.
That persistence is why the API holds it as a singleton, and why the benchmark
must build a *fresh* one per run — a shared table would leak state between
experiments and make results depend on call order.
"""

from __future__ import annotations

import random

from core.models import NetworkState, RoutingDecision
from core.paths import build_adjacency, build_decision, failed_decision, path_cost
from core.qos import QoSProfile, evaluate_path
from routing.base import Router


class AntColonyRouter(Router):
    """Route packets using an Ant Colony Optimization search."""

    name = "aco"
    label = "Ant Colony"
    description = (
        "Stochastic metaheuristic. Ants sample routes and reinforce good ones "
        "with pheromone, so it keeps exploring alternatives under congestion."
    )

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.2,
        Q: float = 100,
        n_ants: int = 20,
        n_iterations: int = 30,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.pheromones: dict[frozenset[str], float] = {}
        self.random = random.Random(seed)

    def find_route(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        profile: QoSProfile | None = None,
    ) -> RoutingDecision:
        """Find a path by repeatedly sampling and reinforcing ant routes."""
        profile = self.resolve_profile(profile)

        if src not in state.nodes or dst not in state.nodes:
            return failed_decision(src, dst, self.name)

        adjacency = build_adjacency(state)
        self._ensure_pheromones(state)
        best_path: list[str] = []
        best_cost = float("inf")

        for _ in range(self.n_iterations):
            ant_paths: list[tuple[list[str], float]] = []

            for _ in range(self.n_ants):
                path = self._construct_ant_path(src, dst, adjacency)
                if not path:
                    continue

                # Ants are scored on the class objective, with a penalty for
                # violating a hard constraint, so ACO becomes QoS-aware without
                # changing its search mechanics.
                evaluation = evaluate_path(state, path, profile)
                cost = evaluation.score
                if not evaluation.feasible:
                    cost += _INFEASIBILITY_PENALTY

                ant_paths.append((path, cost))
                if cost < best_cost:
                    best_path = path
                    best_cost = cost

            self._update_pheromones(ant_paths)

        if not best_path:
            return failed_decision(src, dst, self.name)

        return build_decision(
            state,
            src,
            dst,
            best_path,
            self.name,
            diagnostics={"qos": evaluate_path(state, best_path, profile).as_dict()},
        )

    # -- search internals -------------------------------------------------

    def _select_next_node(
        self,
        current: str,
        neighbors: list[tuple[str, float]],
        pheromones: dict[frozenset[str], float],
    ) -> str:
        """Select the next hop using pheromone and inverse-cost probabilities."""
        weights = []
        for neighbor, cost in neighbors:
            edge_key = frozenset((current, neighbor))
            pheromone = pheromones.get(edge_key, 1.0) ** self.alpha
            heuristic = (1.0 / max(cost, 1e-9)) ** self.beta
            weights.append(pheromone * heuristic)

        total = sum(weights)
        if total <= 0:
            return self.random.choice([neighbor for neighbor, _ in neighbors])

        threshold = self.random.uniform(0, total)
        cumulative = 0.0
        for (neighbor, _), weight in zip(neighbors, weights):
            cumulative += weight
            if cumulative >= threshold:
                return neighbor

        return neighbors[-1][0]

    def _update_pheromones(self, ant_paths: list[tuple[list[str], float]]) -> None:
        """Evaporate existing pheromones and deposit new ones on good paths."""
        for edge_key in list(self.pheromones):
            self.pheromones[edge_key] *= 1 - self.evaporation_rate
            self.pheromones[edge_key] = max(self.pheromones[edge_key], 0.01)

        for path, cost in ant_paths:
            deposit = self.Q / max(cost, 1e-9)
            for index in range(len(path) - 1):
                edge_key = frozenset((path[index], path[index + 1]))
                self.pheromones[edge_key] = (
                    self.pheromones.get(edge_key, 1.0) + deposit
                )

    def reset_pheromones(self) -> None:
        """Clear learned pheromone values."""
        self.pheromones.clear()

    def _ensure_pheromones(self, state: NetworkState) -> None:
        """Initialize pheromone entries for every current link."""
        for link in state.links:
            self.pheromones.setdefault(frozenset((link.source, link.target)), 1.0)

    def _construct_ant_path(
        self,
        src: str,
        dst: str,
        adjacency: dict[str, list[tuple[str, float]]],
    ) -> list[str]:
        """Build one ant route without revisiting nodes."""
        path = [src]
        visited = {src}
        current = src

        while current != dst:
            candidates = [
                (neighbor, cost)
                for neighbor, cost in adjacency.get(current, [])
                if neighbor not in visited
            ]
            if not candidates:
                return []

            current = self._select_next_node(current, candidates, self.pheromones)
            path.append(current)
            visited.add(current)

        return path


#: Added to an ant's cost when its path violates a QoS constraint. Large enough
#: that any feasible path outranks any infeasible one, small enough that the
#: search still gradients toward "less badly violating" when nothing is feasible.
_INFEASIBILITY_PENALTY = 1000.0


__all__ = ["AntColonyRouter", "path_cost"]
