"""
Ant Colony Optimization models route discovery as a colony of ants exploring
paths. Better paths receive more pheromone, making future ants more likely to
choose them. Pheromone evaporation prevents the algorithm from getting stuck on
old choices when network traffic changes.

alpha controls how strongly ants follow pheromones. beta controls how strongly
ants prefer low-cost links. ACO is useful for dynamic multi-path routing because
it keeps exploring alternatives while reinforcing efficient routes.
"""

import random

from simulator.data_models import NetworkState, RoutingDecision


class AntColonyRouter:
    """Route packets using an Ant Colony Optimization search."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.2, # tuned for better routing
        Q: float = 100,
        n_ants: int = 20,
        n_iterations: int = 30,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.pheromones: dict[frozenset[str], float] = {}
        self.random = random.Random(42)

    def find_path(self, state: NetworkState, src: str, dst: str) -> RoutingDecision:
        """Find a path by repeatedly sampling and reinforcing ant routes."""
        if src not in state.nodes or dst not in state.nodes:
            return _failed_decision(src, dst)

        adjacency = _build_adjacency(state)
        self._ensure_pheromones(state)
        best_path: list[str] = []
        best_cost = float("inf")

        for _ in range(self.n_iterations):
            ant_paths: list[tuple[list[str], float]] = []

            for _ in range(self.n_ants):
                path = self._construct_ant_path(src, dst, adjacency)
                if not path:
                    continue

                cost = _path_cost(state, path)
                ant_paths.append((path, cost))

                if cost < best_cost:
                    best_path = path
                    best_cost = cost

            self._update_pheromones(ant_paths)

        if not best_path:
            return _failed_decision(src, dst)

        return RoutingDecision(
            source=src,
            destination=dst,
            path=best_path,
            algorithm="aco",
            total_latency=best_cost,
            avg_utilization=_average_path_utilization(state, best_path),
            success=True,
        )

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
        """Evaporate existing pheromones and deposit new pheromones on good paths."""
        for edge_key in list(self.pheromones):
            self.pheromones[edge_key] *= 1 - self.evaporation_rate
            self.pheromones[edge_key] = max(self.pheromones[edge_key], 0.01)

        for path, cost in ant_paths:
            deposit = self.Q / max(cost, 1e-9)
            for index in range(len(path) - 1):
                edge_key = frozenset((path[index], path[index + 1]))
                self.pheromones[edge_key] = self.pheromones.get(edge_key, 1.0) + deposit

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


def _build_adjacency(state: NetworkState) -> dict[str, list[tuple[str, float]]]:
    """Build an undirected adjacency list with congestion-adjusted costs."""
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in state.nodes}
    for link in state.links:
        cost = link.base_latency * (1 + 4 * link.utilization ** 2)
        adjacency[link.source].append((link.target, cost))
        adjacency[link.target].append((link.source, cost))
    return adjacency


def _path_cost(state: NetworkState, path: list[str]) -> float:
    """Calculate total congestion-adjusted latency across a path."""
    lookup = {
        frozenset((link.source, link.target)): link.base_latency * (1 + 4 * link.utilization ** 2)
        for link in state.links
    }
    total = 0.0
    for index in range(len(path) - 1):
        edge_key = frozenset((path[index], path[index + 1]))
        if edge_key not in lookup:
            return float("inf")
        total += lookup[edge_key]
    return total


def _average_path_utilization(state: NetworkState, path: list[str]) -> float:
    """Calculate mean utilization across the selected path."""
    lookup = {
        frozenset((link.source, link.target)): link.utilization
        for link in state.links
    }
    values = [
        lookup[frozenset((path[index], path[index + 1]))]
        for index in range(len(path) - 1)
        if frozenset((path[index], path[index + 1])) in lookup
    ]
    return sum(values) / len(values) if values else 0.0


def _failed_decision(src: str, dst: str) -> RoutingDecision:
    """Create a failed ACO decision."""
    return RoutingDecision(
        source=src,
        destination=dst,
        path=[],
        algorithm="aco",
        total_latency=float("inf"),
        avg_utilization=0.0,
        success=False,
    )
