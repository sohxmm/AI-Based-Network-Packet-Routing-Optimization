import random
from pathlib import Path

from simulator.data_models import NetworkState, RoutingDecision


class RLRouter:
    """Placeholder RL router that falls back to a random valid path."""

    def __init__(self, seed: int = 42) -> None:
        self.model_path: Path | None = None
        self.model_loaded = False
        self.random = random.Random(seed)

    def load_model(self, path: str) -> None:
        """Mark a future trained model as loaded if its file exists."""
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"RL model not found: {path}")

        self.model_path = candidate
        self.model_loaded = True

    @property
    def is_trained(self) -> bool:
        """Return whether a trained model has been loaded."""
        return self.model_loaded

    def predict(self, state: NetworkState, src: str, dst: str) -> RoutingDecision:
        """Return a model decision or a random valid fallback path."""
        if src not in state.nodes or dst not in state.nodes:
            return self._failed_decision(src, dst)

        candidate_paths = _find_candidate_paths(state, src, dst)
        if not candidate_paths:
            return self._failed_decision(src, dst)

        path = self.random.choice(candidate_paths)
        return RoutingDecision(
            source=src,
            destination=dst,
            path=path,
            algorithm="rl",
            total_latency=_path_cost(state, path),
            avg_utilization=_average_path_utilization(state, path),
            success=True,
        )

    def _failed_decision(self, src: str, dst: str) -> RoutingDecision:
        """Create a failed RL placeholder decision."""
        return RoutingDecision(
            source=src,
            destination=dst,
            path=[],
            algorithm="rl",
            total_latency=float("inf"),
            avg_utilization=0.0,
            success=False,
        )


def _find_candidate_paths(state: NetworkState, src: str, dst: str, limit: int = 10) -> list[list[str]]:
    """Find simple valid paths with breadth-first search."""
    adjacency = _build_adjacency(state)
    queue: list[list[str]] = [[src]]
    paths: list[list[str]] = []

    while queue and len(paths) < limit:
        path = queue.pop(0)
        current = path[-1]

        if current == dst:
            paths.append(path)
            continue

        for neighbor in adjacency.get(current, []):
            if neighbor not in path:
                queue.append([*path, neighbor])

    return paths


def _build_adjacency(state: NetworkState) -> dict[str, list[str]]:
    """Build an undirected adjacency list."""
    adjacency: dict[str, list[str]] = {node: [] for node in state.nodes}
    for link in state.links:
        adjacency[link.source].append(link.target)
        adjacency[link.target].append(link.source)
    return adjacency


def _path_cost(state: NetworkState, path: list[str]) -> float:
    """Calculate total congestion-adjusted latency across a path."""
    lookup = {
        frozenset((link.source, link.target)): link.base_latency * (1 + 4 * link.utilization ** 2)
        for link in state.links
    }
    return sum(
        lookup[frozenset((path[index], path[index + 1]))]
        for index in range(len(path) - 1)
    )


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
