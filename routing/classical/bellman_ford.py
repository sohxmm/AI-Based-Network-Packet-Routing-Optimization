"""
Bellman-Ford is slower than Dijkstra because it relaxes every edge V-1 times,
giving it O(VE) time complexity. Dijkstra with a heap is usually O((V + E)logV)
for non-negative weights.

Bellman-Ford is still valuable in distributed and dynamic routing systems
because it naturally models repeated neighbor-to-neighbor relaxation and can
detect negative-weight cycles. This project uses non-negative congestion costs,
but the cycle check is included for learning and correctness.
"""

from simulator.data_models import NetworkState, RoutingDecision


def find_route(state: NetworkState, src: str, dst: str) -> RoutingDecision:
    """Find a minimum-cost path using the Bellman-Ford algorithm."""
    if src not in state.nodes or dst not in state.nodes:
        return _failed_decision(src, dst)

    edges = _build_edges(state)
    distances = {node: float("inf") for node in state.nodes}
    previous: dict[str, str | None] = {node: None for node in state.nodes}
    distances[src] = 0.0

    for _ in range(len(state.nodes) - 1):
        changed = False
        for start, end, weight in edges:
            if distances[start] + weight < distances[end]:
                distances[end] = distances[start] + weight
                previous[end] = start
                changed = True
        if not changed:
            break

    for start, end, weight in edges:
        if distances[start] + weight < distances[end]:
            return _failed_decision(src, dst)

    if distances[dst] == float("inf"):
        return _failed_decision(src, dst)

    path = _reconstruct_path(previous, src, dst)
    return RoutingDecision(
        source=src,
        destination=dst,
        path=path,
        algorithm="bellman_ford",
        total_latency=distances[dst],
        avg_utilization=_average_path_utilization(state, path),
        success=True,
    )


def _build_edges(state: NetworkState) -> list[tuple[str, str, float]]:
    """Build directed edge pairs for an undirected network state."""
    edges: list[tuple[str, str, float]] = []
    for link in state.links:
        weight = link.base_latency * (1 + 4 * link.utilization ** 2)
        edges.append((link.source, link.target, weight))
        edges.append((link.target, link.source, weight))
    return edges


def _reconstruct_path(previous: dict[str, str | None], src: str, dst: str) -> list[str]:
    """Rebuild a path by following Bellman-Ford predecessor pointers."""
    path = [dst]
    current = dst
    while current != src:
        parent = previous[current]
        if parent is None:
            return []
        path.append(parent)
        current = parent
    path.reverse()
    return path


def _failed_decision(src: str, dst: str) -> RoutingDecision:
    """Create a failed Bellman-Ford decision."""
    return RoutingDecision(
        source=src,
        destination=dst,
        path=[],
        algorithm="bellman_ford",
        total_latency=float("inf"),
        avg_utilization=0.0,
        success=False,
    )


def _average_path_utilization(state: NetworkState, path: list[str]) -> float:
    """Calculate mean utilization across the selected path."""
    if len(path) < 2:
        return 0.0

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
