"""
Dijkstra's algorithm, step by step:

1. Build an adjacency list from the current NetworkState.
2. Start at the source node with cost 0.
3. Keep a min-heap of the cheapest frontier paths seen so far.
4. Repeatedly expand the currently cheapest node.
5. When a neighbor can be reached with a lower cost, update its best known
   cost and remember its parent.
6. Stop when the destination is reached, then reconstruct the path by walking
   backward through the parent map.

Dijkstra works well here because all link costs are non-negative.
"""

import heapq

from simulator.data_models import NetworkState, RoutingDecision


def find_route(state: NetworkState, src: str, dst: str) -> RoutingDecision:
    """Find the minimum-cost path using Dijkstra's algorithm."""
    adjacency = _build_adjacency(state)

    if src not in state.nodes or dst not in state.nodes:
        return _failed_decision(src, dst)

    distances = {node: float("inf") for node in state.nodes}
    previous: dict[str, str | None] = {node: None for node in state.nodes}
    distances[src] = 0.0
    heap: list[tuple[float, str]] = [(0.0, src)]
    visited: set[str] = set()

    while heap:
        current_cost, current = heapq.heappop(heap)

        if current in visited:
            continue

        visited.add(current)

        if current == dst:
            break

        for neighbor, weight in adjacency.get(current, []):
            new_cost = current_cost + weight
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                previous[neighbor] = current
                heapq.heappush(heap, (new_cost, neighbor))

    if distances[dst] == float("inf"):
        return _failed_decision(src, dst)

    path = _reconstruct_path(previous, src, dst)
    return RoutingDecision(
        source=src,
        destination=dst,
        path=path,
        algorithm="dijkstra",
        total_latency=distances[dst],
        avg_utilization=_average_path_utilization(state, path),
        success=True,
    )


def _build_adjacency(state: NetworkState) -> dict[str, list[tuple[str, float]]]:
    """Build an undirected adjacency list with congestion-adjusted weights."""
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in state.nodes}
    for link in state.links:
        weight = link.base_latency * (1 + 4 * link.utilization ** 2)
        adjacency[link.source].append((link.target, weight))
        adjacency[link.target].append((link.source, weight))
    return adjacency


def _reconstruct_path(previous: dict[str, str | None], src: str, dst: str) -> list[str]:
    """Rebuild a path from destination to source using parent pointers."""
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
    """Create a failed Dijkstra decision."""
    return RoutingDecision(
        source=src,
        destination=dst,
        path=[],
        algorithm="dijkstra",
        total_latency=float("inf"),
        avg_utilization=0.0,
        success=False,
    )


def _average_path_utilization(state: NetworkState, path: list[str]) -> float:
    """Calculate mean utilization across the links used in a path."""
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
