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

Dijkstra works well here because all link costs are non-negative. It is also
*provably optimal* for any single additive edge cost, which is the central
reason no learned policy could beat it in the project's original open-loop,
single-objective setting.

Its one structural weakness, which the QoS experiments are designed to expose:
Dijkstra optimises an additive objective and cannot express a **constraint**.
Given a class that forbids paths through links above 70% utilization, Dijkstra
will happily return the cheapest path straight through a saturated link. That
is not a bug in this implementation; it is what the algorithm is.
"""

from __future__ import annotations

import heapq

from core.models import NetworkState, RoutingDecision
from core.paths import build_decision, failed_decision, link_lookup
from core.qos import QoSProfile, qos_link_cost
from routing.base import Router


def find_route(
    state: NetworkState,
    src: str,
    dst: str,
    profile: QoSProfile | None = None,
) -> RoutingDecision:
    """Find the minimum-cost path using Dijkstra's algorithm."""
    profile = Router.resolve_profile(profile)

    if src not in state.nodes or dst not in state.nodes:
        return failed_decision(src, dst, "dijkstra")

    adjacency = _qos_adjacency(state, profile)
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
        return failed_decision(src, dst, "dijkstra")

    path = _reconstruct_path(previous, src, dst)
    return build_decision(state, src, dst, path, "dijkstra")


def _qos_adjacency(
    state: NetworkState, profile: QoSProfile
) -> dict[str, list[tuple[str, float]]]:
    """Adjacency weighted by the class-specific additive cost."""
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in state.nodes}
    for link in state.links:
        weight = qos_link_cost(link, profile)
        adjacency.setdefault(link.source, []).append((link.target, weight))
        adjacency.setdefault(link.target, []).append((link.source, weight))
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


class DijkstraRouter(Router):
    """Class wrapper so the dispatcher sees one uniform interface."""

    name = "dijkstra"
    label = "Dijkstra"
    description = (
        "Exact shortest path over the congestion-adjusted additive cost. "
        "Optimal for the objective, blind to QoS constraints."
    )

    def find_route(self, state, src, dst, profile=None):
        return find_route(state, src, dst, profile)


__all__ = ["DijkstraRouter", "find_route", "link_lookup"]
