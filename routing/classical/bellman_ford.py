"""
Bellman-Ford is slower than Dijkstra because it relaxes every edge V-1 times,
giving it O(VE) time complexity. Dijkstra with a heap is usually O((V + E)logV)
for non-negative weights.

Bellman-Ford is still valuable in distributed and dynamic routing systems
because it naturally models repeated neighbor-to-neighbor relaxation and can
detect negative-weight cycles. This project uses non-negative congestion costs,
but the cycle check is included for learning and correctness.

**How to read its benchmark row.** With identical weights and non-negative
costs, Bellman-Ford and Dijkstra are both exact, so they necessarily return the
same cost — we measure ``dijkstra_match_rate = 1.000`` in every
scenario. That is mathematically required, not a defect. Bellman-Ford is
therefore reported as a *correctness cross-check on Dijkstra*, not as an
independent baseline, and it is excluded from the degeneracy guardrail for
exactly that reason.
"""

from __future__ import annotations

from core.models import NetworkState, RoutingDecision
from core.paths import build_decision, failed_decision
from core.qos import QoSProfile, qos_link_cost
from routing.base import Router


def find_route(
    state: NetworkState,
    src: str,
    dst: str,
    profile: QoSProfile | None = None,
) -> RoutingDecision:
    """Find a minimum-cost path using the Bellman-Ford algorithm."""
    profile = Router.resolve_profile(profile)

    if src not in state.nodes or dst not in state.nodes:
        return failed_decision(src, dst, "bellman_ford")

    edges = _build_edges(state, profile)
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

    # Negative-cycle detection. Unreachable with this cost model, kept for
    # correctness and because it is the algorithm's defining capability.
    for start, end, weight in edges:
        if distances[start] + weight < distances[end]:
            return failed_decision(src, dst, "bellman_ford")

    if distances[dst] == float("inf"):
        return failed_decision(src, dst, "bellman_ford")

    path = _reconstruct_path(previous, src, dst)
    return build_decision(state, src, dst, path, "bellman_ford")


def _build_edges(
    state: NetworkState, profile: QoSProfile
) -> list[tuple[str, str, float]]:
    """Build directed edge pairs for an undirected network state."""
    edges: list[tuple[str, str, float]] = []
    for link in state.links:
        weight = qos_link_cost(link, profile)
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


class BellmanFordRouter(Router):
    """Class wrapper so the dispatcher sees one uniform interface."""

    name = "bellman_ford"
    label = "Bellman-Ford"
    description = (
        "Exact shortest path by repeated edge relaxation. Reported as a "
        "correctness cross-check on Dijkstra, not an independent baseline."
    )

    def find_route(self, state, src, dst, profile=None):
        return find_route(state, src, dst, profile)


__all__ = ["BellmanFordRouter", "find_route"]
