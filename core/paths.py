"""Shared path primitives: candidate generation, costing and decision building.

These replace six near-identical private copies that used to live inside each
router module. Two behaviours here are deliberate corrections of the originals:

* :func:`path_cost` returns ``inf`` for a path that traverses a non-existent
  edge. The old per-router copies filtered missing edges out of the sum, which
  reported a *lower* cost for an invalid path and biased results toward the
  learned routers.
* :func:`candidate_paths` is weighted by :func:`core.cost.link_cost` by default.
  Training used a weighted generator while inference used an unweighted one, so
  action index *k* meant a different path in training than in serving.
"""

from __future__ import annotations

import networkx as nx

from core.cost import link_cost
from core.models import LinkState, NetworkState, RoutingDecision


def link_lookup(state: NetworkState) -> dict[frozenset[str], LinkState]:
    """Index the links of *state* by their unordered endpoint pair."""
    return {frozenset((link.source, link.target)): link for link in state.links}


def build_adjacency(state: NetworkState) -> dict[str, list[tuple[str, float]]]:
    """Undirected adjacency list with congestion-adjusted edge weights."""
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in state.nodes}
    for link in state.links:
        cost = link_cost(link)
        adjacency.setdefault(link.source, []).append((link.target, cost))
        adjacency.setdefault(link.target, []).append((link.source, cost))
    return adjacency


def build_graph(state: NetworkState) -> nx.Graph:
    """Build a weighted networkx graph from *state* (weight key ``w``)."""
    graph = nx.Graph()
    graph.add_nodes_from(state.nodes)
    for link in state.links:
        graph.add_edge(link.source, link.target, w=link_cost(link))
    return graph


def path_links(state: NetworkState, path: list[str]) -> list[LinkState] | None:
    """Links along *path*, or ``None`` if any hop does not exist in *state*."""
    if len(path) < 2:
        return None
    lookup = link_lookup(state)
    links: list[LinkState] = []
    for index in range(len(path) - 1):
        key = frozenset((path[index], path[index + 1]))
        if key not in lookup:
            return None
        links.append(lookup[key])
    return links


def path_cost(state: NetworkState, path: list[str]) -> float:
    """Total congestion-adjusted latency. Returns ``inf`` for an invalid path."""
    links = path_links(state, path)
    return float("inf") if links is None else sum(link_cost(link) for link in links)


def average_path_utilization(state: NetworkState, path: list[str]) -> float:
    """Mean utilization across the links of *path* (0.0 when the path is invalid)."""
    links = path_links(state, path)
    if not links:
        return 0.0
    return sum(link.utilization for link in links) / len(links)


def max_path_utilization(state: NetworkState, path: list[str]) -> float:
    """Utilization of the single busiest link on *path* (its bottleneck)."""
    links = path_links(state, path)
    if not links:
        return 0.0
    return max(link.utilization for link in links)


def candidate_paths(
    state: NetworkState,
    src: str,
    dst: str,
    k: int = 5,
    weighted: bool = True,
) -> list[list[str]]:
    """Up to *k* loopless paths, ordered by congestion-adjusted cost.

    ``weighted=True`` must be used everywhere. It is the ordering the learned
    policies were trained against, so action index *k* refers to the same path
    at training time and at serving time.
    """
    if src == dst:
        return []
    graph = build_graph(state)
    if src not in graph or dst not in graph:
        return []
    if not nx.has_path(graph, src, dst):
        return []
    generator = nx.shortest_simple_paths(graph, src, dst, weight="w" if weighted else None)
    return [path for _, path in zip(range(k), generator)]


def failed_decision(src: str, dst: str, algorithm: str) -> RoutingDecision:
    """A decision recording that *algorithm* could not route src -> dst.

    ``is_fallback`` is True because no model produced this answer; leaving it
    False (as the six old copies did) made failed AI decisions look like
    genuine model output in the fallback-rate guardrail.
    """
    return RoutingDecision(
        source=src,
        destination=dst,
        path=[],
        algorithm=algorithm,
        total_latency=float("inf"),
        avg_utilization=0.0,
        success=False,
        is_fallback=True,
    )


def build_decision(
    state: NetworkState,
    src: str,
    dst: str,
    path: list[str],
    algorithm: str,
    is_fallback: bool = False,
    diagnostics: dict | None = None,
) -> RoutingDecision:
    """Build a populated :class:`RoutingDecision` for *path*."""
    if not path:
        return failed_decision(src, dst, algorithm)
    return RoutingDecision(
        source=src,
        destination=dst,
        path=path,
        algorithm=algorithm,
        total_latency=path_cost(state, path),
        avg_utilization=average_path_utilization(state, path),
        success=True,
        is_fallback=is_fallback,
        diagnostics=diagnostics or {},
    )


def hop_breakdown(state: NetworkState, path: list[str]) -> list[dict[str, float | str]]:
    """Per-hop cost breakdown, used by the dashboard to explain a decision."""
    links = path_links(state, path)
    if not links:
        return []
    rows: list[dict[str, float | str]] = []
    for index, link in enumerate(links):
        rows.append(
            {
                "from": path[index],
                "to": path[index + 1],
                "base_latency": round(float(link.base_latency), 2),
                "utilization": round(float(link.utilization), 4),
                "cost": round(link_cost(link), 2),
            }
        )
    return rows
