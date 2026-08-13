"""network_partition.py — Partition a 25-node topology into 3–4 regions.

Strategy:
  1. Try networkx greedy_modularity_communities() for a data-driven split.
  2. If the result has fewer than 3 or more than 5 communities (unstable on
     small graphs), fall back to a deterministic bucket split.

Exports:
  partition_network(graph) -> dict[int, list[str]]
  get_node_region(partition, node) -> int
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx


def partition_network(
    graph: nx.Graph,
    min_regions: int = 3,
    max_regions: int = 4,
) -> Dict[int, List[str]]:
    """Partition *graph* into regions and return {region_id: [node, …]}.

    Uses greedy modularity maximisation when the resulting number of
    communities falls within [min_regions, max_regions].  Otherwise
    falls back to a stable deterministic bucket split so that results
    are reproducible regardless of networkx version quirks.
    """
    nodes = sorted(graph.nodes())
    if len(nodes) == 0:
        return {}

    # --- Attempt community detection ------------------------------------
    try:
        communities = list(
            nx.community.greedy_modularity_communities(graph)
        )
        if min_regions <= len(communities) <= max_regions:
            partition: Dict[int, List[str]] = {}
            for region_id, community in enumerate(communities):
                partition[region_id] = sorted(community)
            print(
                f"[Partition] Greedy modularity produced {len(partition)} "
                f"regions: {[len(v) for v in partition.values()]} nodes each"
            )
            return partition
        else:
            print(
                f"[Partition] Modularity gave {len(communities)} communities "
                f"(outside [{min_regions}, {max_regions}]); using bucket split"
            )
    except Exception as exc:
        print(f"[Partition] Community detection failed ({exc}); using bucket split")

    # --- Deterministic bucket fallback -----------------------------------
    n_regions = min(max_regions, max(min_regions, 3))
    bucket_size = len(nodes) // n_regions
    partition = {}
    for i in range(n_regions):
        start = i * bucket_size
        end = start + bucket_size if i < n_regions - 1 else len(nodes)
        partition[i] = nodes[start:end]

    print(
        f"[Partition] Bucket split into {n_regions} regions: "
        f"{[len(v) for v in partition.values()]} nodes each"
    )
    return partition


def get_node_region(
    partition: Dict[int, List[str]],
    node: str,
) -> int:
    """Return the region id that *node* belongs to, or -1 if unknown."""
    for region_id, members in partition.items():
        if node in members:
            return region_id
    return -1


def build_region_lookup(
    partition: Dict[int, List[str]],
) -> Dict[str, int]:
    """Invert the partition dict into {node: region_id} for O(1) lookups."""
    lookup: Dict[str, int] = {}
    for region_id, members in partition.items():
        for node in members:
            lookup[node] = region_id
    return lookup
