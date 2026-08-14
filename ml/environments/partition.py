"""Partition a topology into routing regions, one per multi-agent policy.

Strategy:
  1. Try networkx ``greedy_modularity_communities()`` for a data-driven split.
  2. If the result falls outside the requested region count (community detection
     is unstable on small graphs and varies between networkx versions), fall
     back to a deterministic bucket split so results stay reproducible.

This module was always sound — it was what wrapped it that was broken. The router used to derive regions from a *freshly built
25-node simulator* regardless of the topology it was actually serving, so on the
100-node scenario three quarters of all nodes mapped to region -1 and forced a
heuristic fallback. Partitioning now always runs against the live graph.
"""

from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def partition_network(
    graph: nx.Graph,
    min_regions: int = 3,
    max_regions: int = 5,
) -> dict[int, list[str]]:
    """Partition *graph* into regions and return ``{region_id: [node, ...]}``."""
    nodes = sorted(graph.nodes())
    if not nodes:
        return {}

    if len(nodes) < min_regions:
        return {0: nodes}

    try:
        communities = list(nx.community.greedy_modularity_communities(graph))
        if min_regions <= len(communities) <= max_regions:
            partition = {
                region_id: sorted(community)
                for region_id, community in enumerate(communities)
            }
            logger.debug(
                "Modularity split: %d regions, sizes %s",
                len(partition),
                [len(v) for v in partition.values()],
            )
            return partition
        logger.debug(
            "Modularity gave %d communities (outside [%d, %d]); using bucket split",
            len(communities),
            min_regions,
            max_regions,
        )
    except Exception as exc:  # noqa: BLE001 - networkx raises several types here
        logger.debug("Community detection failed (%s); using bucket split", exc)

    # Deterministic fallback: contiguous buckets over the sorted node list.
    n_regions = max(min_regions, min(max_regions, max(1, len(nodes) // 10)))
    bucket_size = max(1, len(nodes) // n_regions)
    partition: dict[int, list[str]] = {}
    for index in range(n_regions):
        start = index * bucket_size
        end = start + bucket_size if index < n_regions - 1 else len(nodes)
        if start < len(nodes):
            partition[index] = nodes[start:end]

    logger.debug(
        "Bucket split: %d regions, sizes %s",
        len(partition),
        [len(v) for v in partition.values()],
    )
    return partition


def get_node_region(partition: dict[int, list[str]], node: str) -> int:
    """Return the region id that *node* belongs to, or -1 if unknown."""
    for region_id, members in partition.items():
        if node in members:
            return region_id
    return -1


def build_region_lookup(partition: dict[int, list[str]]) -> dict[str, int]:
    """Invert the partition into ``{node: region_id}`` for O(1) lookups."""
    return {
        node: region_id
        for region_id, members in partition.items()
        for node in members
    }


__all__ = ["build_region_lookup", "get_node_region", "partition_network"]
