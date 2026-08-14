"""Observation builders for genuinely decentralized multi-agent routing.

The old "MARL" was nothing of the kind: N independent PPO agents, *each
receiving the full global observation*, *each emitting a complete end-to-end
path*, selected by a lookup on the source node. No joint action space, no
communication, no credit assignment. That is a mixture-of-experts with a
hardcoded gating function, and describing it as centralized-critic /
decentralized-execution was the claim least likely to survive a viva.

This module implements the observation half of the fix.

**Decentralized execution.** An agent observes only its own neighbourhood: which
node the packet is at, where it is going, and the state of that node's incident
links. The width of that vector is a constant — it does not depend on the total
number of links or nodes in the network. ``tests/unit/ml/test_marl_locality.py``
asserts exactly that, because it is the property that makes the claim true.

**Centralized training.** A fixed-width *summary* of the global network is
appended for the critic only. The actor's feature extractor slices it off, so
the value function can use global information the policy never sees. That
asymmetry is the actual CTDE claim, and it is verified by a test that perturbs
the global block and asserts the action distribution does not move.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from core.cost import MAX_BASE_LATENCY, MAX_LOSS, MAX_QUEUE
from core.models import NetworkState
from core.qos import PROFILE_VECTOR_DIM, QoSProfile, get_profile, profile_vector

# --- fixed layout constants -------------------------------------------------
#: Maximum neighbours an agent can choose between. Nodes with a higher degree
#: expose their MAX_DEGREE cheapest incident links.
MAX_DEGREE = 8
#: Padding width for the "which node am I at" one-hot within a region.
MAX_REGION_SIZE = 40
#: Per-neighbour features, see build_local_observation.
NEIGHBOUR_FEATS = 8
#: Features describing the destination relative to the current node.
DEST_FEATS = 3

LOCAL_DIM = (
    MAX_REGION_SIZE + DEST_FEATS + MAX_DEGREE * NEIGHBOUR_FEATS + PROFILE_VECTOR_DIM
)
#: Fixed-width global summary, used by the critic only.
GLOBAL_DIM = 16
OBS_DIM = LOCAL_DIM + GLOBAL_DIM


def hop_distances(graph: nx.Graph, destination: str) -> dict[str, int]:
    """Unweighted hop counts from every node to *destination*."""
    if destination not in graph:
        return {}
    return nx.single_source_shortest_path_length(graph, destination)


def neighbours_of(
    state_graph: nx.Graph, node: str, limit: int = MAX_DEGREE
) -> list[str]:
    """Deterministic neighbour ordering: cheapest incident links first.

    Ordering must be identical in training and serving, otherwise action index
    *k* means a different next hop in each — the same class of bug that
    invalidated the single-agent PPO results.
    """
    if node not in state_graph:
        return []
    neighbours = sorted(
        state_graph.neighbors(node),
        key=lambda other: (state_graph[node][other].get("w", 0.0), other),
    )
    return neighbours[:limit]


def build_local_observation(
    state: NetworkState,
    graph: nx.Graph,
    link_map: dict[frozenset[str], object],
    current: str,
    destination: str,
    region_nodes: list[str],
    distances: dict[str, int],
    region_of: dict[str, int],
    profile: QoSProfile | None = None,
) -> np.ndarray:
    """Build one agent's purely local view. Width is always ``LOCAL_DIM``."""
    profile = profile if profile is not None else get_profile(None)
    obs = np.zeros(LOCAL_DIM, dtype=np.float32)

    # Block 1: which node in my region the packet is currently at.
    if current in region_nodes:
        index = region_nodes.index(current)
        if index < MAX_REGION_SIZE:
            obs[index] = 1.0
    cursor = MAX_REGION_SIZE

    # Block 2: where the packet is going, expressed relatively.
    my_region = region_of.get(current, -1)
    current_distance = distances.get(current, 99)
    obs[cursor + 0] = 1.0 if destination == current else 0.0
    obs[cursor + 1] = min(1.0, current_distance / 20.0)
    obs[cursor + 2] = 1.0 if region_of.get(destination, -2) == my_region else 0.0
    cursor += DEST_FEATS

    # Block 3: the incident links this agent can actually choose between.
    for slot, neighbour in enumerate(neighbours_of(graph, current)):
        link = link_map.get(frozenset((current, neighbour)))
        if link is None:
            continue
        base = cursor + slot * NEIGHBOUR_FEATS
        neighbour_distance = distances.get(neighbour, 99)
        obs[base + 0] = 1.0  # this action slot is usable
        obs[base + 1] = float(np.clip(link.utilization, 0.0, 1.0))
        obs[base + 2] = float(np.clip(link.queue_size / MAX_QUEUE, 0.0, 1.0))
        obs[base + 3] = float(np.clip(link.packet_loss_rate / MAX_LOSS, 0.0, 1.0))
        obs[base + 4] = float(np.clip(link.base_latency / MAX_BASE_LATENCY, 0.0, 1.0))
        obs[base + 5] = 1.0 if neighbour == destination else 0.0
        obs[base + 6] = 1.0 if region_of.get(neighbour, -2) == my_region else 0.0
        # Does this hop take us closer? The one piece of routing knowledge an
        # agent genuinely needs and cannot derive from its own links alone.
        obs[base + 7] = 1.0 if neighbour_distance < current_distance else 0.0
    cursor += MAX_DEGREE * NEIGHBOUR_FEATS

    # Block 4: which traffic class we are serving.
    obs[cursor : cursor + PROFILE_VECTOR_DIM] = np.asarray(
        profile_vector(profile), dtype=np.float32
    )
    return obs


def build_global_summary(state: NetworkState) -> np.ndarray:
    """Fixed-width global network summary for the centralized critic.

    Deliberately a *summary* rather than the raw link vector: its width must not
    depend on the topology, or the policy could not transfer between the 25-node
    and 100-node networks.
    """
    summary = np.zeros(GLOBAL_DIM, dtype=np.float32)
    if not state.links:
        return summary

    utils = np.asarray([link.utilization for link in state.links], dtype=np.float32)
    losses = np.asarray([link.packet_loss_rate for link in state.links], dtype=np.float32)
    queues = np.asarray([link.queue_size for link in state.links], dtype=np.float32)

    # A 6-bin histogram of utilization: the shape of the load distribution is
    # what a load-balancing critic needs, and it is scale free.
    histogram, _ = np.histogram(utils, bins=6, range=(0.0, 1.0))
    summary[0:6] = histogram / max(1, len(utils))

    summary[6] = float(utils.mean())
    summary[7] = float(utils.max())
    summary[8] = float(utils.min())
    summary[9] = float(utils.var())
    summary[10] = float(np.percentile(utils, 95))
    summary[11] = float(np.clip(queues.mean() / MAX_QUEUE, 0.0, 1.0))
    summary[12] = float(np.clip(losses.mean() / MAX_LOSS, 0.0, 1.0))
    summary[13] = float((utils > 0.7).mean())
    summary[14] = float((utils > 0.9).mean())
    summary[15] = float(min(1.0, len(state.links) / 200.0))
    return summary


def build_agent_observation(
    state: NetworkState,
    graph: nx.Graph,
    link_map: dict[frozenset[str], object],
    current: str,
    destination: str,
    region_nodes: list[str],
    distances: dict[str, int],
    region_of: dict[str, int],
    profile: QoSProfile | None = None,
    include_global: bool = True,
) -> np.ndarray:
    """Concatenate the local view and the global summary into one observation.

    ``include_global=False`` zeroes the critic block, which is what the router
    does at serving time: execution is decentralized, so the actor must produce
    the same action whether or not global information is available.
    """
    local = build_local_observation(
        state, graph, link_map, current, destination, region_nodes, distances, region_of, profile
    )
    global_block = (
        build_global_summary(state)
        if include_global
        else np.zeros(GLOBAL_DIM, dtype=np.float32)
    )
    return np.concatenate([local, global_block]).astype(np.float32)


def phase_features(step_count: int, period: int = 40) -> tuple[float, float]:
    """Sine/cosine encoding of position in the diurnal cycle."""
    angle = 2 * math.pi * (step_count % period) / period
    return math.sin(angle), math.cos(angle)


__all__ = [
    "DEST_FEATS",
    "GLOBAL_DIM",
    "LOCAL_DIM",
    "MAX_DEGREE",
    "MAX_REGION_SIZE",
    "NEIGHBOUR_FEATS",
    "OBS_DIM",
    "build_agent_observation",
    "build_global_summary",
    "build_local_observation",
    "hop_distances",
    "neighbours_of",
    "phase_features",
]
