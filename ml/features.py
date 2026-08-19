"""Feature builders shared by training and inference.

Every train/serve skew bug came from the same root cause: the
training script and the router each built their own inputs. Three different
candidate-path generators existed, and the RL observation was constructed twice
from two different code paths. Action index *k* therefore meant a different path
in training than in serving, which silently invalidates every learned result.

There is now exactly one builder for the GNN's graph tensors and exactly one for
the PPO observation vector, and ``tests/unit/ml/test_train_serve_parity.py``
asserts that the environment and the router produce identical arrays.
"""

from __future__ import annotations

import numpy as np
import torch

from core.cost import MAX_BASE_LATENCY, MAX_LATENCY_MS, MAX_LOSS, MAX_QUEUE, link_cost
from core.models import NetworkState
from core.paths import link_lookup
from core.qos import PROFILE_VECTOR_DIM, QoSProfile, get_profile, profile_vector

# --- RL observation layout -------------------------------------------------
K_PATHS = 5
#: [valid, hops_norm, cost_norm, mean_util, max_util, mean_loss]
PATH_FEATS = 6
LINK_FEATS = 4
#: Width of the QoS conditioning block appended to the PPO observation.
QOS_FEATS = PROFILE_VECTOR_DIM


# ===========================================================================
# GNN graph tensors
# ===========================================================================
def build_graph_tensors(
    state: NetworkState,
    candidate_paths: list[list[str]],
    src: str,
    dst: str,
    profile: QoSProfile | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[list[int]], torch.Tensor]:
    """Build every tensor :class:`ml.architectures.gnn.GNNRouterModel` needs.

    Returns ``(x, edge_index, edge_attr, paths_idx, path_edges, path_feats)``.

    ``path_edges`` indexes rows of ``edge_attr``. Every undirected link
    contributes two rows (u->v and v->u); the forward row is recorded for each
    link so the path scorer pools over exactly the links it traverses.
    """
    nodes = state.nodes
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # -- node features: [is_source, is_destination, normalised degree] -------
    degrees = [0] * len(nodes)
    for link in state.links:
        degrees[node_to_idx[link.source]] += 1
        degrees[node_to_idx[link.target]] += 1
    max_degree = max(1, *degrees) if degrees else 1

    x = torch.tensor(
        [
            [
                1.0 if node == src else 0.0,
                1.0 if node == dst else 0.0,
                float(degrees[i]) / max_degree,
            ]
            for i, node in enumerate(nodes)
        ],
        dtype=torch.float32,
    )

    # -- edges: both directions, plus a key -> forward-row lookup -----------
    edges_src: list[int] = []
    edges_dst: list[int] = []
    edge_attr_rows: list[list[float]] = []
    edge_row_of: dict[frozenset[str], int] = {}

    for link in state.links:
        u = node_to_idx[link.source]
        v = node_to_idx[link.target]
        attr = [
            float(link.utilization),
            float(link.queue_size) / MAX_QUEUE,
            float(link.packet_loss_rate) / MAX_LOSS,
            float(link.base_latency) / MAX_BASE_LATENCY,
        ]
        edge_row_of[frozenset((link.source, link.target))] = len(edge_attr_rows)
        edges_src.append(u)
        edges_dst.append(v)
        edge_attr_rows.append(attr)
        edges_src.append(v)
        edges_dst.append(u)
        edge_attr_rows.append(attr)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)

    # -- per-path node indices, edge rows and explicit features -------------
    lookup = link_lookup(state)
    qos_block = profile_vector(get_profile(None) if profile is None else profile)
    paths_idx: list[list[int]] = []
    path_edges: list[list[int]] = []
    feats: list[list[float]] = []

    for path in candidate_paths:
        paths_idx.append([node_to_idx[n] for n in path if n in node_to_idx])

        rows: list[int] = []
        links = []
        for i in range(len(path) - 1):
            key = frozenset((path[i], path[i + 1]))
            if key in edge_row_of:
                rows.append(edge_row_of[key])
                links.append(lookup[key])
        path_edges.append(rows)

        if links:
            utils = [link.utilization for link in links]
            feats.append(
                [
                    min(1.0, len(links) / 10.0),
                    float(np.mean(utils)),
                    float(max(utils)),
                    *qos_block,
                ]
            )
        else:
            feats.append([0.0, 0.0, 0.0, *qos_block])

    width = 3 + len(qos_block)
    path_feats = (
        torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros((0, width))
    )
    return x, edge_index, edge_attr, paths_idx, path_edges, path_feats


# ===========================================================================
# PPO observation
# ===========================================================================
def path_features(state: NetworkState, path: list[str]) -> list[float]:
    """Six normalised features describing one candidate path."""
    lookup = link_lookup(state)
    links = [
        lookup[frozenset((path[i], path[i + 1]))]
        for i in range(len(path) - 1)
        if frozenset((path[i], path[i + 1])) in lookup
    ]
    if not links:
        return [0.0] * PATH_FEATS

    cost = sum(link_cost(link) for link in links)
    return [
        1.0,
        min(1.0, len(links) / 10.0),
        min(1.0, cost / MAX_LATENCY_MS),
        float(np.mean([link.utilization for link in links])),
        float(max(link.utilization for link in links)),
        float(np.mean([link.packet_loss_rate for link in links]) / MAX_LOSS),
    ]


def observation_dim(n_links: int, n_nodes: int) -> int:
    """Total width of the PPO observation vector."""
    return n_links * LINK_FEATS + n_nodes * 2 + K_PATHS * PATH_FEATS + QOS_FEATS


def build_observation(
    state: NetworkState,
    n_links: int,
    n_nodes: int,
    src_idx: int,
    dst_idx: int,
    paths: list[list[str]],
    profile: QoSProfile | None = None,
) -> np.ndarray:
    """Build the PPO observation: link state, the task, and the choices.

    The original observation encoded per-link features only. It omitted the
    (source, destination) pair while the environment resampled the routing task
    every step, so the meaning of "action 2" changed completely between steps.
    That is not a partially observable MDP, it is an unobservable one, and it
    fully explains the flat learning curve we measured (r-squared 0.001).

    Three blocks, all in [0, 1]:

    1. per-link state    ``n_links * 4``
    2. the task          ``n_nodes * 2``   one-hot source, one-hot destination
    3. the choices       ``K_PATHS * 6``   features of each candidate path
    4. the QoS class     ``6``             objective weights and constraints
    """
    link_block = np.zeros(n_links * LINK_FEATS, dtype=np.float32)
    for i, link in enumerate(state.links[:n_links]):
        base = i * LINK_FEATS
        link_block[base + 0] = np.clip(link.utilization, 0.0, 1.0)
        link_block[base + 1] = np.clip(link.queue_size / MAX_QUEUE, 0.0, 1.0)
        link_block[base + 2] = np.clip(link.packet_loss_rate / MAX_LOSS, 0.0, 1.0)
        link_block[base + 3] = np.clip(link.base_latency / MAX_BASE_LATENCY, 0.0, 1.0)

    task_block = np.zeros(n_nodes * 2, dtype=np.float32)
    if 0 <= src_idx < n_nodes:
        task_block[src_idx] = 1.0
    if 0 <= dst_idx < n_nodes:
        task_block[n_nodes + dst_idx] = 1.0

    path_block = np.zeros(K_PATHS * PATH_FEATS, dtype=np.float32)
    for i, path in enumerate(paths[:K_PATHS]):
        path_block[i * PATH_FEATS : (i + 1) * PATH_FEATS] = path_features(state, path)

    qos_block = np.asarray(
        profile_vector(get_profile(None) if profile is None else profile),
        dtype=np.float32,
    )

    return np.concatenate([link_block, task_block, path_block, qos_block]).astype(
        np.float32
    )


__all__ = [
    "K_PATHS",
    "LINK_FEATS",
    "PATH_FEATS",
    "QOS_FEATS",
    "build_graph_tensors",
    "build_observation",
    "observation_dim",
    "path_features",
]
