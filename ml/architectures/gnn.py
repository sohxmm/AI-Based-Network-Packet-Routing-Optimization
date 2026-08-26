"""Graph neural network that ranks candidate paths.

Message passing is implemented from scratch rather than via PyTorch Geometric.
That is a deliberate call: PyG's install matrix is a
real deployment liability, and a readable ~40-line implementation demonstrates
what message passing actually *is*.

Two architectural defects in the previous version are fixed here.

**Mean aggregation instead of unnormalised sum.** ``index_add_`` alone makes a
node's embedding magnitude scale with its degree. Training on a degree-2 ring
and serving on a degree-4 mesh therefore doubled activation magnitudes at
inference — a self-inflicted covariate shift. Aggregation is now degree-normalised.

**Edge-aware path pooling instead of a mean over node embeddings.** The old
scorer computed ``h[path_nodes].mean(0)``, which is:

* permutation invariant, so ``A->B->C`` scored identically to ``A->C->B``;
* length invariant, so a 2-hop and a 7-hop path looked alike;
* edge blind, so the utilization of the links actually *on* the path was never
  fed in directly.

It was being asked to predict a path's cost while being shown neither the path's
length nor its links. The scorer now pools node embeddings *and* edge embeddings
along the path and concatenates explicit path-level features.
"""

from __future__ import annotations

import torch
from torch import nn

from core.qos import PROFILE_VECTOR_DIM

#: Explicit features appended to the pooled embeddings. Three describe the
#: path itself (hop count normalised, mean utilization, max utilization) and
#: six encode the QoS profile being served, so one trained model conditions on
#: the traffic class rather than needing five separate models.
PATH_STRUCTURE_DIM = 3
PATH_FEATURE_DIM = PATH_STRUCTURE_DIM + PROFILE_VECTOR_DIM


class MessagePassingLayer(nn.Module):
    """One round of edge-conditioned message passing with mean aggregation."""

    def __init__(self, in_node_dim: int, in_edge_dim: int, out_dim: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_node_dim * 2 + in_edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_node_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return updated node embeddings and the per-edge messages.

        Args:
            x: ``[num_nodes, in_node_dim]``
            edge_index: ``[2, num_edges]`` (row 0 source, row 1 destination)
            edge_attr: ``[num_edges, in_edge_dim]``

        Returns:
            ``(node_embeddings [num_nodes, out_dim], edge_messages [num_edges, out_dim])``
            The edge messages are what the path scorer pools over.
        """
        src, dst = edge_index[0], edge_index[1]

        # One message per directed edge, conditioned on both endpoints and the
        # edge's own features (utilization, queue, loss, latency).
        edge_inputs = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(edge_inputs)

        # Mean-aggregate at the destination node. Degree normalisation keeps
        # embedding magnitudes comparable across topologies of different density.
        aggregated = torch.zeros(x.size(0), messages.size(-1), device=x.device)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros(x.size(0), 1, device=x.device)
        degree.index_add_(0, dst, torch.ones(dst.size(0), 1, device=x.device))
        aggregated = aggregated / degree.clamp(min=1.0)

        updated = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return updated, messages


class GNNRouterModel(nn.Module):
    """Score candidate paths from network state, topology and path structure.

    The output is used only through ``argmin``, so it is trained as a *ranker*
    (pairwise margin loss) rather than a regressor. Its absolute value carries
    no meaning and should never be reported as a latency estimate.
    """

    def __init__(self, node_dim: int = 3, edge_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.conv1 = MessagePassingLayer(node_dim, edge_dim, hidden_dim)
        self.conv2 = MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)

        # Node pooling + edge pooling + explicit path features.
        self.path_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + PATH_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        paths: list[list[int]],
        path_edges: list[list[int]],
        path_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Return one score per candidate path (lower is better).

        Args:
            x: node features ``[num_nodes, node_dim]``
            edge_index: ``[2, num_edges]``
            edge_attr: ``[num_edges, edge_dim]``
            paths: node indices per candidate path
            path_edges: row indices into ``edge_attr`` for the links of each path
            path_feats: ``[num_paths, PATH_FEATURE_DIM]``
        """
        h, _ = self.conv1(x, edge_index, edge_attr)
        h = torch.relu(h)
        h, edge_embeddings = self.conv2(h, edge_index, edge_attr)
        h = torch.relu(h)
        edge_embeddings = torch.relu(edge_embeddings)

        scores = []
        for index, (node_ids, edge_ids) in enumerate(zip(paths, path_edges, strict=True)):
            if not node_ids or not edge_ids:
                # An unusable candidate is scored as maximally bad rather than
                # dropped, so the returned tensor always aligns with `paths`.
                scores.append(torch.full((1,), 1e5, device=x.device))
                continue

            node_pool = h[torch.tensor(node_ids, dtype=torch.long, device=x.device)].mean(0)
            edge_pool = edge_embeddings[
                torch.tensor(edge_ids, dtype=torch.long, device=x.device)
            ].mean(0)
            features = torch.cat([node_pool, edge_pool, path_feats[index]], dim=-1)
            scores.append(self.path_mlp(features).view(1))

        return torch.cat(scores, dim=0)

    def parameter_count(self) -> int:
        """Total trainable parameters, reported in the model card."""
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "PATH_FEATURE_DIM",
    "PATH_STRUCTURE_DIM",
    "GNNRouterModel",
    "MessagePassingLayer",
]
