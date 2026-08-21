"""GNN architecture tests.

The previous suite tested that the layers produced tensors of the right shape.
That is worth keeping, but it could not have caught either of the two defects
we found, because both were about *what the model can represent*:

* aggregation was an unnormalised sum, so embedding magnitudes scaled with node
  degree and a model trained on a degree-2 ring saw roughly double the
  activations on a degree-4 mesh;
* the path scorer was a mean over node embeddings, which is permutation
  invariant, length invariant and edge blind — it was asked to predict a path's
  cost while being shown neither the path's length nor its links.

The tests below assert the *properties*, not the shapes.
"""

from __future__ import annotations

import torch

from core.paths import candidate_paths
from core.qos import TrafficClass, get_profile
from core.simulator import NetworkSimulator
from ml.architectures.gnn import PATH_FEATURE_DIM, GNNRouterModel, MessagePassingLayer
from ml.features import build_graph_tensors
from routing.learned.gnn import GNNRouter


def _sample(seed: int = 42, steps: int = 10):
    sim = NetworkSimulator(num_nodes=25, seed=seed)
    for _ in range(steps):
        sim.step()
    state = sim.get_state()
    paths = candidate_paths(state, "R1", "R14", k=5)
    profile = get_profile(TrafficClass.BEST_EFFORT)
    return state, paths, build_graph_tensors(state, paths, "R1", "R14", profile)


class TestMessagePassing:
    def test_returns_node_and_edge_embeddings(self):
        """The path scorer pools over edges, so the layer has to expose them."""
        layer = MessagePassingLayer(3, 4, 16)
        x = torch.randn(10, 3)
        edge_index = torch.randint(0, 10, (2, 24))
        edge_attr = torch.randn(24, 4)

        nodes, edges = layer(x, edge_index, edge_attr)
        assert nodes.shape == (10, 16)
        assert edges.shape == (24, 16)

    def test_aggregation_is_degree_normalised(self):
        """A high-degree node must not get systematically larger activations.

        With an unnormalised sum, a node with twice the neighbours gets twice
        the signal, which is exactly the covariate shift that broke the model
        when it was trained on a ring and served on a mesh.
        """
        layer = MessagePassingLayer(2, 1, 8)
        torch.manual_seed(0)

        # Node 0 has one neighbour; node 1 has four. Identical edge features.
        x = torch.ones(6, 2)
        sparse = torch.tensor([[1], [0]])
        dense = torch.tensor([[2, 3, 4, 5], [1, 1, 1, 1]])
        edge_index = torch.cat([sparse, dense], dim=1)
        edge_attr = torch.ones(edge_index.shape[1], 1)

        nodes, _ = layer(x, edge_index, edge_attr)
        low_degree = nodes[0].abs().mean().item()
        high_degree = nodes[1].abs().mean().item()

        ratio = high_degree / max(low_degree, 1e-9)
        assert 0.5 < ratio < 2.0, (
            f"Degree-4 activations are {ratio:.2f}x the degree-1 ones. Mean "
            f"aggregation should keep this near 1; a sum would put it near 4."
        )


class TestPathScoring:
    def test_scorer_distinguishes_path_length(self):
        """A mean over node embeddings could not see how long a path was."""
        state, paths, tensors = _sample()
        x, edge_index, edge_attr, paths_idx, path_edges, path_feats = tensors

        lengths = {len(p) for p in paths}
        assert len(lengths) > 1, "need candidates of different lengths"

        torch.manual_seed(0)
        model = GNNRouterModel(hidden_dim=32)
        scores = model(x, edge_index, edge_attr, paths_idx, path_edges, path_feats)

        assert scores.shape == (len(paths),)
        assert len(set(scores.tolist())) == len(paths), (
            "Every candidate scored identically, so the representation cannot "
            "distinguish them."
        )

    def test_scorer_responds_to_link_state_not_just_topology(self):
        """Congesting a path's links must change its score."""
        state, paths, tensors = _sample()
        x, edge_index, edge_attr, paths_idx, path_edges, path_feats = tensors

        torch.manual_seed(0)
        model = GNNRouterModel(hidden_dim=32)
        model.eval()
        with torch.no_grad():
            before = model(x, edge_index, edge_attr, paths_idx, path_edges, path_feats)
            congested = edge_attr.clone()
            congested[:, 0] = 0.95  # utilization
            after = model(x, edge_index, congested, paths_idx, path_edges, path_feats)

        assert not torch.allclose(before, after), (
            "Scores ignored link utilization, so the model is topology-only."
        )

    def test_path_features_carry_the_qos_class(self):
        """One model serves five traffic classes, so the class is an input."""
        sim = NetworkSimulator(num_nodes=25, seed=42)
        for _ in range(10):
            sim.step()
        state = sim.get_state()
        paths = candidate_paths(state, "R1", "R14", k=5)

        _, _, _, _, _, emergency = build_graph_tensors(
            state, paths, "R1", "R14", get_profile(TrafficClass.EMERGENCY)
        )
        _, _, _, _, _, bulk = build_graph_tensors(
            state, paths, "R1", "R14", get_profile(TrafficClass.BULK)
        )

        assert emergency.shape[1] == PATH_FEATURE_DIM
        assert not torch.allclose(emergency, bulk)

    def test_empty_candidate_is_scored_not_dropped(self):
        """The returned tensor must stay aligned with the candidate list."""
        state, paths, tensors = _sample()
        x, edge_index, edge_attr, paths_idx, path_edges, path_feats = tensors

        model = GNNRouterModel(hidden_dim=32)
        broken_edges = [*path_edges[:-1], []]
        scores = model(x, edge_index, edge_attr, paths_idx, broken_edges, path_feats)

        assert scores.shape == (len(paths),)
        assert scores[-1].item() > 1e4, "an unusable candidate must rank last"


class TestGNNRouter:
    def test_falls_back_honestly_without_a_model(self):
        state, _, _ = _sample()
        router = GNNRouter()  # deliberately not loading a checkpoint

        decision = router.find_route(state, "R1", "R14")
        assert decision.success
        assert decision.is_fallback, (
            "A decision made without a model must be flagged, or heuristic "
            "output gets reported as AI output."
        )

    def test_reports_failure_for_an_unknown_node(self):
        state, _, _ = _sample()
        decision = GNNRouter().find_route(state, "R1", "R999")
        assert not decision.success
        assert decision.is_fallback

    def test_uses_the_model_when_one_is_present(self):
        state, _, _ = _sample()
        router = GNNRouter()
        if not router.try_load_model():
            import pytest

            pytest.skip("no trained GNN checkpoint present")

        decision = router.find_route(state, "R1", "R14")
        assert decision.success
        assert not decision.is_fallback
        assert decision.diagnostics.get("candidates_considered", 0) > 0
