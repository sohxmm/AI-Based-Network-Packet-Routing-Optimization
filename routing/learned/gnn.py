"""GNNRouter — inference wrapper around the trained path-ranking GNN.

Inference only. The architecture lives in ``ml/architectures/gnn.py`` and the
training loop in ``ml/training/train_gnn.py``, so a router module can never
drift from the model it serves.

The model ranks the same congestion-weighted k-shortest candidate set every
other router sees, and the ranking is conditioned on the QoS profile. There is
deliberately **no post-hoc constraint filtering**: if the GNN returns a path
that violates the traffic class's constraints, that is reported as a miss.
Filtering the model's output through the QoS oracle would make its constraint
satisfaction rate identical to the classical constrained baseline by
construction, which would be a meaningless win.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.models import NetworkState, RoutingDecision
from core.paths import build_decision, candidate_paths, failed_decision
from core.qos import QoSProfile, evaluate_path
from ml.model_registry import path_for
from routing.base import Router

logger = logging.getLogger(__name__)

_torch: Any = None
_model_class: Any = None
_build_graph_tensors: Any = None


def _try_import_torch() -> bool:
    """Import torch lazily so the API starts without the ML stack installed."""
    global _torch, _model_class, _build_graph_tensors
    if _torch is not None:
        return True
    try:
        import torch

        from ml.architectures.gnn import GNNRouterModel
        from ml.features import build_graph_tensors

        _torch = torch
        _model_class = GNNRouterModel
        _build_graph_tensors = build_graph_tensors
        return True
    except ImportError:
        return False


class GNNRouter(Router):
    """Route packets with a trained GNN ranker, falling back to a heuristic."""

    name = "gnn"
    label = "Graph Neural Network"
    description = (
        "Two rounds of edge-conditioned message passing produce node and edge "
        "embeddings; a QoS-conditioned MLP ranks the candidate paths."
    )

    def __init__(self, seed: int = 42, k_paths: int = 5) -> None:
        self._seed = seed
        self._k_paths = k_paths
        self._model: Any = None
        self._model_path: Path | None = None
        self._device: Any = None

    # -- model management -------------------------------------------------

    def load_model(self, path: str | Path | None = None) -> None:
        """Load the GNN checkpoint, raising on failure."""
        candidate = Path(path) if path else path_for("gnn")

        if not candidate.exists():
            raise FileNotFoundError(
                f"GNN model not found at {candidate}. "
                "Train it with: python -m ml.training.train_gnn"
            )
        if not _try_import_torch():
            raise ImportError("torch is required to load the GNN model.")

        # weights_only=True: the checkpoint holds only tensors and plain
        # scalars, and torch.load otherwise executes arbitrary pickle payloads.
        checkpoint = _torch.load(str(candidate), map_location="cpu", weights_only=True)

        # Build into a local first. Assigning self._model before load_state_dict
        # succeeds would leave a randomly-initialised model installed after a
        # failed load, and is_trained would report True while the router served
        # noise — a worse failure than not loading at all.
        model = _model_class(
            node_dim=checkpoint.get("node_dim", 3),
            edge_dim=checkpoint.get("edge_dim", 4),
            hidden_dim=checkpoint.get("hidden_dim", 64),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._model_path = candidate
        logger.info("GNNRouter loaded %s (device=%s)", candidate.name, self._device)

    def try_load_model(self, path: str | Path | None = None) -> bool:
        """Load the model, reporting failure loudly instead of swallowing it.

        The original was ``except (FileNotFoundError, ImportError, Exception):
        return False`` with no logging. That is precisely how a missing artifact
        stayed invisible for the life of the project.
        """
        try:
            self.load_model(path)
            return True
        except FileNotFoundError as exc:
            logger.warning(
                "GNNRouter: no trained model (%s). Falling back to the "
                "congestion-aware heuristic. Train with: python -m ml.training.train_gnn",
                exc,
            )
            return False
        except ImportError as exc:
            logger.warning("GNNRouter: ML dependencies unavailable (%s).", exc)
            return False
        except Exception:
            logger.exception("GNNRouter: unexpected error loading the model.")
            return False

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def requires_model(self) -> bool:
        return True

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "requires_model": True,
            "model_path": str(self._model_path) if self._model_path else None,
        }

    # -- inference --------------------------------------------------------

    def find_route(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        profile: QoSProfile | None = None,
    ) -> RoutingDecision:
        profile = self.resolve_profile(profile)

        if src not in state.nodes or dst not in state.nodes:
            return failed_decision(src, dst, self.name)

        paths = candidate_paths(state, src, dst, k=self._k_paths)
        if not paths:
            return failed_decision(src, dst, self.name)

        if self._model is not None:
            path = self._rank(state, src, dst, paths, profile)
            is_fallback = False
        else:
            path = min(paths, key=lambda p: _heuristic_cost(state, p, profile))
            is_fallback = True

        return build_decision(
            state,
            src,
            dst,
            path,
            self.name,
            is_fallback=is_fallback,
            diagnostics={
                "qos": evaluate_path(state, path, profile).as_dict(),
                "candidates_considered": len(paths),
            },
        )

    def _rank(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        paths: list[list[str]],
        profile: QoSProfile,
    ) -> list[str]:
        """Return the candidate the model scores best (lowest)."""
        x, edge_index, edge_attr, paths_idx, path_edges, path_feats = _build_graph_tensors(
            state, paths, src, dst, profile
        )
        x = x.to(self._device)
        edge_index = edge_index.to(self._device)
        edge_attr = edge_attr.to(self._device)
        path_feats = path_feats.to(self._device)

        with _torch.no_grad():
            scores = self._model(x, edge_index, edge_attr, paths_idx, path_edges, path_feats)

        return paths[int(_torch.argmin(scores).item())]


#: Heuristic-fallback tuning constants, named rather than inline magic numbers.
BOTTLENECK_THRESHOLD = 0.7
BOTTLENECK_WEIGHT = 100.0
IMBALANCE_WEIGHT = 50.0


def _heuristic_cost(state: NetworkState, path: list[str], profile: QoSProfile) -> float:
    """Load-balancing-aware fallback cost used when no model is loaded.

    Deliberately *not* identical to plain Dijkstra: it penalises the path's
    bottleneck link and its load imbalance against the network mean, so the
    fallback still expresses the project's load-balancing intent. It is still a
    heuristic, and every decision it produces is flagged ``is_fallback=True``.
    """
    evaluation = evaluate_path(state, path, profile)
    if evaluation.score == float("inf"):
        return float("inf")

    bottleneck_penalty = (
        max(0.0, evaluation.bottleneck_utilization - BOTTLENECK_THRESHOLD)
        * BOTTLENECK_WEIGHT
    )

    all_utils = [link.utilization for link in state.links]
    network_mean = sum(all_utils) / len(all_utils) if all_utils else 0.0
    imbalance_penalty = (
        max(0.0, evaluation.bottleneck_utilization - network_mean) * IMBALANCE_WEIGHT
    )

    infeasibility = 0.0 if evaluation.feasible else 1000.0
    return evaluation.score + bottleneck_penalty + imbalance_penalty + infeasibility


__all__ = ["GNNRouter"]
