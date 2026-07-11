"""GNNRouter – wraps a trained Graph Neural Network model for inference.

At startup the router tries to load backend/ml/models/gnn_router.pt.
If the file is absent (model not yet trained), it falls back to the
congestion-aware Dijkstra heuristic so the API never fails cold.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from simulator.data_models import NetworkState, RoutingDecision

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "ml" / "models" / "gnn_router.pt"
)

# ---------------------------------------------------------------------------
# Lazy imports – torch is only needed when the GNN model is loaded
# ---------------------------------------------------------------------------
_torch: Any = None
_gnn_model_class: Any = None


def _try_import_torch():
    global _torch, _gnn_model_class
    if _torch is not None:
        return True
    try:
        import torch  # noqa: PLC0415
        from ml.gnn_model import GNNRouterModel  # noqa: PLC0415

        _torch = torch
        _gnn_model_class = GNNRouterModel
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# GNNRouter
# ---------------------------------------------------------------------------
class GNNRouter:
    """Route packets using a trained GNN policy, with a heuristic fallback.

    The GNN understands network topology by performing message passing
    over nodes and edges. It scores candidate paths based on both
    network state and graph structure.

    Typical usage
    -------------
    >>> router = GNNRouter()
    >>> router.load_model()           # auto-discovers the default model path
    >>> decision = router.predict(state, "R1", "R5")
    """

    def __init__(self, seed: int = 42, k_paths: int = 5) -> None:
        self._seed = seed
        self._k_paths = k_paths
        self._model: Any = None
        self._model_path: Path | None = None
        self._device: Any = None
        self._random = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Model management                                                    #
    # ------------------------------------------------------------------ #

    def load_model(self, path: str | Path | None = None) -> None:
        """Load the GNN model from *path* (defaults to the standard location).

        Raises
        ------
        FileNotFoundError
            If the model file is not found.
        """
        candidate = Path(path) if path else _DEFAULT_MODEL_PATH

        if not candidate.exists():
            raise FileNotFoundError(
                f"GNN model not found at {candidate}.\n"
                "Run `python -m ml.train_gnn` from the backend/ directory to train."
            )

        if not _try_import_torch():
            raise ImportError(
                "torch is required to load the GNN model. "
                "Install it with: pip install torch"
            )

        # Load model weights
        checkpoint = _torch.load(str(candidate), map_location="cpu")
        self._model = _gnn_model_class(
            node_dim=checkpoint.get("node_dim", 3),
            edge_dim=checkpoint.get("edge_dim", 4),
            hidden_dim=checkpoint.get("hidden_dim", 32),
        )
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()  # Set to evaluation mode

        # Determine device
        self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._model_path = candidate

        print(
            f"[GNNRouter] Loaded GNN model from {candidate} (device={self._device})"
        )

    def try_load_model(self, path: str | Path | None = None) -> bool:
        """Attempt to load the model, returning False instead of raising."""
        try:
            self.load_model(path)
            return True
        except (FileNotFoundError, ImportError, Exception):
            return False

    @property
    def is_trained(self) -> bool:
        """Return True if a GNN model is loaded and ready for inference."""
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, state: NetworkState, src: str, dst: str) -> RoutingDecision:
        """Return the best routing decision for the given (src, dst) pair.

        If the GNN model is loaded, it scores candidate paths by passing
        the network graph through message-passing layers and aggregating
        node embeddings along each path.

        Falls back to the congestion-aware heuristic when the model is
        absent or if no path exists between src and dst.
        """
        if src not in state.nodes or dst not in state.nodes:
            return self._failed_decision(src, dst)

        candidate_paths = _find_candidate_paths(state, src, dst, limit=self._k_paths)
        if not candidate_paths:
            return self._failed_decision(src, dst)

        if self._model is not None:
            path = self._gnn_select_path(state, candidate_paths)
        else:
            # Heuristic fallback: pick path with lowest congestion-adjusted cost
            path = min(candidate_paths, key=lambda p: _path_cost(state, p))

        return RoutingDecision(
            source=src,
            destination=dst,
            path=path,
            algorithm="gnn",
            total_latency=_path_cost(state, path),
            avg_utilization=_average_path_utilization(state, path),
            success=True,
        )

    def _gnn_select_path(
        self, state: NetworkState, candidate_paths: list[list[str]]
    ) -> list[str]:
        """Use the GNN to score and select the best candidate path."""
        x, edge_index, edge_attr, paths_idx = _build_graph_data(state, candidate_paths)

        # Move tensors to device
        x = x.to(self._device)
        edge_index = edge_index.to(self._device)
        edge_attr = edge_attr.to(self._device)

        with _torch.no_grad():
            scores = self._model(x, edge_index, edge_attr, paths_idx)

        # Select path with lowest (best) score
        best_idx = _torch.argmin(scores).item()
        return candidate_paths[best_idx]

    def _failed_decision(self, src: str, dst: str) -> RoutingDecision:
        """Create a failed GNN routing decision."""
        return RoutingDecision(
            source=src,
            destination=dst,
            path=[],
            algorithm="gnn",
            total_latency=float("inf"),
            avg_utilization=0.0,
            success=False,
        )


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _build_graph_data(
    state: NetworkState, candidate_paths: list[list[str]]
) -> tuple[Any, Any, Any, list[list[int]]]:
    """Build node features, edge index, and edge attributes for the GNN."""
    import torch as torch_module  # noqa: PLC0415

    nodes = state.nodes
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # 1. Build node features [num_nodes, 3]
    degrees = [0] * num_nodes
    for link in state.links:
        degrees[node_to_idx[link.source]] += 1
        degrees[node_to_idx[link.target]] += 1
    max_deg = max(1, max(degrees)) if degrees else 1

    x_list = []
    for i, node in enumerate(nodes):
        # Node features: [is_src, is_dst, degree_norm]
        # These will be updated per-inference if we pass src/dst
        is_src = 0.0
        is_dst = 0.0
        deg = float(degrees[i]) / max_deg
        x_list.append([is_src, is_dst, deg])

    x = torch_module.tensor(x_list, dtype=torch_module.float32)

    # 2. Build edge index and edge attributes [num_edges, 4]
    edges_src = []
    edges_dst = []
    edge_attr_list = []

    for link in state.links:
        u_idx = node_to_idx[link.source]
        v_idx = node_to_idx[link.target]
        attr = [
            float(link.utilization),
            float(link.queue_size) / 100.0,
            float(link.packet_loss_rate) / 0.06,
            float(link.base_latency) / 25.0,
        ]
        # u -> v
        edges_src.append(u_idx)
        edges_dst.append(v_idx)
        edge_attr_list.append(attr)
        # v -> u (bidirectional)
        edges_src.append(v_idx)
        edges_dst.append(u_idx)
        edge_attr_list.append(attr)

    edge_index = torch_module.tensor([edges_src, edges_dst], dtype=torch_module.long)
    edge_attr = torch_module.tensor(edge_attr_list, dtype=torch_module.float32)

    # 3. Convert candidate paths to node indices
    paths_idx = [[node_to_idx[n] for n in p] for p in candidate_paths]

    return x, edge_index, edge_attr, paths_idx


def _find_candidate_paths(
    state: NetworkState, src: str, dst: str, limit: int = 10
) -> list[list[str]]:
    """Find simple valid paths with breadth-first search."""
    adjacency = _build_adjacency(state)
    queue: list[list[str]] = [[src]]
    paths: list[list[str]] = []

    while queue and len(paths) < limit:
        path = queue.pop(0)
        current = path[-1]

        if current == dst:
            paths.append(path)
            continue

        for neighbor in adjacency.get(current, []):
            if neighbor not in path:
                queue.append([*path, neighbor])

    return paths


def _build_adjacency(state: NetworkState) -> dict[str, list[str]]:
    """Build an undirected adjacency list from the network state."""
    adjacency: dict[str, list[str]] = {node: [] for node in state.nodes}
    for link in state.links:
        adjacency[link.source].append(link.target)
        adjacency[link.target].append(link.source)
    return adjacency


def _path_cost(state: NetworkState, path: list[str]) -> float:
    """Calculate total congestion-adjusted latency across a path."""
    lookup = {
        frozenset((link.source, link.target)): link.base_latency
        * (1 + 4 * link.utilization**2)
        for link in state.links
    }
    return sum(
        lookup[frozenset((path[i], path[i + 1]))]
        for i in range(len(path) - 1)
        if frozenset((path[i], path[i + 1])) in lookup
    )


def _average_path_utilization(state: NetworkState, path: list[str]) -> float:
    """Calculate mean utilization across the selected path."""
    lookup = {
        frozenset((link.source, link.target)): link.utilization
        for link in state.links
    }
    values = [
        lookup[frozenset((path[i], path[i + 1]))]
        for i in range(len(path) - 1)
        if frozenset((path[i], path[i + 1])) in lookup
    ]
    return sum(values) / len(values) if values else 0.0
