from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from ml.gnn_model import GNNRouterModel
from simulator.data_models import NetworkState, RoutingDecision


_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "ml" / "models" / "gnn_router.pt"
)


class GNNRouter:
    """Route packets using a trained Graph Neural Network (GNN) model."""

def _load_balanced_cost(self, state: NetworkState, path: list[str]) -> float:
    """
    Cost function that balances load across the network.
    Unlike Dijkstra which only sees per-path cost, this penalizes
    paths that go through already-congested areas of the network.
    """
    lookup = {
        frozenset((link.source, link.target)): link
        for link in state.links
    }
    path_links = [
        lookup[frozenset((path[i], path[i + 1]))]
        for i in range(len(path) - 1)
        if frozenset((path[i], path[i + 1])) in lookup
    ]
    if not path_links:
        return float("inf")

    # Standard latency cost
    latency = sum(
        lnk.base_latency * (1 + 4 * lnk.utilization ** 2)
        for lnk in path_links
    )

    # Extra penalty for bottleneck links (>70% util)
    max_util = max(lnk.utilization for lnk in path_links)
    bottleneck_penalty = max(0.0, max_util - 0.7) * 100.0

    # Load balancing: penalize if path util >> network average
    all_utils = [lnk.utilization for lnk in state.links]
    network_mean = sum(all_utils) / len(all_utils) if all_utils else 0.0
    path_mean = sum(lnk.utilization for lnk in path_links) / len(path_links)
    imbalance_penalty = max(0.0, path_mean - network_mean) * 50.0

    return latency + bottleneck_penalty + imbalance_penalty

    def __init__(self, k_paths: int = 5, device: str | None = None) -> None:
        self._k_paths = k_paths
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: GNNRouterModel | None = None
        self._model_path: Path | None = None

    def load_model(self, path: str | Path | None = None) -> None:
        """Load the GNN model weights."""
        candidate = Path(path) if path else _DEFAULT_MODEL_PATH
        if not candidate.exists():
            raise FileNotFoundError(f"GNN model not found at {candidate}.")

        self._model = GNNRouterModel(node_dim=3, edge_dim=4, hidden_dim=32)
        checkpoint = torch.load(candidate, map_location=self.device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.to(self.device)
        self._model.eval()
        self._model_path = candidate
        print(f"[GNNRouter] Loaded GNN model from {candidate}")

    def try_load_model(self, path: str | Path | None = None) -> bool:
        """Attempt to load the model, returning False if it fails."""
        try:
            self.load_model(path)
            return True
        except Exception as exc:
            print(f"[GNNRouter] Warning: Could not load model: {exc}")
            return False

    @property
    def is_trained(self) -> bool:
        """Return True if a GNN model is loaded and ready for inference."""
        return self._model is not None

    def predict(self, state: NetworkState, src: str, dst: str) -> RoutingDecision:
        """Select the best path between source and destination using GNN predictions."""
        if src not in state.nodes or dst not in state.nodes:
            return self._failed_decision(src, dst)

        # Pre-compute up to k candidate paths via BFS/shortest simple paths
        candidate_paths = self._find_candidate_paths(state, src, dst, limit=self._k_paths)
        if not candidate_paths:
            return self._failed_decision(src, dst)

        if self._model is not None:
            path = self._gnn_select_path(state, src, dst, candidate_paths)
        else:
            # Fallback to the congestion-aware Dijkstra heuristic
            path = min(candidate_paths, key=lambda p: self._load_balanced_cost(state, p))

        return RoutingDecision(
            source=src,
            destination=dst,
            path=path,
            algorithm="gnn",
            total_latency=self._path_cost(state, path),
            avg_utilization=self._average_path_utilization(state, path),
            success=True,
        )

    def _gnn_select_path(
        self, state: NetworkState, src: str, dst: str, candidate_paths: list[list[str]]
    ) -> list[str]:
        """Convert network state to tensors, run GNN, and pick the path with lowest predicted cost."""
        node_to_idx = {node: i for i, node in enumerate(state.nodes)}
        num_nodes = len(state.nodes)

        # 1. Build node features: [is_source, is_destination, degree_normalized]
        # Calculate degrees
        degrees = [0] * num_nodes
        for link in state.links:
            degrees[node_to_idx[link.source]] += 1
            degrees[node_to_idx[link.target]] += 1
        max_deg = max(1, max(degrees))

        x_list = []
        for i, node in enumerate(state.nodes):
            is_src = 1.0 if node == src else 0.0
            is_dst = 1.0 if node == dst else 0.0
            deg = float(degrees[i]) / max_deg
            x_list.append([is_src, is_dst, deg])
        x = torch.tensor(x_list, dtype=torch.float32, device=self.device)

        # 2. Build edge index and edge attributes (undirected links -> bidirectional edges)
        edges_src = []
        edges_dst = []
        edge_attr_list = []

        for link in state.links:
            u_idx = node_to_idx[link.source]
            v_idx = node_to_idx[link.target]
            
            # Link attributes
            attr = [
                float(link.utilization),
                float(link.queue_size) / 100.0,
                float(link.packet_loss_rate) / 0.06,
                float(link.base_latency) / 25.0,
            ]
            
            # Direction 1: u -> v
            edges_src.append(u_idx)
            edges_dst.append(v_idx)
            edge_attr_list.append(attr)

            # Direction 2: v -> u
            edges_src.append(v_idx)
            edges_dst.append(u_idx)
            edge_attr_list.append(attr)

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32, device=self.device)

        # 3. Formulate candidate paths as lists of node indices
        paths_idx = []
        for path in candidate_paths:
            paths_idx.append([node_to_idx[n] for n in path])

        # 4. GNN Forward Pass
        with torch.no_grad():
            predicted_costs = self._model(x, edge_index, edge_attr, paths_idx)
            best_idx = torch.argmin(predicted_costs).item()

        return candidate_paths[best_idx]

    def _find_candidate_paths(
        self, state: NetworkState, src: str, dst: str, limit: int = 10
    ) -> list[list[str]]:
        """Find simple valid paths using Breadth-First Search."""
        adjacency = self._build_adjacency(state)
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

    def _build_adjacency(self, state: NetworkState) -> dict[str, list[str]]:
        """Build an undirected adjacency list from network state."""
        adjacency: dict[str, list[str]] = {node: [] for node in state.nodes}
        for link in state.links:
            adjacency[link.source].append(link.target)
            adjacency[link.target].append(link.source)
        return adjacency

    def _path_cost(self, state: NetworkState, path: list[str]) -> float:
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

    def _average_path_utilization(self, state: NetworkState, path: list[str]) -> float:
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

    def _failed_decision(self, src: str, dst: str) -> RoutingDecision:
        """Create a failed routing decision."""
        return RoutingDecision(
            source=src,
            destination=dst,
            path=[],
            algorithm="gnn",
            total_latency=float("inf"),
            avg_utilization=0.0,
            success=False,
        )
