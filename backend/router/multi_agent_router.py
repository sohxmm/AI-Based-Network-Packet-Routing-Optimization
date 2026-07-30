"""MultiAgentRouter — route packets using regional PPO policies.

Each region of the 25-node topology has its own trained PPO model.
When a routing request arrives, the router determines which region
owns the source node and uses that region's policy to select from
candidate paths.

Falls back to a congestion-aware heuristic if no model is loaded for
the source region (partial training is an acceptable documented outcome).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from simulator.data_models import NetworkState, RoutingDecision
from ml.network_partition import partition_network, build_region_lookup

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "ml" / "models"

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
_ppo_class: Any = None
_numpy: Any = None


def _try_import_ppo():
    global _ppo_class, _numpy
    if _ppo_class is not None:
        return True
    try:
        from stable_baselines3 import PPO
        import numpy as np
        _ppo_class = PPO
        _numpy = np
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Observation builder (must match ml/multi_agent_rl_environment.py)
# ---------------------------------------------------------------------------
_MAX_QUEUE = 100.0
_MAX_LOSS = 0.06
_MAX_BASE_LATENCY = 25.0


def _state_to_obs(state: NetworkState, n_links: int):
    import numpy as np
    obs = np.zeros(n_links * 4, dtype=np.float32)
    for i, link in enumerate(state.links[:n_links]):
        base = i * 4
        obs[base + 0] = float(max(0.0, min(1.0, link.utilization)))
        obs[base + 1] = float(max(0.0, min(1.0, link.queue_size / _MAX_QUEUE)))
        obs[base + 2] = float(max(0.0, min(1.0, link.packet_loss_rate / max(_MAX_LOSS, 1e-9))))
        obs[base + 3] = float(max(0.0, min(1.0, link.base_latency / _MAX_BASE_LATENCY)))
    return obs


# ---------------------------------------------------------------------------
# MultiAgentRouter
# ---------------------------------------------------------------------------
class MultiAgentRouter:
    """Route packets using per-region PPO policies with heuristic fallback.

    Usage
    -----
    >>> router = MultiAgentRouter()
    >>> router.try_load_models()
    >>> decision = router.find_route(state, "R1", "R15")
    """

    def __init__(self, seed: int = 42, k_paths: int = 5) -> None:
        self._seed = seed
        self._k_paths = k_paths
        self._random = random.Random(seed)

        # Will be populated by load_models() / try_load_models()
        self._models: Dict[int, Any] = {}           # region_id -> PPO model
        self._n_links: Dict[int, int] = {}           # region_id -> n_links
        self._partition: Dict[int, List[str]] = {}   # region_id -> [node, …]
        self._region_lookup: Dict[str, int] = {}     # node -> region_id
        self._partition_loaded: bool = False
        self._regions_loaded: List[int] = []
        self._regions_fallback: List[int] = []

    # ------------------------------------------------------------------ #
    # Model management                                                    #
    # ------------------------------------------------------------------ #

    def load_models(self, model_dir: str | Path | None = None) -> None:
        """Load partition + per-region PPO models from *model_dir*."""
        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

        if not _try_import_ppo():
            raise ImportError("stable-baselines3 is required.")

        # Build partition from a fresh simulator (same seed = same topology)
        from simulator.network_sim import NetworkSimulator
        sim = NetworkSimulator(num_nodes=25, seed=self._seed)
        self._partition = partition_network(sim.graph)
        self._region_lookup = build_region_lookup(self._partition)
        self._partition_loaded = True

        self._models.clear()
        self._n_links.clear()
        self._regions_loaded.clear()
        self._regions_fallback.clear()

        for rid in self._partition:
            model_path = model_dir / f"multi_agent_region_{rid}.zip"
            if model_path.exists():
                model = _ppo_class.load(str(model_path))
                obs_dim = model.observation_space.shape[0]
                self._models[rid] = model
                self._n_links[rid] = obs_dim // 4
                self._regions_loaded.append(rid)
                print(f"[MultiAgentRouter] Loaded region {rid} model from {model_path}")
            else:
                self._regions_fallback.append(rid)
                print(f"[MultiAgentRouter] No model for region {rid} — will use heuristic fallback")

    def try_load_models(self, model_dir: str | Path | None = None) -> bool:
        """Attempt to load models, returning False instead of raising."""
        try:
            self.load_models(model_dir)
            return len(self._regions_loaded) > 0
        except Exception as exc:
            print(f"[MultiAgentRouter] Failed to load models: {exc}")
            return False

    @property
    def is_trained(self) -> bool:
        """Return True if at least one region has a trained model."""
        return len(self._regions_loaded) > 0

    @property
    def num_regions(self) -> int:
        return len(self._partition)

    @property
    def regions_loaded(self) -> List[int]:
        return list(self._regions_loaded)

    @property
    def regions_fallback(self) -> List[int]:
        return list(self._regions_fallback)

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def find_route(
        self, state: NetworkState, src: str, dst: str
    ) -> RoutingDecision:
        """Route using the regional policy that owns *src*."""
        if src not in state.nodes or dst not in state.nodes:
            return self._failed_decision(src, dst)

        candidate_paths = _find_candidate_paths(state, src, dst, limit=self._k_paths)
        if not candidate_paths:
            return self._failed_decision(src, dst)

        # Determine region
        region_id = self._region_lookup.get(src, -1)

        is_fallback = False
        if region_id >= 0 and region_id in self._models:
            # Use trained regional policy
            path = self._ppo_select_path(
                state, candidate_paths, self._models[region_id],
                self._n_links[region_id],
            )
            print(f"[MultiAgentRouter] DEBUG: src={src} assigned to Region {region_id}. Model loaded=True. Decision=Trained Regional Policy.")
        else:
            # Heuristic fallback
            path = min(candidate_paths, key=lambda p: _path_cost(state, p))
            reason = "no partition" if region_id < 0 else f"region {region_id} not trained"
            is_fallback = True
            print(f"[MultiAgentRouter] DEBUG: src={src} assigned to Region {region_id}. Model loaded=False ({reason}). Decision=Heuristic Fallback.")

        return RoutingDecision(
            source=src,
            destination=dst,
            path=path,
            algorithm="multi_agent",
            total_latency=_path_cost(state, path),
            avg_utilization=_average_path_utilization(state, path),
            success=True,
            is_fallback=is_fallback,
        )

    def _ppo_select_path(
        self,
        state: NetworkState,
        candidate_paths: List[List[str]],
        model: Any,
        n_links: int,
    ) -> List[str]:
        obs = _state_to_obs(state, n_links)
        action, _ = model.predict(obs, deterministic=True)
        action_idx = int(action) % len(candidate_paths)
        print(f"[MultiAgentRouter DEBUG] predicted action: {action}, action_idx: {action_idx}")
        return candidate_paths[action_idx]

    def _failed_decision(self, src: str, dst: str) -> RoutingDecision:
        return RoutingDecision(
            source=src,
            destination=dst,
            path=[],
            algorithm="multi_agent",
            total_latency=float("inf"),
            avg_utilization=0.0,
            success=False,
        )


# ---------------------------------------------------------------------------
# Graph helpers (same as rl_agent.py)
# ---------------------------------------------------------------------------

def _find_candidate_paths(
    state: NetworkState, src: str, dst: str, limit: int = 10
) -> list[list[str]]:
    """Find simple valid paths with breadth-first search."""
    import networkx as nx
    G = nx.Graph()
    for link in state.links:
        G.add_edge(link.source, link.target)
    
    if src not in G or dst not in G or not nx.has_path(G, src, dst):
        return []

    paths = []
    try:
        path_generator = nx.shortest_simple_paths(G, src, dst)
        for _, path in zip(range(limit), path_generator):
            paths.append(path)
    except nx.NetworkXNoPath:
        pass
    return paths


def _build_adjacency(state: NetworkState) -> Dict[str, List[str]]:
    adjacency: Dict[str, List[str]] = {node: [] for node in state.nodes}
    for link in state.links:
        adjacency[link.source].append(link.target)
        adjacency[link.target].append(link.source)
    return adjacency


def _path_cost(state: NetworkState, path: List[str]) -> float:
    lookup = {
        frozenset((link.source, link.target)): link.base_latency * (1 + 4 * link.utilization ** 2)
        for link in state.links
    }
    return sum(
        lookup[frozenset((path[i], path[i + 1]))]
        for i in range(len(path) - 1)
        if frozenset((path[i], path[i + 1])) in lookup
    )


def _average_path_utilization(state: NetworkState, path: List[str]) -> float:
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
