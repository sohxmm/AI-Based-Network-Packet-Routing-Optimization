"""RLRouter – wraps a trained Stable-Baselines3 PPO model for inference.

At startup the router tries to load backend/ml/models/rl_router_final.zip.
If the file is absent (model not yet trained), it falls back to the
congestion-aware Dijkstra heuristic so the API never fails cold.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# Ensure backend root is on sys.path when this module is imported from the
# router package without the broader package context being set up.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from simulator.data_models import NetworkState, RoutingDecision

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "ml" / "models" / "rl_router_final"
)

# ---------------------------------------------------------------------------
# Lazy imports – torch / SB3 are only needed when the PPO model is loaded
# ---------------------------------------------------------------------------
_ppo_class: Any = None
_numpy: Any = None


def _try_import_ppo():
    global _ppo_class, _numpy
    if _ppo_class is not None:
        return True
    try:
        from stable_baselines3 import PPO  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        _ppo_class = PPO
        _numpy = np
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Observation builder (must match ml/rl_environment.py exactly)
# ---------------------------------------------------------------------------
_MAX_QUEUE = 100.0
_MAX_LOSS = 0.06
_MAX_BASE_LATENCY = 25.0


def _state_to_obs(state: NetworkState, n_links: int):
    """Build the same flat observation vector used during training."""
    import numpy as np  # noqa: PLC0415

    obs = np.zeros(n_links * 4, dtype=np.float32)
    for i, link in enumerate(state.links[:n_links]):
        base = i * 4
        obs[base + 0] = float(max(0.0, min(1.0, link.utilization)))
        obs[base + 1] = float(max(0.0, min(1.0, link.queue_size / _MAX_QUEUE)))
        obs[base + 2] = float(
            max(0.0, min(1.0, link.packet_loss_rate / max(_MAX_LOSS, 1e-9)))
        )
        obs[base + 3] = float(
            max(0.0, min(1.0, link.base_latency / _MAX_BASE_LATENCY))
        )
    return obs


# ---------------------------------------------------------------------------
# RLRouter
# ---------------------------------------------------------------------------
class RLRouter:
    """Route packets using a trained PPO policy, with a heuristic fallback.

    Typical usage
    -------------
    >>> router = RLRouter()
    >>> router.load_model()           # auto-discovers the default model path
    >>> decision = router.predict(state, "R1", "R5")
    """

    def __init__(self, seed: int = 42, k_paths: int = 5) -> None:
        self._seed = seed
        self._k_paths = k_paths
        self._model: Any = None
        self._n_links: int | None = None
        self._model_path: Path | None = None
        self._random = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Model management                                                    #
    # ------------------------------------------------------------------ #

    def load_model(self, path: str | Path | None = None) -> None:
        """Load the PPO model from *path* (defaults to the standard location).

        Raises
        ------
        FileNotFoundError
            If neither the .zip file nor a bare directory is found.
        """
        candidate = Path(path) if path else _DEFAULT_MODEL_PATH
        # SB3 saves as <name>.zip; accept both forms
        zip_path = candidate.with_suffix(".zip") if candidate.suffix != ".zip" else candidate

        if not zip_path.exists() and not candidate.exists():
            raise FileNotFoundError(
                f"PPO model not found at {zip_path} or {candidate}.\n"
                "Run `python -m ml.train_rl` from the backend/ directory to train."
            )

        if not _try_import_ppo():
            raise ImportError(
                "stable-baselines3 is required to load the RL model. "
                "Install it with: pip install stable-baselines3"
            )

        actual_path = zip_path if zip_path.exists() else candidate
        self._model = _ppo_class.load(str(actual_path))
        self._model_path = actual_path

        # Infer n_links from the policy input dimension (obs_dim = n_links * 4)
        obs_dim: int = self._model.observation_space.shape[0]
        self._n_links = obs_dim // 4
        print(
            f"[RLRouter] Loaded PPO model from {actual_path} "
            f"(n_links={self._n_links})"
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
        """Return True if a PPO model is loaded and ready for inference."""
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, state: NetworkState, src: str, dst: str) -> RoutingDecision:
        """Return the best routing decision for the given (src, dst) pair.

        If the PPO model is loaded, it ranks the *k* candidate paths by
        inferring the policy's action from the current observation and
        maps the discrete action index back to a concrete node-sequence path.

        Falls back to the congestion-aware heuristic when the model is
        absent or if no path exists between src and dst.
        """
        if src not in state.nodes or dst not in state.nodes:
            return self._failed_decision(src, dst)

        candidate_paths = _find_candidate_paths(state, src, dst, limit=self._k_paths)
        if not candidate_paths:
            return self._failed_decision(src, dst)

        if self._model is not None and self._n_links is not None:
            path = self._ppo_select_path(state, candidate_paths)
        else:
            # Heuristic fallback: pick path with lowest congestion-adjusted cost
            path = min(candidate_paths, key=lambda p: _path_cost(state, p))

        return RoutingDecision(
            source=src,
            destination=dst,
            path=path,
            algorithm="rl",
            total_latency=_path_cost(state, path),
            avg_utilization=_average_path_utilization(state, path),
            success=True,
        )

    def _ppo_select_path(
        self, state: NetworkState, candidate_paths: list[list[str]]
    ) -> list[str]:
        """Use the PPO policy to select the best candidate path."""
        obs = _state_to_obs(state, self._n_links)  # type: ignore[arg-type]
        action, _ = self._model.predict(obs, deterministic=True)
        action_idx = int(action) % len(candidate_paths)
        return candidate_paths[action_idx]

    def _failed_decision(self, src: str, dst: str) -> RoutingDecision:
        """Create a failed RL routing decision."""
        return RoutingDecision(
            source=src,
            destination=dst,
            path=[],
            algorithm="rl",
            total_latency=float("inf"),
            avg_utilization=0.0,
            success=False,
        )


# ---------------------------------------------------------------------------
# Graph helpers (shared with fallback heuristic)
# ---------------------------------------------------------------------------

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
