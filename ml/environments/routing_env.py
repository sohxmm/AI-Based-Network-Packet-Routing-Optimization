"""NetworkRoutingEnv – Gymnasium environment for PPO-based routing.

Observation space (flat, normalised to [0, 1]):
  Per link: [utilization, queue_size/100, packet_loss_rate/0.06, base_latency/100]
  → shape (n_links * 4,)

Action space:
  Discrete – index of the chosen candidate path (0 … k-1) for a randomly
  sampled (src, dst) pair that is resampled every episode step.

Reward:
  r = -w_latency * normalised_latency - w_util * mean_utilization - w_loss * mean_loss
      - congestion_penalty

The episode runs for a fixed horizon (default 200 steps); done is set by
a step counter only (no terminal state in the network model).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Allow the environment to be imported both as a module and via direct
# execution when sys.path does not yet contain the backend root.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from simulator.network_sim import NetworkSimulator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_LATENCY_MS = 200.0        # upper bound used for normalisation
_MAX_QUEUE = 100.0             # queue_size is already 0-100 in the sim
_MAX_LOSS = 0.06               # max theoretical loss from the sim formula
_MAX_BASE_LATENCY = 25.0       # sim samples base_latency in [5, 25]
_K_PATHS = 5                   # candidate paths exposed to the agent
_EPISODE_STEPS = 200           # steps before truncation


# ---------------------------------------------------------------------------
# Helper: flatten a NetworkState into a normalised numpy observation
# ---------------------------------------------------------------------------
def _state_to_obs(state, n_links: int) -> np.ndarray:
    """Return a float32 array of shape (n_links * 4,) normalised to [0, 1]."""
    obs = np.zeros(n_links * 4, dtype=np.float32)
    for i, link in enumerate(state.links[:n_links]):
        base = i * 4
        obs[base + 0] = float(np.clip(link.utilization, 0.0, 1.0))
        obs[base + 1] = float(np.clip(link.queue_size / _MAX_QUEUE, 0.0, 1.0))
        obs[base + 2] = float(np.clip(link.packet_loss_rate / max(_MAX_LOSS, 1e-9), 0.0, 1.0))
        obs[base + 3] = float(np.clip(link.base_latency / _MAX_BASE_LATENCY, 0.0, 1.0))
    return obs


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class NetworkRoutingEnv(gym.Env):
    """Gymnasium environment for AI-based network packet routing.

    The agent selects one of *k* pre-computed candidate paths between a
    randomly chosen (src, dst) pair at each step.  The simulator advances
    one tick after each action, so the observation at the next step reflects
    the consequences of congestion evolution.
    """

    metadata = {"render_modes": ["human"]}

    # reward weights --------------------------------------------------------
    W_LATENCY = 0.5
    W_UTIL = 0.3
    W_LOSS = 0.2
    CONGESTION_THRESHOLD = 0.85
    CONGESTION_PENALTY = 2.0

    def __init__(
        self,
        num_nodes: int = 25,
        seed: int = 42,
        k_paths: int = _K_PATHS,
        episode_steps: int = _EPISODE_STEPS,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.k_paths = k_paths
        self.episode_steps = episode_steps
        self.render_mode = render_mode

        # Build the simulator ------------------------------------------------
        self._sim = NetworkSimulator(num_nodes=num_nodes, seed=seed)
        self._rng = random.Random(seed)
        initial_state = self._sim.get_state()
        self.n_links: int = len(initial_state.links)
        self._nodes: list[str] = initial_state.nodes

        # Spaces -------------------------------------------------------------
        obs_dim = self.n_links * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(k_paths)

        # Episode bookkeeping ------------------------------------------------
        self._step_count: int = 0
        self._current_src: str = ""
        self._current_dst: str = ""
        self._current_paths: list[list[str]] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            self._sim = NetworkSimulator(num_nodes=self.num_nodes, seed=seed)
            self._nodes = self._sim.get_state().nodes

        self._step_count = 0
        self._sample_routing_task()
        state = self._sim.get_state()
        obs = _state_to_obs(state, self.n_links)
        return obs, {"src": self._current_src, "dst": self._current_dst}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Clamp action to valid path indices
        action = int(action) % max(1, len(self._current_paths))

        # Compute reward BEFORE stepping the simulator
        state = self._sim.get_state()
        reward = self._compute_reward(action, state)

        # Advance simulator
        new_state = self._sim.step()
        self._step_count += 1

        # Resample routing task each step (keeps exploration diverse)
        self._sample_routing_task()

        obs = _state_to_obs(new_state, self.n_links)
        truncated = self._step_count >= self.episode_steps
        terminated = False

        info = {
            "step": self._step_count,
            "src": self._current_src,
            "dst": self._current_dst,
            "chosen_path": self._current_paths[action] if self._current_paths else [],
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            state = self._sim.get_state()
            print(
                f"[Step {self._step_count}] {self._current_src} → {self._current_dst} "
                f"| links={len(state.links)} "
                f"| mean_util={np.mean([l.utilization for l in state.links]):.3f}"
            )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_routing_task(self) -> None:
        """Pick a random (src, dst) pair and pre-compute candidate paths."""
        nodes = self._nodes
        src = self._rng.choice(nodes)
        dst = self._rng.choice([n for n in nodes if n != src])
        self._current_src = src
        self._current_dst = dst
        self._current_paths = self._sim.get_candidate_paths(src, dst, k=self.k_paths)
        # Ensure at least one dummy path so the agent always has a valid choice
        if not self._current_paths:
            self._current_paths = [[src, dst]]

    def _compute_reward(self, action: int, state) -> float:
        """
        New reward function that GNN/RL can optimize but Dijkstra cannot:
        - Penalizes path latency (same as before)
        - Penalizes congestion on chosen path (same as before)
        - NEW: Penalizes GLOBAL network load variance (load balancing)
        - NEW: Penalizes the single most overloaded link in the network
        These last two terms give RL a genuinely different objective from Dijkstra,
        which only sees per-path cost and has no concept of global load distribution.
        """
        paths = self._current_paths
        if not paths or action >= len(paths):
            return -self.CONGESTION_PENALTY

        path = paths[action]
        link_lookup = {
            frozenset((lnk.source, lnk.target)): lnk
            for lnk in state.links
        }

        path_links = []
        for i in range(len(path) - 1):
            key = frozenset((path[i], path[i + 1]))
            if key in link_lookup:
                path_links.append(link_lookup[key])

        if not path_links:
            return -self.CONGESTION_PENALTY

        # --- Original per-path terms ---
        mean_util = float(np.mean([lnk.utilization for lnk in path_links]))
        mean_loss = float(np.mean([lnk.packet_loss_rate for lnk in path_links]))
        total_latency_raw = sum(
            lnk.base_latency * (1 + 4 * lnk.utilization ** 2)
            for lnk in path_links
        )
        norm_latency = float(np.clip(total_latency_raw / _MAX_LATENCY_MS, 0.0, 1.0))
        norm_loss = float(np.clip(mean_loss / max(_MAX_LOSS, 1e-9), 0.0, 1.0))

        reward = -(
            self.W_LATENCY * norm_latency
            + self.W_UTIL * mean_util
            + self.W_LOSS * norm_loss
        )

        # Extra penalty for congested links on chosen path
        congested = sum(1 for lnk in path_links if lnk.utilization > self.CONGESTION_THRESHOLD)
        reward -= congested * self.CONGESTION_PENALTY * 0.1

        # --- NEW: Global load balancing terms ---
        all_utils = [lnk.utilization for lnk in state.links]

        # Penalize high VARIANCE across all links — Dijkstra ignores this entirely
        # High variance means some links are overloaded while others are idle
        util_variance = float(np.var(all_utils))
        reward -= 0.4 * util_variance

        # Penalize the single worst link in the network heavily
        # Dijkstra may route through the cheapest path but inadvertently
        # push one link to 95%+ utilization — RL learns to avoid this
        max_util = float(max(all_utils))
        if max_util > 0.8:
            reward -= 0.5 * (max_util - 0.8)  # only penalize above 80% threshold

        return float(reward)