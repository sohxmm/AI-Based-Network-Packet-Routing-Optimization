"""multi_agent_rl_environment.py — Regional Gymnasium environment for multi-agent PPO.

Architecture: Centralized-reward / Decentralized-execution
  - Each region has its own RegionalRoutingEnv (and its own PPO policy).
  - The observation is the FULL network state (centralized critic view).
  - Actions only route traffic that ORIGINATES in the region's nodes.
  - Reward combines per-path terms (from rl_environment._compute_reward)
    PLUS a shared global term (network-wide utilization variance and
    max-link penalty) computed across ALL links, not just the acting
    agent's region.

This is NOT full simultaneous MARL — each env is stepped independently
with standard SB3 PPO.  The multi-agent coordination emerges from the
shared global reward signal and the naive self-play training rotation
in train_multi_agent.py.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from simulator.network_sim import NetworkSimulator
from ml.network_partition import partition_network, build_region_lookup

# ---------------------------------------------------------------------------
# Constants (must match rl_environment.py for observation compatibility)
# ---------------------------------------------------------------------------
_MAX_LATENCY_MS = 200.0
_MAX_QUEUE = 100.0
_MAX_LOSS = 0.06
_MAX_BASE_LATENCY = 25.0
_K_PATHS = 5
_EPISODE_STEPS = 200


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


class RegionalRoutingEnv(gym.Env):
    """Gymnasium environment for one region's PPO routing agent.

    The agent selects one of *k* pre-computed candidate paths between a
    randomly chosen (src, dst) pair where **src belongs to this region**.
    The observation is the FULL network state so the agent has global
    visibility (centralized critic), but it can only act on traffic
    originating in its assigned region (decentralized execution).
    """

    metadata = {"render_modes": ["human"]}

    # Weights for reward components (imbalance fix)
    W_LATENCY = 1.0
    W_UTIL = 1.0
    W_LOSS = 2.0
    CONGESTION_PENALTY = 5.0
    CONGESTION_THRESHOLD = 0.8

    # Increased global weights so they dominate the local path terms
    W_GLOBAL_VARIANCE = 50.0
    W_MAX_LINK = 20.0

    def __init__(
        self,
        region_id: int,
        region_nodes: List[str],
        num_nodes: int = 25,
        seed: int = 42,
        k_paths: int = _K_PATHS,
        episode_steps: int = _EPISODE_STEPS,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.region_id = region_id
        self.region_nodes = list(region_nodes)
        self.num_nodes = num_nodes
        self.k_paths = k_paths
        self.episode_steps = episode_steps
        self.render_mode = render_mode

        # Build simulator (shared topology, same seed for reproducibility)
        self._sim = NetworkSimulator(num_nodes=num_nodes, seed=seed)
        self._rng = random.Random(seed + region_id)
        initial_state = self._sim.get_state()
        self.n_links: int = len(initial_state.links)
        self._all_nodes: List[str] = initial_state.nodes

        # Observation: full network (centralized critic)
        obs_dim = self.n_links * 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(k_paths)

        # Episode bookkeeping
        self._step_count: int = 0
        self._current_src: str = ""
        self._current_dst: str = ""
        self._current_paths: List[List[str]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed + self.region_id)
            self._sim = NetworkSimulator(num_nodes=self.num_nodes, seed=seed)
            self._all_nodes = self._sim.get_state().nodes

        self._step_count = 0
        self._sample_routing_task()
        state = self._sim.get_state()
        obs = _state_to_obs(state, self.n_links)
        return obs, {"src": self._current_src, "dst": self._current_dst, "region": self.region_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action) % max(1, len(self._current_paths))

        state = self._sim.get_state()
        reward = self._compute_reward(action, state)

        new_state = self._sim.step()
        self._step_count += 1
        self._sample_routing_task()

        obs = _state_to_obs(new_state, self.n_links)
        truncated = self._step_count >= self.episode_steps
        terminated = False

        info = {
            "step": self._step_count,
            "src": self._current_src,
            "dst": self._current_dst,
            "region": self.region_id,
            "chosen_path": self._current_paths[action] if self._current_paths else [],
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            state = self._sim.get_state()
            print(
                f"[Region {self.region_id} | Step {self._step_count}] "
                f"{self._current_src} → {self._current_dst} "
                f"| mean_util={np.mean([l.utilization for l in state.links]):.3f}"
            )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_routing_task(self) -> None:
        """Pick a random (src, dst) pair where src is in this region."""
        src = self._rng.choice(self.region_nodes)
        # dst can be ANY node in the network (cross-region traffic is normal)
        dst = self._rng.choice([n for n in self._all_nodes if n != src])
        self._current_src = src
        self._current_dst = dst
        self._current_paths = self._sim.get_candidate_paths(src, dst, k=self.k_paths)
        if not self._current_paths:
            self._current_paths = [[src, dst]]

    def _compute_reward(self, action: int, state) -> float:
        """Reward = per-path terms + GLOBAL load-balancing terms.

        The per-path terms are identical to rl_environment._compute_reward().
        The global terms penalise network-wide utilization variance and the
        single most overloaded link.  These are computed across ALL links,
        not just those in this region, which is the key mechanism for
        inter-region cooperation.
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

        # --- Per-path terms (same as single-agent) ---
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

        # Congestion penalty on chosen path
        congested = sum(1 for lnk in path_links if lnk.utilization > self.CONGESTION_THRESHOLD)
        reward -= congested * self.CONGESTION_PENALTY * 0.1

        # --- GLOBAL load-balancing terms (shared across all regions) ---
        # BUGFIX: Previously, all_utils used `state.links` directly, making it
        # independent of the chosen action. To provide a useful gradient, we
        # estimate the action's impact by artificially inflating the chosen
        # path's utilization before computing global variance.
        
        path_keys = {frozenset((lnk.source, lnk.target)) for lnk in path_links}
        all_utils = []
        for lnk in state.links:
            key = frozenset((lnk.source, lnk.target))
            if key in path_keys:
                # Simulate traffic impact
                all_utils.append(min(1.0, lnk.utilization + 0.05))
            else:
                all_utils.append(lnk.utilization)

        # Penalize high VARIANCE — the core cooperative signal
        util_variance = float(np.var(all_utils))
        reward -= self.W_GLOBAL_VARIANCE * util_variance

        # Penalize the single worst link heavily
        max_util = float(max(all_utils))
        if max_util > 0.8:
            reward -= self.W_MAX_LINK * (max_util - 0.8)

        return float(reward)
