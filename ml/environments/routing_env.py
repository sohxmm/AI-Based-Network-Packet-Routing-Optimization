"""NetworkRoutingEnv — Gymnasium environment for the single-agent PPO router.

Every defect in the previous version is addressed here, and each one alone
was enough to prevent learning.

**The task was not observable.** The observation encoded per-link features only.
It never encoded the source or the destination, while ``_sample_routing_task()``
re-drew ``(src, dst)`` *every step*. The agent was asked to pick "path index 2"
without being told which pair it was routing, and the meaning of index 2 changed
completely between steps. That is not a partially observable MDP, it is an
unobservable one, and it fully explains the flat evaluation curve we
measured (slope not significant, r-squared 0.001 over 500k timesteps). The
observation now contains the task and the features of each candidate.

**Reward and observation described different problems.** Reward was computed
from the pre-step state, then the task was resampled, then the observation was
built — so ``obs_{t+1}`` described a different routing problem than ``r_t``
scored. Ordering is now explicit and commented.

**The load-balancing terms had zero gradient.** ``util_variance`` and
``max_util`` were computed over *all* links, independent of the action taken. In
policy-gradient terms that is a pure state-dependent baseline: it shifts returns
and adds variance while contributing nothing to the policy gradient. Now that
the simulator is closed-loop, the chosen path is registered *before* the tick and
the global term is measured on the resulting state, so it genuinely depends on
what the agent did.

**Train/serve skew.** Training drew candidates ordered by congestion-adjusted
cost while inference used an unweighted hop-count ordering, so action index *k*
meant different paths in each. Both now call
:func:`core.paths.candidate_paths`.

**A non-stationary training distribution.** The simulator was never reset
between episodes, so utilization random-walked to the [0,1] boundaries and the
agent spent most of its budget on states it would never see at inference. Each
episode now starts from a freshly seeded simulator.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.cost import MAX_LATENCY_MS
from core.paths import candidate_paths
from core.qos import ALL_CLASSES, QOS_PROFILES, evaluate_path, get_profile
from core.simulator import NetworkSimulator
from ml.features import K_PATHS, build_observation, observation_dim

EPISODE_STEPS = 200

#: Reward shaping weights. Kept small and named so the model card can quote them.
W_QOS = 1.0
W_INFEASIBLE = 0.5
W_GLOBAL_LOAD = 0.6
W_BOTTLENECK = 0.4


class NetworkRoutingEnv(gym.Env):
    """Choose one of k candidate paths for a sampled demand and traffic class."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_nodes: int = 25,
        seed: int = 42,
        k_paths: int = K_PATHS,
        episode_steps: int = EPISODE_STEPS,
        background_flows: int = 3,
        qos_classes: list | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.k_paths = k_paths
        self.episode_steps = episode_steps
        self.background_flows = background_flows
        self.render_mode = render_mode
        self.qos_classes = list(qos_classes) if qos_classes else list(ALL_CLASSES)

        self._base_seed = seed
        self._episode_index = 0
        self._rng = random.Random(seed)
        self._sim = self._new_simulator(seed)

        initial = self._sim.get_state()
        self.n_links = len(initial.links)
        self.n_nodes = len(initial.nodes)
        self._nodes = list(initial.nodes)
        self._node_to_idx = {node: i for i, node in enumerate(self._nodes)}

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_dim(self.n_links, self.n_nodes),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(k_paths)

        self._step_count = 0
        self._current_src = ""
        self._current_dst = ""
        self._current_paths: list[list[str]] = []
        self._current_profile = get_profile(None)

    def _new_simulator(self, seed: int) -> NetworkSimulator:
        return NetworkSimulator(
            num_nodes=self.num_nodes,
            seed=seed,
            background_flows=self.background_flows,
        )

    # -- Gymnasium API ----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # A fresh simulator per episode. Without this the utilization process
        # saturates and the agent trains on a distribution it never sees served.
        episode_seed = seed if seed is not None else self._base_seed + self._episode_index
        self._episode_index += 1
        self._rng = random.Random(episode_seed)
        self._sim = self._new_simulator(episode_seed)

        # Warm up so episodes do not all start from the identical initial draw.
        for _ in range(self._rng.randint(0, 30)):
            self._sim.step()

        self._step_count = 0
        self._sample_task()
        observation = self._observe(self._sim.get_state())
        return observation, self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action) % max(1, len(self._current_paths))

        # 1. Score the decision against the state it was actually made in.
        state = self._sim.get_state()
        chosen = self._current_paths[action] if self._current_paths else []
        immediate = self._decision_reward(chosen, state)

        # 2. Apply the decision to the network, then advance time. This is what
        #    makes the global load term action-dependent rather than a constant
        #    baseline: our own flow is part of what we are about to be judged on.
        if chosen:
            self._sim.register_flow(chosen)
        new_state = self._sim.step()
        self._step_count += 1

        reward = immediate + self._global_reward(new_state)

        # 3. Draw the *next* task, then build the observation for it. The
        #    observation now carries the task, so obs_{t+1} fully describes the
        #    decision problem the agent is about to face.
        self._sample_task()
        observation = self._observe(new_state)

        truncated = self._step_count >= self.episode_steps
        info = self._info()
        info["chosen_path"] = chosen
        info["reward"] = reward
        return observation, float(reward), False, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            state = self._sim.get_state()
            mean_util = float(np.mean([link.utilization for link in state.links]))
            print(
                f"[{self._step_count}] {self._current_src} -> {self._current_dst} "
                f"({self._current_profile.traffic_class.value}) mean_util={mean_util:.3f}"
            )

    def close(self) -> None:
        return None

    # -- internals --------------------------------------------------------

    def _sample_task(self) -> None:
        """Draw a demand and a traffic class, and pre-compute the candidates."""
        state = self._sim.get_state()
        nodes = state.nodes
        src = self._rng.choice(nodes)
        dst = self._rng.choice([n for n in nodes if n != src])

        self._current_src = src
        self._current_dst = dst
        self._current_profile = QOS_PROFILES[self._rng.choice(self.qos_classes)]
        # Same generator, same ordering, same weighting as inference.
        self._current_paths = candidate_paths(state, src, dst, k=self.k_paths)

    def _observe(self, state) -> np.ndarray:
        return build_observation(
            state,
            self.n_links,
            self.n_nodes,
            self._node_to_idx.get(self._current_src, 0),
            self._node_to_idx.get(self._current_dst, 0),
            self._current_paths,
            self._current_profile,
        )

    def _decision_reward(self, path: list[str], state) -> float:
        """Negative class-weighted cost of the chosen path, plus a feasibility term."""
        if not path:
            return -W_QOS - W_INFEASIBLE

        evaluation = evaluate_path(state, path, self._current_profile)
        if evaluation.score == float("inf"):
            return -W_QOS - W_INFEASIBLE

        # qos scores are sums of normalised per-link terms; divide by a hop
        # budget so the scale is comparable across path lengths.
        normalised = min(1.0, evaluation.score / (MAX_LATENCY_MS / 20.0))
        reward = -W_QOS * normalised
        if not evaluation.feasible:
            reward -= W_INFEASIBLE
        return reward

    @staticmethod
    def _global_reward(state) -> float:
        """Penalise a badly balanced network, measured *after* our flow landed."""
        utils = [link.utilization for link in state.links]
        if not utils:
            return 0.0
        variance = float(np.var(utils))
        worst = float(max(utils))
        penalty = W_GLOBAL_LOAD * variance
        if worst > 0.8:
            penalty += W_BOTTLENECK * (worst - 0.8)
        return -penalty

    def _info(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "src": self._current_src,
            "dst": self._current_dst,
            "traffic_class": self._current_profile.traffic_class.value,
            "n_candidates": len(self._current_paths),
        }

    # -- helpers used by the evaluation baselines -------------------------

    @property
    def current_paths(self) -> list[list[str]]:
        return list(self._current_paths)

    @property
    def current_profile(self):
        return self._current_profile

    def oracle_action(self) -> int:
        """Index of the candidate that maximises the immediate reward.

        This is the greedy ceiling used to normalise the PPO score. It is greedy,
        not optimal: it ignores the downstream consequences of its own load, so a
        policy that learns to plan ahead can in principle exceed it.
        """
        if not self._current_paths:
            return 0
        state = self._sim.get_state()
        rewards = [self._decision_reward(p, state) for p in self._current_paths]
        return int(np.argmax(rewards))


__all__ = ["EPISODE_STEPS", "NetworkRoutingEnv"]
