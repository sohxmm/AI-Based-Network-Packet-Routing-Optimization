"""RegionalRoutingEnv — the training environment for one decentralized agent.

This is the environment half of turning the project's "multi-agent RL" from a
gated mixture-of-experts into something that earns the name.

The setting is **decentralized execution with centralized training and
independent learners**, which is a standard and honest MARL formulation:

* Each region owns one agent. An agent acts only when the packet is sitting on a
  node inside its region; when the packet leaves, control passes to whichever
  agent owns the next region and this environment fast-forwards through those
  hops using the current partner policy.
* An action is a **single next hop**, chosen from the current node's ordered
  neighbour list — not a complete end-to-end path.
* The observation is **purely local**: the current node, the destination
  expressed relatively, and the incident links. Its width does not depend on the
  size of the network.
* The **critic** additionally receives a fixed-width global summary, which the
  policy network never sees. That asymmetry is what "centralized training" means
  here, and it is enforced in ``ml/training/train_regional.py``.

What it is not: there is no explicit inter-agent messaging and no shared critic
across agents. Agents are independent learners that cooperate through the shared
environment and a shared team reward term. The model card says exactly this.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from core.cost import MAX_BASE_LATENCY, link_cost
from core.paths import build_graph, link_lookup
from core.qos import ALL_CLASSES, QOS_PROFILES, get_profile
from core.simulator import NetworkSimulator
from ml.local_features import (
    MAX_DEGREE,
    OBS_DIM,
    build_agent_observation,
    hop_distances,
    neighbours_of,
)

EPISODE_DECISIONS = 200

# Reward terms, named so the model card can quote them.
R_ARRIVAL = 1.0
R_LOOP = -1.0
R_DEADEND = -1.0
W_HOP = 1.0
W_TEAM = 0.5


class RegionalRoutingEnv(gym.Env):
    """One region's view of hop-by-hop packet forwarding."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        region_id: int,
        partition: dict[int, list[str]],
        num_nodes: int = 25,
        seed: int = 42,
        episode_decisions: int = EPISODE_DECISIONS,
        background_flows: int = 3,
        partner_policies: dict[int, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.region_id = region_id
        self.partition = {rid: list(nodes) for rid, nodes in partition.items()}
        self.region_nodes = self.partition.get(region_id, [])
        self.region_of = {
            node: rid for rid, nodes in self.partition.items() for node in nodes
        }
        self.num_nodes = num_nodes
        self.episode_decisions = episode_decisions
        self.background_flows = background_flows
        self.render_mode = render_mode

        #: Frozen policies for the other regions. Independent learners with
        #: fixed partners; None means "use the greedy heuristic partner".
        self.partner_policies = partner_policies or {}

        self._base_seed = seed + 1000 * region_id
        self._episode_index = 0
        self._rng = random.Random(self._base_seed)
        self._sim = self._new_simulator(self._base_seed)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_DEGREE)

        self._decisions = 0
        self._reset_packet_state()

    def _new_simulator(self, seed: int) -> NetworkSimulator:
        return NetworkSimulator(
            num_nodes=self.num_nodes, seed=seed, background_flows=self.background_flows
        )

    def _reset_packet_state(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._link_map: dict = {}
        self._distances: dict[str, int] = {}
        self._current = ""
        self._destination = ""
        self._path: list[str] = []
        self._visited: set[str] = set()
        self._hop_cap = 12
        self._profile = get_profile(None)
        self._prev_variance = 0.0

    # -- Gymnasium API ----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        episode_seed = seed if seed is not None else self._base_seed + self._episode_index
        self._episode_index += 1
        self._rng = random.Random(episode_seed)
        self._sim = self._new_simulator(episode_seed)
        for _ in range(self._rng.randint(0, 30)):
            self._sim.step()

        self._decisions = 0
        self._reset_packet_state()
        self._start_packet()
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        self._decisions += 1

        state = self._sim.get_state()
        neighbours = neighbours_of(self._graph, self._current)
        options = [n for n in neighbours if n not in self._visited]

        if not options:
            reward += R_DEADEND
            reward += self._finish_packet(state, delivered=False)
            self._start_packet()
            return self._observe(), float(reward), False, self._truncated(), self._info()

        nxt = self._resolve(int(action), neighbours, options)
        reward += self._hop_reward(state, self._current, nxt)
        self._advance_to(nxt)

        # Hand control to the partner policies until the packet either finishes
        # or comes back to a node this agent owns.
        while (
            self._current != self._destination
            and len(self._path) <= self._hop_cap
            and self.region_of.get(self._current, -1) != self.region_id
        ):
            partner_next = self._partner_hop(state)
            if partner_next is None:
                break
            self._advance_to(partner_next)

        delivered = self._current == self._destination
        stuck = not delivered and (
            len(self._path) > self._hop_cap
            or self.region_of.get(self._current, -1) != self.region_id
        )

        if delivered:
            reward += R_ARRIVAL
        if delivered or stuck:
            if stuck:
                reward += R_LOOP
            reward += self._finish_packet(state, delivered=delivered)
            self._start_packet()

        return self._observe(), float(reward), False, self._truncated(), self._info()

    def close(self) -> None:
        return None

    # -- packet lifecycle -------------------------------------------------

    def _start_packet(self) -> None:
        """Draw a demand whose source this agent owns, so it acts first."""
        state = self._sim.get_state()
        self._graph = build_graph(state)
        self._link_map = link_lookup(state)

        candidates = [n for n in self.region_nodes if n in self._graph]
        if not candidates:
            candidates = list(self._graph.nodes)
        source = self._rng.choice(candidates)
        others = [n for n in self._graph.nodes if n != source]
        if not others:
            self._destination = source
            self._current = source
            self._path = [source]
            self._visited = {source}
            return

        self._destination = self._rng.choice(others)
        self._distances = hop_distances(self._graph, self._destination)
        self._current = source
        self._path = [source]
        self._visited = {source}
        self._hop_cap = max(6, 3 * max(self._distances.get(source, 4), 2))
        self._profile = QOS_PROFILES[self._rng.choice(list(ALL_CLASSES))]
        utils = [link.utilization for link in state.links]
        self._prev_variance = float(np.var(utils)) if utils else 0.0

    def _advance_to(self, node: str) -> None:
        self._path.append(node)
        self._visited.add(node)
        self._current = node

    def _finish_packet(self, state, delivered: bool) -> float:
        """Register the delivered flow, tick the sim, return the team reward."""
        if delivered and len(self._path) > 1:
            self._sim.register_flow(self._path)

        new_state = self._sim.step()
        utils = [link.utilization for link in new_state.links]
        variance = float(np.var(utils)) if utils else 0.0

        # Shared team term: did the network get better or worse balanced as a
        # result of everyone's routing this round? Every agent sees the same
        # signal, which is what gives them a reason to cooperate rather than
        # each greedily grabbing the cheapest link.
        team = W_TEAM * (self._prev_variance - variance) * 10.0
        self._prev_variance = variance

        self._graph = build_graph(new_state)
        self._link_map = link_lookup(new_state)
        return float(np.clip(team, -1.0, 1.0))

    def _partner_hop(self, state) -> str | None:
        """Let the agent owning the current node choose, or fall back to greedy."""
        neighbours = neighbours_of(self._graph, self._current)
        options = [n for n in neighbours if n not in self._visited]
        if not options:
            return None

        region = self.region_of.get(self._current, -1)
        model = self.partner_policies.get(region)
        if model is None:
            # Greedy partner: move toward the destination, breaking ties on cost.
            return min(
                options,
                key=lambda n: (
                    self._distances.get(n, 99),
                    link_cost(self._link_map[frozenset((self._current, n))])
                    if frozenset((self._current, n)) in self._link_map
                    else 0.0,
                ),
            )

        observation = build_agent_observation(
            state,
            self._graph,
            self._link_map,
            self._current,
            self._destination,
            self.partition.get(region, []),
            self._distances,
            self.region_of,
            self._profile,
            include_global=False,
        )
        action, _ = model.predict(observation, deterministic=True)
        return self._resolve(int(action), neighbours, options)

    @staticmethod
    def _resolve(action: int, neighbours: list[str], options: list[str]) -> str:
        if 0 <= action < len(neighbours) and neighbours[action] in options:
            return neighbours[action]
        return options[0]

    # -- reward -----------------------------------------------------------

    def _hop_reward(self, state, current: str, nxt: str) -> float:
        """Cost of taking this hop, plus a shaping term for making progress."""
        link = self._link_map.get(frozenset((current, nxt)))
        if link is None:
            return R_DEADEND

        cost = link_cost(link) / (MAX_BASE_LATENCY * 5.0)
        reward = -W_HOP * float(np.clip(cost, 0.0, 1.0))

        # Potential-based shaping on hop distance: rewards genuine progress
        # toward the destination without changing the optimal policy.
        before = self._distances.get(current, 99)
        after = self._distances.get(nxt, 99)
        reward += 0.10 * float(np.sign(before - after))
        return reward

    # -- plumbing ---------------------------------------------------------

    def _observe(self) -> np.ndarray:
        return build_agent_observation(
            self._sim.get_state(),
            self._graph,
            self._link_map,
            self._current,
            self._destination,
            self.region_nodes,
            self._distances,
            self.region_of,
            self._profile,
            include_global=True,
        )

    def _truncated(self) -> bool:
        return self._decisions >= self.episode_decisions

    def _info(self) -> dict[str, Any]:
        return {
            "region": self.region_id,
            "decisions": self._decisions,
            "current": self._current,
            "destination": self._destination,
            "hops": len(self._path) - 1,
        }


__all__ = ["EPISODE_DECISIONS", "RegionalRoutingEnv"]
