"""MultiAgentRouter — decentralized hop-by-hop routing by regional PPO policies.

What this was, and why it was renamed in the audit's eyes
--------------------------------------------------------
The previous implementation was N independently trained PPO agents that each
saw the *entire* global link vector and each emitted a *complete* end-to-end
path, with the acting agent chosen by a lookup on the source node. That is a
mixture-of-experts with a fixed gating function. It had no joint action space,
no inter-agent communication, no credit assignment, and its execution was not
decentralized in any sense. Describing it as "centralized-critic,
decentralized-execution multi-agent RL" was not defensible.

What it is now
--------------
Genuinely decentralized execution:

* **Local observation.** Each agent sees only the node the packet is at, the
  destination expressed relatively, and that node's incident links. The
  observation width is a constant, independent of the size of the network.
* **Next-hop action.** An agent chooses one neighbour, not a whole path.
* **Control transfers.** The packet moves, and whichever region owns the new
  node acts next. A single route is therefore produced by several agents in
  sequence, which is what makes it multi-agent at all.
* **Centralized critic.** During training the value function additionally sees a
  fixed-width global summary that the policy network never receives. See
  ``ml/training/train_regional.py``.

What is still honest to say about it
------------------------------------
There is still no explicit inter-agent message passing, and agents are trained
with independent PPO rather than a shared critic across agents. It is
decentralized-execution / centralized-training with independent learners — a
real and standard MARL setting, but not a joint-action solver. The model card
states this in the same terms.

The other fix here is the partition. The old router built a *fresh 25-node
simulator* to derive its regions regardless of the topology it was actually
serving, so on the 100-node scenario every node above R25 mapped to region -1
and forced a heuristic fallback — the audit measured ``fallback_rate = 0.75``.
Regions are now derived from the live state and recomputed whenever the node set
changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx

from core.models import NetworkState, RoutingDecision
from core.paths import (
    build_decision,
    build_graph,
    candidate_paths,
    failed_decision,
    link_lookup,
    path_cost,
)
from core.qos import QoSProfile, evaluate_path
from ml.environments.partition import build_region_lookup, partition_network
from ml.local_features import build_agent_observation, hop_distances, neighbours_of
from ml.model_registry import MODEL_DIR, regional_path
from routing.base import Router

logger = logging.getLogger(__name__)

_ppo_class: Any = None


def _try_import_ppo() -> bool:
    global _ppo_class
    if _ppo_class is not None:
        return True
    try:
        from stable_baselines3 import PPO

        _ppo_class = PPO
        return True
    except ImportError:
        return False


class MultiAgentRouter(Router):
    """Route hop by hop, handing control to whichever region owns each node."""

    name = "multi_agent"
    label = "Multi-Agent RL"
    description = (
        "Regional PPO agents each choose one next hop from purely local "
        "observations; control transfers between regions along the path."
    )

    #: A route longer than this is abandoned as pathological.
    MAX_HOP_MULTIPLIER = 3

    def __init__(self, seed: int = 42, k_paths: int = 5) -> None:
        self._seed = seed
        self._k_paths = k_paths
        self._models: dict[int, Any] = {}
        self._partition: dict[int, list[str]] = {}
        self._region_of: dict[str, int] = {}
        self._partition_key: tuple[str, ...] | None = None
        self._regions_loaded: list[int] = []

    # -- model management -------------------------------------------------

    def load_models(self, model_dir: str | Path | None = None) -> None:
        """Load every regional policy present on disk."""
        directory = Path(model_dir) if model_dir else MODEL_DIR
        if not _try_import_ppo():
            raise ImportError("stable-baselines3 is required for the multi-agent router.")

        self._models.clear()
        self._regions_loaded.clear()

        for region_id in range(16):  # generous upper bound on region count
            candidate = (
                directory / f"multi_agent_region_{region_id}.zip"
                if model_dir
                else regional_path(region_id)
            )
            if not candidate.exists():
                continue
            self._models[region_id] = _ppo_class.load(str(candidate), device="cpu")
            self._regions_loaded.append(region_id)

        if self._regions_loaded:
            logger.info(
                "MultiAgentRouter loaded %d regional policies: %s",
                len(self._regions_loaded),
                self._regions_loaded,
            )
        else:
            raise FileNotFoundError(
                f"No regional policies found in {directory}. "
                "Train them with: python -m ml.training.train_regional"
            )

    def try_load_models(self, model_dir: str | Path | None = None) -> bool:
        """Load the policies, reporting failure loudly instead of swallowing it."""
        try:
            self.load_models(model_dir)
            return bool(self._regions_loaded)
        except FileNotFoundError as exc:
            logger.warning(
                "MultiAgentRouter: %s Falling back to the congestion-aware heuristic.",
                exc,
            )
            return False
        except ImportError as exc:
            logger.warning("MultiAgentRouter: ML dependencies unavailable (%s).", exc)
            return False
        except Exception:
            logger.exception("MultiAgentRouter: unexpected error loading policies.")
            return False

    @property
    def is_trained(self) -> bool:
        return bool(self._regions_loaded)

    @property
    def requires_model(self) -> bool:
        return True

    @property
    def num_regions(self) -> int:
        return len(self._partition)

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "requires_model": True,
            "regions_loaded": list(self._regions_loaded),
            "num_regions": self.num_regions,
        }

    # -- partition --------------------------------------------------------

    def ensure_partition(self, state: NetworkState) -> None:
        """(Re)derive regions from the live topology when the node set changes."""
        key = tuple(sorted(state.nodes))
        if key == self._partition_key and self._partition:
            return

        graph = nx.Graph()
        graph.add_nodes_from(state.nodes)
        for link in state.links:
            graph.add_edge(link.source, link.target)

        self._partition = partition_network(graph)
        self._region_of = build_region_lookup(self._partition)
        self._partition_key = key
        logger.info(
            "MultiAgentRouter repartitioned %d nodes into %d regions %s",
            len(state.nodes),
            len(self._partition),
            [len(v) for v in self._partition.values()],
        )

    # -- inference --------------------------------------------------------

    def find_route(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        profile: QoSProfile | None = None,
    ) -> RoutingDecision:
        profile = self.resolve_profile(profile)

        if src not in state.nodes or dst not in state.nodes or src == dst:
            return failed_decision(src, dst, self.name)

        self.ensure_partition(state)

        if not self._models:
            return self._heuristic_route(state, src, dst, profile, reason="no_models")

        graph = build_graph(state)
        if src not in graph or dst not in graph or not nx.has_path(graph, src, dst):
            return failed_decision(src, dst, self.name)

        path, agents_used, reason = self._walk(state, graph, src, dst, profile)
        if not path:
            return self._heuristic_route(state, src, dst, profile, reason=reason)

        return build_decision(
            state,
            src,
            dst,
            path,
            self.name,
            is_fallback=False,
            diagnostics={
                "qos": evaluate_path(state, path, profile).as_dict(),
                "agents_used": agents_used,
                "regions": len(self._partition),
                "decentralized": True,
            },
        )

    def _walk(
        self,
        state: NetworkState,
        graph: nx.Graph,
        src: str,
        dst: str,
        profile: QoSProfile,
    ) -> tuple[list[str], list[int], str]:
        """Walk hop by hop, letting each region's agent pick the next neighbour."""
        link_map = link_lookup(state)
        distances = hop_distances(graph, dst)
        hop_cap = self.MAX_HOP_MULTIPLIER * max(4, len(state.nodes) // 4)

        path = [src]
        visited = {src}
        agents_used: list[int] = []
        current = src

        while current != dst and len(path) <= hop_cap:
            region_id = self._region_of.get(current, -1)
            model = self._models.get(region_id)
            neighbours = neighbours_of(graph, current)
            options = [n for n in neighbours if n not in visited]
            if not options:
                return [], agents_used, "dead_end"

            if model is None:
                # This region has no trained policy. Take the greedy hop rather
                # than abandoning the walk, and record that an untrained region
                # participated so the decision can be reported honestly.
                nxt = min(options, key=lambda n: distances.get(n, 99))
                agents_used.append(-1)
            else:
                observation = build_agent_observation(
                    state,
                    graph,
                    link_map,
                    current,
                    dst,
                    self._partition.get(region_id, []),
                    distances,
                    self._region_of,
                    profile,
                    include_global=False,  # execution is decentralized
                )
                action, _ = model.predict(observation, deterministic=True)
                nxt = self._resolve_action(int(action), neighbours, options)
                agents_used.append(region_id)

            path.append(nxt)
            visited.add(nxt)
            current = nxt

        if current != dst:
            return [], agents_used, "hop_cap"
        return path, agents_used, "ok"

    @staticmethod
    def _resolve_action(action: int, neighbours: list[str], options: list[str]) -> str:
        """Map an action index onto a usable neighbour.

        The agent indexes the node's full ordered neighbour list. If it selects
        a slot that is empty or already visited, fall through to the nearest
        unvisited option rather than producing a loop.
        """
        if 0 <= action < len(neighbours):
            chosen = neighbours[action]
            if chosen in options:
                return chosen
        return options[0]

    def _heuristic_route(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        profile: QoSProfile,
        reason: str,
    ) -> RoutingDecision:
        """Congestion-aware fallback, always flagged as such."""
        paths = candidate_paths(state, src, dst, k=self._k_paths)
        if not paths:
            return failed_decision(src, dst, self.name)
        path = min(paths, key=lambda p: path_cost(state, p))
        logger.debug("MultiAgentRouter fell back to heuristic (%s)", reason)
        return build_decision(
            state,
            src,
            dst,
            path,
            self.name,
            is_fallback=True,
            diagnostics={
                "qos": evaluate_path(state, path, profile).as_dict(),
                "fallback_reason": reason,
            },
        )


__all__ = ["MultiAgentRouter"]
