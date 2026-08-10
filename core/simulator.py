"""NetworkSimulator — a closed-loop, small-world packet routing testbed.

Three properties of this simulator were changed as a direct result of the
technical audit, and each one invalidates a previously-published result:

1. **The loop is closed.** Routing decisions now add load to the links they
   traverse (:meth:`register_flow`). Previously utilization was a random walk
   independent of routing, which made per-path latency minimisation exactly
   optimal — Dijkstra solves that exactly, so no learned policy could ever win.
   With flow feedback, greedy shortest-path self-congests and load-aware
   policies have room to beat it.

2. **The topology is small-world, not a ring.** The old generator capped edges
   at ``min(50, ...)``, so a 100-node graph was a pure ring: degree 2,
   diameter 50, exactly two simple paths between any pair. Edges now scale with
   ``avg_degree`` and ring-lattice shortcuts are added before random extras.

3. **Utilization is autocorrelated with a diurnal cycle**, rather than a pure
   random walk. Predicting a random walk is unfalsifiable busywork — persistence
   is Bayes-optimal by construction. An AR(1) process around a sinusoidal
   baseline gives a forecaster something real to learn while keeping persistence
   a strong baseline.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict
from typing import Any

import networkx as nx

from core.cost import raw_edge_cost
from core.models import LinkState, NetworkState

logger = logging.getLogger(__name__)


class NetworkSimulator:
    """Simulate a dynamic packet-routing network as a weighted graph."""

    # Offered-load baseline: a sine wave in [CENTRE - AMP, CENTRE + AMP].
    DIURNAL_CENTRE = 0.30
    DIURNAL_AMPLITUDE = 0.18

    def __init__(
        self,
        num_nodes: int = 25,
        seed: int = 42,
        avg_degree: int = 4,
        background_flows: int = 3,
        load_per_flow: float = 0.06,
        flow_decay: float = 0.90,
        ar_coefficient: float = 0.85,
        diurnal_period: int = 40,
        noise_sigma: float = 0.03,
    ) -> None:
        self.num_nodes = num_nodes
        self.seed = seed
        self.avg_degree = avg_degree
        self.random = random.Random(seed)
        self.graph = nx.Graph()
        self.failed_edges: dict[tuple[str, str], dict[str, Any]] = {}
        self.step_count = 0
        self.congestion_link: tuple[str, str] | None = None
        self.congestion_remaining = 0

        # -- closed-loop flow accounting ------------------------------------
        self.background_flows = background_flows
        self.load_per_flow = load_per_flow
        self.flow_decay = flow_decay
        self.flow_load: dict[tuple[str, str], float] = {}

        # -- traffic dynamics -----------------------------------------------
        self.ar_coefficient = ar_coefficient
        self.diurnal_period = diurnal_period
        self.noise_sigma = noise_sigma

        self._build_topology()
        self._log_topology_stats()

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _build_topology(self) -> None:
        """Build a small-world ring lattice: ring, then shortcuts, then randoms."""
        nodes = [f"R{i}" for i in range(1, self.num_nodes + 1)]
        self.graph.add_nodes_from(nodes)

        # 1. Base ring guarantees connectivity.
        for index in range(self.num_nodes):
            self._add_random_link(nodes[index], nodes[(index + 1) % self.num_nodes])

        # Edge budget scales with node count instead of the old min(50, ...) cap,
        # which turned any topology above 50 nodes into a degree-2 ring.
        target_edges = max(self.num_nodes, self.num_nodes * self.avg_degree // 2)

        # 2. Ring-lattice shortcuts (i -> i+2) give small-world structure.
        shortcut_budget = max(0, (target_edges - self.graph.number_of_edges()) // 2)
        for index in range(self.num_nodes):
            if shortcut_budget <= 0:
                break
            src, dst = nodes[index], nodes[(index + 2) % self.num_nodes]
            if src != dst and not self.graph.has_edge(src, dst):
                self._add_random_link(src, dst)
                shortcut_budget -= 1

        # 3. Random long-range links fill the remaining budget.
        extra_links = max(0, target_edges - self.graph.number_of_edges())
        attempts = 0
        max_attempts = extra_links * 200 + 1000
        while extra_links > 0 and attempts < max_attempts:
            attempts += 1
            src = self.random.choice(nodes)
            dst = self.random.choice(nodes)
            if src != dst and not self.graph.has_edge(src, dst):
                self._add_random_link(src, dst)
                extra_links -= 1

    def _log_topology_stats(self) -> None:
        """Record the shape of the generated graph so a ring can never hide again."""
        stats = self.topology_stats()
        logger.info(
            "Topology: %d nodes, %d edges, avg_degree %.2f, diameter %s, connected=%s",
            stats["num_nodes"],
            stats["num_edges"],
            stats["avg_degree"],
            stats["diameter"],
            stats["is_connected"],
        )

    def topology_stats(self) -> dict[str, Any]:
        """Return node/edge/degree/diameter statistics for the current graph."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        connected = num_nodes > 0 and nx.is_connected(self.graph)
        try:
            diameter = nx.diameter(self.graph) if connected else None
        except (nx.NetworkXError, ValueError):
            diameter = None
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": (2 * num_edges / num_nodes) if num_nodes else 0.0,
            "diameter": diameter,
            "is_connected": bool(connected),
        }

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def step(self) -> NetworkState:
        """Advance the simulation by one tick and return the new state."""
        self.step_count += 1
        self._maybe_start_congestion_burst()
        self._decay_flow_load()
        self._inject_background_traffic()

        phase_base = 2 * math.pi * (self.step_count % self.diurnal_period) / self.diurnal_period

        for src, dst, data in self.graph.edges(data=True):
            key = self._edge_key(src, dst)

            # Offered load for this link: a deterministic diurnal baseline
            # (offset per link so links peak at different times), any sustained
            # scenario bias, and the accumulated load of flows actually routed
            # over it. Utilization relaxes toward offered load as an AR(1)
            # process, so the flow term maps 1:1 onto steady-state utilization
            # instead of being amplified by 1/(1 - ar_coefficient).
            phase = phase_base + data.get("phase_offset", 0.0)
            offered = (
                self.DIURNAL_CENTRE
                + self.DIURNAL_AMPLITUDE * math.sin(phase)
                + data.get("congestion_bias", 0.0)
                + self.flow_load.get(key, 0.0)
            )

            previous = data["utilization"]
            utilization = (
                self.ar_coefficient * previous
                + (1 - self.ar_coefficient) * offered
                + self.random.gauss(0, self.noise_sigma)
            )

            if self.congestion_link == key:
                utilization = max(utilization, self.random.uniform(0.85, 1.0))

            utilization = max(0.0, min(1.0, utilization))
            data["utilization"] = utilization
            data["queue_size"] = int(utilization * 100)
            data["packet_loss_rate"] = max(0.0, utilization - 0.7) * 0.2

        if self.congestion_remaining > 0:
            self.congestion_remaining -= 1
            if self.congestion_remaining == 0:
                self.congestion_link = None

        return self.get_state()

    def reset(self) -> NetworkState:
        """Reset the simulation to its initial state."""
        self.random = random.Random(self.seed)
        self.graph.clear()
        self.failed_edges.clear()
        self.flow_load.clear()
        self.step_count = 0
        self.congestion_link = None
        self.congestion_remaining = 0
        self._build_topology()
        return self.get_state()

    # ------------------------------------------------------------------
    # Closed-loop flow accounting
    # ------------------------------------------------------------------

    def register_flow(self, path: list[str], demand: float = 1.0) -> None:
        """Record that a flow was routed over *path*, adding load to each link.

        This is what makes the simulator closed-loop: routing decisions now
        change the network state that subsequent routing decisions observe.
        """
        if not path or len(path) < 2:
            return
        for index in range(len(path) - 1):
            key = self._edge_key(path[index], path[index + 1])
            if self.graph.has_edge(*key):
                self.flow_load[key] = (
                    self.flow_load.get(key, 0.0) + self.load_per_flow * demand
                )

    def _decay_flow_load(self) -> None:
        """Exponentially forget injected load so congestion is transient."""
        for key in list(self.flow_load):
            self.flow_load[key] *= self.flow_decay
            if self.flow_load[key] < 1e-4:
                del self.flow_load[key]

    def _inject_background_traffic(self) -> None:
        """Route a few random demands greedily so the network carries load when idle."""
        if self.background_flows <= 0:
            return
        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            return
        for _ in range(self.background_flows):
            src, dst = self.random.sample(nodes, 2)
            paths = self.get_candidate_paths(src, dst, k=1)
            if paths:
                self.register_flow(paths[0], demand=0.5)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self) -> NetworkState:
        """Return the current network state without advancing the simulation."""
        links = [
            LinkState(
                source=u,
                target=v,
                base_latency=data["base_latency"],
                bandwidth=data["bandwidth"],
                utilization=data["utilization"],
                queue_size=data["queue_size"],
                packet_loss_rate=data["packet_loss_rate"],
            )
            for u, v, data in self.graph.edges(data=True)
        ]
        return NetworkState(
            nodes=list(self.graph.nodes),
            links=links,
            timestamp=time.time(),
            step_count=self.step_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the current state in a JSON-serializable dictionary."""
        state = self.get_state()
        return {
            "nodes": state.nodes,
            "links": [asdict(link) for link in state.links],
            "timestamp": state.timestamp,
            "step_count": state.step_count,
        }

    # ------------------------------------------------------------------
    # Failure injection
    # ------------------------------------------------------------------

    def inject_failure(self, src: str, dst: str) -> None:
        """Remove a link temporarily to simulate a network failure."""
        self._validate_node(src)
        self._validate_node(dst)

        if not self.graph.has_edge(src, dst):
            raise ValueError(f"Link {src}-{dst} does not exist or is already failed.")

        key = self._edge_key(src, dst)
        self.failed_edges[key] = dict(self.graph[src][dst])
        self.graph.remove_edge(src, dst)
        self.flow_load.pop(key, None)

    def restore_link(self, src: str, dst: str) -> None:
        """Restore a failed link using its saved edge attributes."""
        self._validate_node(src)
        self._validate_node(dst)

        key = self._edge_key(src, dst)
        if key not in self.failed_edges:
            raise ValueError(f"Link {src}-{dst} is not currently failed.")

        self.graph.add_edge(src, dst, **self.failed_edges.pop(key))

    def can_fail_safely(self, src: str, dst: str) -> bool:
        """True when removing this edge leaves the graph connected."""
        if not self.graph.has_edge(src, dst):
            return False
        probe = self.graph.copy()
        probe.remove_edge(src, dst)
        return nx.is_connected(probe)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def get_candidate_paths(self, src: str, dst: str, k: int = 5) -> list[list[str]]:
        """Return up to k simple candidate paths ordered by weighted cost."""
        self._validate_node(src)
        self._validate_node(dst)
        if src == dst:
            return []
        try:
            paths = nx.shortest_simple_paths(
                self.graph,
                src,
                dst,
                weight=lambda u, v, _: self.get_edge_weight(u, v),
            )
            return [path for _, path in zip(range(k), paths)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_edge_weight(self, src: str, dst: str) -> float:
        """Calculate congestion-adjusted latency for one graph edge."""
        if not self.graph.has_edge(src, dst):
            raise ValueError(f"Link {src}-{dst} does not exist.")
        data = self.graph[src][dst]
        return raw_edge_cost(data["base_latency"], data["utilization"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_random_link(self, src: str, dst: str) -> None:
        """Add a link with reproducible random network attributes."""
        utilization = self.random.uniform(0.1, 0.5)
        self.graph.add_edge(
            src,
            dst,
            base_latency=self.random.randint(5, 25),
            bandwidth=self.random.choice([100, 500, 1000]),
            utilization=utilization,
            queue_size=int(utilization * 100),
            packet_loss_rate=max(0.0, utilization - 0.7) * 0.2,
            # Per-link phase so links peak at different points in the cycle.
            phase_offset=self.random.uniform(0, 2 * math.pi),
            congestion_bias=0.0,
        )

    def _maybe_start_congestion_burst(self) -> None:
        """Trigger a 10-step utilization spike roughly every 50 steps."""
        if self.congestion_remaining > 0 or self.step_count % 50 != 0:
            return
        edges = list(self.graph.edges())
        if not edges:
            return
        self.congestion_link = self._edge_key(*self.random.choice(edges))
        self.congestion_remaining = 10

    def _validate_node(self, node: str) -> None:
        """Raise a clear error if a router node is not in the topology."""
        if node not in self.graph:
            raise ValueError(f"Node {node} does not exist.")

    @staticmethod
    def _edge_key(src: str, dst: str) -> tuple[str, str]:
        """Normalize undirected edge keys for lookups."""
        return tuple(sorted((src, dst)))  # type: ignore[return-value]
