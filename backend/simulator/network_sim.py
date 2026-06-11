from __future__ import annotations

import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.data_models import LinkState, NetworkState


class NetworkSimulator:
    """Simulate a dynamic packet-routing network as a weighted graph."""

    def __init__(self, num_nodes: int = 10, seed: int = 42) -> None:
        self.num_nodes = num_nodes
        self.seed = seed
        self.random = random.Random(seed)
        self.graph = nx.Graph()
        self.failed_edges: dict[tuple[str, str], dict[str, Any]] = {}
        self.step_count = 0
        self.congestion_link: tuple[str, str] | None = None
        self.congestion_remaining = 0
        self._build_topology()

    def _build_topology(self) -> None:
        """Create routers in a ring, then add random extra links."""
        nodes = [f"R{i}" for i in range(1, self.num_nodes + 1)]
        self.graph.add_nodes_from(nodes)

        for i in range(self.num_nodes):
            self._add_random_link(nodes[i], nodes[(i + 1) % self.num_nodes])

        target_edges = min(20, self.num_nodes * (self.num_nodes - 1) // 2)
        extra_links = max(0, target_edges - self.graph.number_of_edges())

        while extra_links > 0:
            src = self.random.choice(nodes)
            dst = self.random.choice(nodes)

            if src != dst and not self.graph.has_edge(src, dst):
                self._add_random_link(src, dst)
                extra_links -= 1

    def step(self) -> NetworkState:
        """Advance the simulation by one tick and return the new state."""
        self.step_count += 1
        self._maybe_start_congestion_burst()

        for src, dst, data in self.graph.edges(data=True):
            utilization = data["utilization"] + self.random.gauss(0, 0.05)

            if self.congestion_link == self._edge_key(src, dst):
                utilization = max(utilization, self.random.uniform(0.85, 1.0))

            utilization = max(0.0, min(1.0, utilization))
            data["utilization"] = utilization
            data["queue_size"] = int(utilization * 100)
            data["packet_loss_rate"] = max(0, utilization - 0.7) * 0.2

        if self.congestion_remaining > 0:
            self.congestion_remaining -= 1
            if self.congestion_remaining == 0:
                self.congestion_link = None

        return self.get_state()

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

    def inject_failure(self, src: str, dst: str) -> None:
        """Remove a link temporarily to simulate a network failure."""
        self._validate_node(src)
        self._validate_node(dst)

        if not self.graph.has_edge(src, dst):
            raise ValueError(f"Link {src}-{dst} does not exist or is already failed.")

        key = self._edge_key(src, dst)
        self.failed_edges[key] = dict(self.graph[src][dst])
        self.graph.remove_edge(src, dst)

    def restore_link(self, src: str, dst: str) -> None:
        """Restore a failed link using its saved edge attributes."""
        self._validate_node(src)
        self._validate_node(dst)

        key = self._edge_key(src, dst)
        if key not in self.failed_edges:
            raise ValueError(f"Link {src}-{dst} is not currently failed.")

        self.graph.add_edge(src, dst, **self.failed_edges.pop(key))

    def get_candidate_paths(self, src: str, dst: str, k: int = 5) -> list[list[str]]:
        """Return up to k simple candidate paths ordered by weighted cost."""
        self._validate_node(src)
        self._validate_node(dst)

        try:
            paths = nx.shortest_simple_paths(
                self.graph,
                src,
                dst,
                weight=lambda u, v, _: self.get_edge_weight(u, v),
            )
            return [path for _, path in zip(range(k), paths)]
        except nx.NetworkXNoPath:
            return []

    def get_edge_weight(self, src: str, dst: str) -> float:
        """Calculate congestion-adjusted latency for one graph edge."""
        if not self.graph.has_edge(src, dst):
            raise ValueError(f"Link {src}-{dst} does not exist.")

        data = self.graph[src][dst]
        utilization = data["utilization"]
        return data["base_latency"] * (1 + 4 * utilization ** 2)

    def to_dict(self) -> dict[str, Any]:
        """Return the current state in a JSON-serializable dictionary."""
        state = self.get_state()
        return {
            "nodes": state.nodes,
            "links": [asdict(link) for link in state.links],
            "timestamp": state.timestamp,
            "step_count": state.step_count,
        }

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
            packet_loss_rate=max(0, utilization - 0.7) * 0.2,
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
        return tuple(sorted((src, dst)))


if __name__ == "__main__":
    sim = NetworkSimulator()

    for _ in range(20):
        state = sim.step()

        print(f"Step {state.step_count}: {len(state.nodes)} nodes, {len(state.links)} links")
        for link in state.links[:3]:
            print(
                f"  {link.source}-{link.target} "
                f"latency={link.base_latency:.1f}ms "
                f"utilization={link.utilization:.2f} "
                f"queue={link.queue_size} "
                f"loss={link.packet_loss_rate:.4f}"
            )
        print("-" * 30)
