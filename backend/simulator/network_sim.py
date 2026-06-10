from __future__ import annotations

import random
import time
import networkx as nx

from simulator.data_models import (
    LinkState,
    NetworkState,
)


class NetworkSimulator:

    def __init__(
        self,
        num_nodes: int = 10,
        seed: int = 42
    ):
        self.num_nodes = num_nodes
        self.seed = seed

        random.seed(seed)

        self.graph = nx.Graph()

        self.step_count = 0

        self.congestion_link = None
        self.congestion_remaining = 0

        self._build_topology()

    def _build_topology(self) -> None:

        nodes = [
            f"R{i}"
            for i in range(1, self.num_nodes + 1)
        ]

        self.graph.add_nodes_from(nodes)

        # Ring topology

        for i in range(self.num_nodes):

            src = nodes[i]
            dst = nodes[(i + 1) % self.num_nodes]

            self.graph.add_edge(
                src,
                dst,
                base_latency=random.randint(5, 25),
                bandwidth=random.choice([100, 500, 1000]),
                utilization=random.uniform(0.1, 0.5),
                queue_size=random.randint(0, 20),
                packet_loss_rate=0.0,
            )

        # Random extra links

        extra_links = 10

        while extra_links > 0:

            src = random.choice(nodes)
            dst = random.choice(nodes)

            if src != dst and not self.graph.has_edge(src, dst):

                self.graph.add_edge(
                    src,
                    dst,
                    base_latency=random.randint(5, 25),
                    bandwidth=random.choice([100, 500, 1000]),
                    utilization=random.uniform(0.1, 0.5),
                    queue_size=random.randint(0, 20),
                    packet_loss_rate=0.0,
                )

                extra_links -= 1

    def step(self) -> NetworkState:

        self.step_count += 1

        for _, _, data in self.graph.edges(data=True):

            utilization = (
                data["utilization"]
                + random.gauss(0, 0.05)
            )

            utilization = max(
                0.0,
                min(1.0, utilization)
            )

            data["utilization"] = utilization

            # Queue size grows with traffic

            data["queue_size"] = int(
                utilization * 100
            )

            # Packet loss begins after 70% utilization

            data["packet_loss_rate"] = (
                max(0, utilization - 0.7)
                * 0.2
            )

        return self.get_state()

    def get_state(self) -> NetworkState:

        links = []

        for u, v, data in self.graph.edges(data=True):

            links.append(
                LinkState(
                    source=u,
                    target=v,
                    base_latency=data["base_latency"],
                    bandwidth=data["bandwidth"],
                    utilization=data["utilization"],
                    queue_size=data["queue_size"],
                    packet_loss_rate=data["packet_loss_rate"],
                )
            )

        return NetworkState(
            nodes=list(self.graph.nodes),
            links=links,
            timestamp=time.time(),
            step_count=self.step_count,
        )


if __name__ == "__main__":

    sim = NetworkSimulator()

    for _ in range(5):

        state = sim.step()

        first_link = state.links[0]

        print(
            f"Step {state.step_count}"
        )

        print(
            f"{first_link.source}-{first_link.target}"
        )

        print(
            f"Utilization: {first_link.utilization:.2f}"
        )

        print(
            f"Queue: {first_link.queue_size}"
        )

        print(
            f"Loss: {first_link.packet_loss_rate:.4f}"
        )

        print("-" * 30)