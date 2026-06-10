"""Smoke test all Phase 2 routing algorithms on the same network state."""

from __future__ import annotations

import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.aco import AntColonyRouter
from router.bellman_ford import find_route as bellman_ford_route
from router.dijkstra import find_route as dijkstra_route
from router.rl_agent import RLRouter
from simulator.network_sim import NetworkSimulator


def main() -> None:
    """Create a simulator state and compare all routing algorithms."""
    simulator = NetworkSimulator(seed=42)
    for _ in range(5):
        simulator.step()

    state = simulator.get_state()
    rng = random.Random(42)
    source, destination = rng.sample(state.nodes, 2)

    aco_router = AntColonyRouter()
    rl_router = RLRouter()

    decisions = [
        dijkstra_route(state, source, destination),
        bellman_ford_route(state, source, destination),
        aco_router.find_path(state, source, destination),
        rl_router.predict(state, source, destination),
    ]

    print(f"Routing from {source} to {destination} at step {state.step_count}")
    print("Algorithm      Success  Cost(ms)  Avg Util  Path")
    print("-" * 72)

    for decision in decisions:
        path = " -> ".join(decision.path) if decision.path else "no path"
        print(
            f"{decision.algorithm:<14}"
            f"{str(decision.success):<9}"
            f"{decision.total_latency:<10.2f}"
            f"{decision.avg_utilization:<10.2f}"
            f"{path}"
        )

    assert all(decision.success for decision in decisions), "All routers must find a valid path."


if __name__ == "__main__":
    main()
