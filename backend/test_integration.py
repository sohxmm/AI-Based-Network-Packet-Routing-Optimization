"""End-to-end routing integration check for the current simulator state."""

from __future__ import annotations

import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from router.aco import AntColonyRouter
from router.bellman_ford import find_route as bellman_ford_route
from router.dijkstra import find_route as dijkstra_route
from router.gnn_router import GNNRouter
from router.rl_agent import RLRouter
from simulator.network_sim import NetworkSimulator


def main() -> None:
    """Run 100 route decisions across all five algorithms and report success."""
    simulator = NetworkSimulator(seed=42)
    rng = random.Random(42)
    aco_router = AntColonyRouter()
    rl_router = RLRouter()
    gnn_router = GNNRouter()
    algorithms = {
        "dijkstra": dijkstra_route,
        "bellman_ford": bellman_ford_route,
        "aco": aco_router.find_path,
        "rl": rl_router.predict,
        "gnn": gnn_router.predict,
    }
    totals = {name: {"success": 0, "latency": 0.0, "count": 0} for name in algorithms}

    for _ in range(25):
        state = simulator.step()
        source, destination = rng.sample(state.nodes, 2)

        for name, find_route in algorithms.items():
            decision = find_route(state, source, destination)
            totals[name]["count"] += 1
            if decision.success:
                totals[name]["success"] += 1
                totals[name]["latency"] += decision.total_latency

    print("Algorithm      Success Rate  Avg Latency")
    print("-" * 44)

    total_success = 0
    total_count = 0
    for name, values in totals.items():
        count = values["count"]
        success = values["success"]
        total_success += success
        total_count += count
        avg_latency = values["latency"] / success if success else 0.0
        print(f"{name:<14}{success / count:>10.1%}  {avg_latency:>10.2f} ms")

    overall_success = total_success / total_count
    print("-" * 44)
    print(f"Overall success: {overall_success:.1%}")
    assert overall_success >= 0.95, "Integration success rate must stay above 95%."


if __name__ == "__main__":
    main()

