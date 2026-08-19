"""Stress test AntColonyRouter and compare its results against Dijkstra."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.simulator import NetworkSimulator
from routing.classical.dijkstra import find_route as dijkstra_route
from routing.heuristic.aco import AntColonyRouter


def test_30_random_pairs_all_succeed() -> None:
    """
    Run ACO on 30 random src/dst pairs on a fully connected network.
    30 instead of 100 (used for Dijkstra) because ACO is far more
    expensive per call: n_ants * n_iterations path constructions per lookup.
    """
    print("[TEST] Running 30 random src/dst pairs through ACO...")
    sim = NetworkSimulator(num_nodes=10, seed=42)

    for _ in range(10):
        sim.step()

    state = sim.get_state()
    rng = random.Random(7)
    aco = AntColonyRouter(n_ants=10, n_iterations=15)

    success_count = 0
    for _ in range(30):
        src, dst = rng.sample(state.nodes, 2)
        decision = aco.find_route(state, src, dst)
        if decision.success:
            success_count += 1

    print(f"  ✓ {success_count}/30 pairs found a valid path")
    assert success_count == 30, f"Expected all pairs to succeed, got {success_count}/30"


def test_same_source_and_destination() -> None:
    """
    Edge case: src == dst. The ant's while-loop condition `current != dst`
    is already false on the first check, so the path should immediately
    be just the starting node with zero cost.
    """
    print("[TEST] Testing src == dst edge case...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    state = sim.get_state()
    node = state.nodes[0]
    aco = AntColonyRouter(n_ants=5, n_iterations=5)

    decision = aco.find_route(state, node, node)

    print(f"  ✓ path={decision.path}, cost={decision.total_latency}")
    assert decision.success, "src == dst should succeed trivially"
    assert decision.path == [node], f"Expected single-node path, got {decision.path}"
    assert decision.total_latency == 0.0, "Cost from a node to itself should be 0"


def test_unreachable_after_failures() -> None:
    """
    Disconnect a node entirely by failing all its links, then confirm
    ACO correctly reports success=False instead of looping forever or
    crashing when every ant dead-ends.
    """
    print("[TEST] Testing unreachable node after link failures...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    state = sim.get_state()

    isolated_node = state.nodes[0]
    neighbors = [
        (link.source, link.target)
        for link in state.links
        if isolated_node in (link.source, link.target)
    ]

    for src, dst in neighbors:
        sim.inject_failure(src, dst)

    state = sim.get_state()
    other_node = next(n for n in state.nodes if n != isolated_node)
    aco = AntColonyRouter(n_ants=10, n_iterations=10)

    decision = aco.find_route(state, isolated_node, other_node)

    print(f"  ✓ success={decision.success} (expected False)")
    assert decision.success is False, "Isolated node should be unreachable"
    assert decision.path == [], "Failed routes should have an empty path"
    assert decision.total_latency == float("inf"), "Failed routes should have infinite cost"


def test_invalid_node_names() -> None:
    """
    Passing a node name that doesn't exist in the network should fail
    gracefully rather than crashing when building the adjacency list.
    """
    print("[TEST] Testing invalid/unknown node names...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    state = sim.get_state()
    real_node = state.nodes[0]
    aco = AntColonyRouter()

    decision = aco.find_route(state, "R999", real_node)

    print(f"  ✓ success={decision.success} (expected False)")
    assert decision.success is False, "Unknown source node should fail gracefully"


def test_pheromone_reset_clears_history() -> None:
    """
    After calling reset_pheromones(), the internal pheromone dict should
    be empty, confirming learned trail strength doesn't leak between
    unrelated routing sessions (e.g. between training episodes later).
    """
    print("[TEST] Testing reset_pheromones()...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    state = sim.get_state()
    aco = AntColonyRouter(n_ants=5, n_iterations=5)

    src, dst = state.nodes[0], state.nodes[3]
    aco.find_route(state, src, dst)

    assert len(aco.pheromones) > 0, "Pheromones should be populated after a search"
    aco.reset_pheromones()
    assert len(aco.pheromones) == 0, "reset_pheromones() should clear all entries"

    print("  ✓ pheromones cleared successfully")


def test_aco_vs_dijkstra_comparison() -> None:
    """
    Head-to-head comparison: run both algorithms on the same 20 src/dst
    pairs and report how often ACO matches Dijkstra's optimal cost,
    how close it gets when it doesn't, and the speed difference.

    ACO is not guaranteed to find the true shortest path -- it is a
    probabilistic search -- so this test does not assert ACO == Dijkstra.
    It only asserts ACO stays reasonably close (within 25%) on average,
    and reports the numbers so the tradeoff is visible.
    """
    print("[TEST] Comparing ACO against Dijkstra on 20 shared pairs...")
    sim = NetworkSimulator(num_nodes=10, seed=42)

    for _ in range(10):
        sim.step()

    state = sim.get_state()
    rng = random.Random(55)
    aco = AntColonyRouter(n_ants=15, n_iterations=20)

    exact_matches = 0
    total_ratio = 0.0
    comparisons = 0

    dijkstra_start = time.time()
    aco_elapsed = 0.0

    for _ in range(20):
        src, dst = rng.sample(state.nodes, 2)

        d_start = time.time()
        dijkstra_decision = dijkstra_route(state, src, dst)
        _ = time.time() - d_start

        # Reset pheromones before each independent pair so leftover trails
        # from a previous, unrelated src/dst search don't bias this one.
        aco.reset_pheromones()

        a_start = time.time()
        aco_decision = aco.find_route(state, src, dst)
        aco_elapsed += time.time() - a_start
        
        if not (dijkstra_decision.success and aco_decision.success):
            continue

        comparisons += 1
        ratio = aco_decision.total_latency / dijkstra_decision.total_latency

        if abs(aco_decision.total_latency - dijkstra_decision.total_latency) < 1e-6:
            exact_matches += 1

        total_ratio += ratio

    dijkstra_elapsed = time.time() - dijkstra_start - aco_elapsed
    avg_ratio = total_ratio / comparisons if comparisons else float("inf")

    print(f"  ✓ {comparisons} valid comparisons made")
    print(f"  ✓ ACO matched Dijkstra's optimal cost exactly in {exact_matches}/{comparisons} cases")
    print(f"  ✓ Average ACO/Dijkstra cost ratio: {avg_ratio:.3f} (1.0 = identical, lower is impossible)")
    print(f"  ✓ Dijkstra total time: {dijkstra_elapsed:.4f}s | ACO total time: {aco_elapsed:.4f}s")
    print(f"  ✓ ACO was ~{aco_elapsed / max(dijkstra_elapsed, 1e-9):.0f}x slower than Dijkstra")

    assert avg_ratio < 1.25, (
        f"ACO paths are too far from optimal on average (ratio={avg_ratio:.3f}); "
        "consider raising n_ants/n_iterations or tuning alpha/beta"
    )


def test_performance_under_load() -> None:
    """
    Run 20 route lookups back-to-back and report throughput. Kept small
    relative to Dijkstra's 500-lookup test since each ACO call internally
    runs n_ants * n_iterations path constructions.
    """
    print("[TEST] Measuring ACO performance over 20 lookups...")
    sim = NetworkSimulator(num_nodes=10, seed=42)

    for _ in range(20):
        sim.step()

    state = sim.get_state()
    rng = random.Random(123)
    aco = AntColonyRouter(n_ants=10, n_iterations=15)

    start = time.time()
    for _ in range(20):
        src, dst = rng.sample(state.nodes, 2)
        aco.find_route(state, src, dst)
    elapsed = time.time() - start

    print(f"  ✓ 20 lookups in {elapsed:.3f}s ({20 / elapsed:.1f} lookups/sec)")


if __name__ == "__main__":
    try:
        test_30_random_pairs_all_succeed()
        test_same_source_and_destination()
        test_unreachable_after_failures()
        test_invalid_node_names()
        test_pheromone_reset_clears_history()
        test_aco_vs_dijkstra_comparison()
        test_performance_under_load()
        print("\n✅ All ACO stress tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)