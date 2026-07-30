"""Stress test Dijkstra's algorithm before it's relied on by FastAPI in Phase 3."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from router.dijkstra import find_route
from simulator.network_sim import NetworkSimulator


def test_100_random_pairs_all_succeed() -> None:
    """
    Run Dijkstra on 100 random src/dst pairs on a fully connected network.
    Every pair should find a valid path since the topology guarantees
    connectivity (ring + extra edges, no failures injected).
    """
    print("[TEST] Running 100 random src/dst pairs...")
    sim = NetworkSimulator(num_nodes=10, seed=42)

    for _ in range(10):
        sim.step()

    state = sim.get_state()
    rng = random.Random(7)

    success_count = 0
    for _ in range(100):
        src, dst = rng.sample(state.nodes, 2)
        decision = find_route(state, src, dst)
        if decision.success:
            success_count += 1

    print(f"  ✓ {success_count}/100 pairs found a valid path")
    assert success_count == 100, f"Expected all pairs to succeed, got {success_count}/100"


def test_same_source_and_destination() -> None:
    """
    Edge case: src == dst. Path reconstruction should not crash and
    should return a single-node path with zero cost.
    """
    print("[TEST] Testing src == dst edge case...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    state = sim.get_state()
    node = state.nodes[0]

    decision = find_route(state, node, node)

    print(f"  ✓ path={decision.path}, cost={decision.total_latency}")
    assert decision.success, "src == dst should succeed trivially"
    assert decision.path == [node], f"Expected single-node path, got {decision.path}"
    assert decision.total_latency == 0.0, "Cost from a node to itself should be 0"


def test_unreachable_after_failures() -> None:
    """
    Disconnect a node entirely by failing all its links, then confirm
    Dijkstra correctly reports success=False instead of crashing.
    """
    print("[TEST] Testing unreachable node after link failures...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    state = sim.get_state()

    isolated_node = state.nodes[0]
    neighbors = [
        (link.source, link.target)
        for link in state.links
        if link.source == isolated_node or link.target == isolated_node
    ]

    for src, dst in neighbors:
        sim.inject_failure(src, dst)

    state = sim.get_state()
    other_node = next(n for n in state.nodes if n != isolated_node)

    decision = find_route(state, isolated_node, other_node)

    print(f"  ✓ success={decision.success} (expected False)")
    assert decision.success is False, "Isolated node should be unreachable"
    assert decision.path == [], "Failed routes should have an empty path"
    assert decision.total_latency == float("inf"), "Failed routes should have infinite cost"


def test_invalid_node_names() -> None:
    """
    Passing a node name that doesn't exist in the network should fail
    gracefully rather than raising a KeyError.
    """
    print("[TEST] Testing invalid/unknown node names...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    state = sim.get_state()
    real_node = state.nodes[0]

    decision = find_route(state, "R999", real_node)

    print(f"  ✓ success={decision.success} (expected False)")
    assert decision.success is False, "Unknown source node should fail gracefully"


def test_matches_networkx_dijkstra() -> None:
    """
    Cross-check correctness: build the same graph in NetworkX and compare
    its built-in dijkstra_path_length against our from-scratch result.
    This validates the heapq implementation is actually correct, not just
    crash-free.
    """
    print("[TEST] Cross-checking against networkx.dijkstra_path_length...")
    sim = NetworkSimulator(num_nodes=10, seed=42)

    for _ in range(15):
        sim.step()

    state = sim.get_state()
    rng = random.Random(99)

    graph = nx.Graph()
    graph.add_nodes_from(state.nodes)
    for link in state.links:
        weight = link.base_latency * (1 + 4 * link.utilization ** 2)
        graph.add_edge(link.source, link.target, weight=weight)

    mismatches = 0
    for _ in range(30):
        src, dst = rng.sample(state.nodes, 2)
        ours = find_route(state, src, dst)

        if not ours.success:
            continue

        nx_cost = nx.dijkstra_path_length(graph, src, dst, weight="weight")
        if abs(ours.total_latency - nx_cost) > 1e-9:
            mismatches += 1
            print(f"  ✗ Mismatch for {src}->{dst}: ours={ours.total_latency}, nx={nx_cost}")

    print(f"  ✓ {30 - mismatches}/30 comparisons matched NetworkX exactly")
    assert mismatches == 0, f"{mismatches} path costs disagreed with NetworkX"


def test_performance_under_load() -> None:
    """
    Run 500 route lookups back-to-back and confirm it stays fast enough
    for a live dashboard (sub-millisecond average on a 10-node network).
    """
    print("[TEST] Measuring performance over 500 lookups...")
    sim = NetworkSimulator(num_nodes=10, seed=42)

    for _ in range(20):
        sim.step()

    state = sim.get_state()
    rng = random.Random(123)

    start = time.time()
    for _ in range(500):
        src, dst = rng.sample(state.nodes, 2)
        find_route(state, src, dst)
    elapsed = time.time() - start

    print(f"  ✓ 500 lookups in {elapsed:.3f}s ({500 / elapsed:.0f} lookups/sec)")


if __name__ == "__main__":
    try:
        test_100_random_pairs_all_succeed()
        test_same_source_and_destination()
        test_unreachable_after_failures()
        test_invalid_node_names()
        test_matches_networkx_dijkstra()
        test_performance_under_load()
        print("\n✅ All Dijkstra stress tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)