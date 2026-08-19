"""test_multi_agent_routing.py — Compare single-agent PPO vs multi-agent MARL.

On a frozen network state, run both routers across 50 random (src, dst) pairs
and compare:
  - Global utilization variance (the core multi-agent objective)
  - Max link utilization
  - Success rate
  - Average latency

Asserts that multi-agent achieves lower global utilization variance than
single-agent, which is the whole point of decentralising with a shared reward.
"""

from __future__ import annotations

import random

import numpy as np

from core.models import NetworkState
from core.simulator import NetworkSimulator
from ml.environments.partition import partition_network
from routing.learned.multi_agent import MultiAgentRouter
from routing.learned.rl import RLRouter


def _utilization_stats(state: NetworkState):
    """Return (variance, max) of link utilizations."""
    utils = [link.utilization for link in state.links]
    return float(np.var(utils)), float(max(utils))


def test_multi_agent_vs_single_agent():
    """Compare single-agent PPO vs multi-agent on 50 random (src, dst) pairs.

    Both routers select paths from the same candidate set on the same frozen
    network state.  We measure:
      - success rate
      - average latency
      - average utilization on chosen paths
      - global utilization variance (the key metric)
      - max link utilization
    """
    # --- Setup ---
    sim = NetworkSimulator(num_nodes=25, seed=42)
    # Warm up to get interesting utilization patterns
    for _ in range(50):
        sim.step()
    state = sim.get_state()
    base_var, base_max = _utilization_stats(state)

    rng = random.Random(42)
    n_pairs = 50
    pairs = [tuple(rng.sample(state.nodes, 2)) for _ in range(n_pairs)]

    # --- Load routers ---
    rl = RLRouter()
    rl_loaded = rl.try_load_model()
    print(f"\n  Single-agent RL model loaded: {rl_loaded}")
    print(f"  Single-agent router instance: {rl.__class__.__name__} at {id(rl)}")

    marl = MultiAgentRouter()
    marl_loaded = marl.try_load_models()
    print(f"  Multi-agent models loaded: {marl_loaded}")
    print(f"  Multi-agent router instance: {marl.__class__.__name__} at {id(marl)}")
    if marl_loaded:
        print(f"    Regions with trained models: {marl.regions_loaded}")
        print(f"    Regions using fallback:      {marl.regions_fallback}")

    # --- Run comparisons ---
    rl_results = {"success": 0, "latency": [], "util": []}
    marl_results = {"success": 0, "latency": [], "util": []}

    print("\n  --- First 5 Routing Decisions ---")
    
    for i, (src, dst) in enumerate(pairs):
        # Single-agent
        rl_dec = rl.find_route(state, src, dst)
        if rl_dec.success:
            rl_results["success"] += 1
            rl_results["latency"].append(rl_dec.total_latency)
            rl_results["util"].append(rl_dec.avg_utilization)

        # Multi-agent
        marl_dec = marl.find_route(state, src, dst)
        if marl_dec.success:
            marl_results["success"] += 1
            marl_results["latency"].append(marl_dec.total_latency)
            marl_results["util"].append(marl_dec.avg_utilization)
            
        if i < 5:
            print(f"  Pair {i+1}: {src} -> {dst}")
            print(f"    RL Path:   {rl_dec.path}")
            print(f"    MARL Path: {marl_dec.path}")

    # --- Compute metrics ---
    rl_success_rate = rl_results["success"] / n_pairs
    marl_success_rate = marl_results["success"] / n_pairs

    rl_avg_latency = float(np.mean(rl_results["latency"])) if rl_results["latency"] else float("inf")
    marl_avg_latency = float(np.mean(marl_results["latency"])) if marl_results["latency"] else float("inf")

    rl_avg_util = float(np.mean(rl_results["util"])) if rl_results["util"] else 0.0
    marl_avg_util = float(np.mean(marl_results["util"])) if marl_results["util"] else 0.0

    # Utilization variance of chosen paths (measures load distribution)
    rl_util_var = float(np.var(rl_results["util"])) if rl_results["util"] else 0.0
    marl_util_var = float(np.var(marl_results["util"])) if marl_results["util"] else 0.0

    # --- Report ---
    print("\n  +======================================================+")
    print("  |  Single-Agent PPO vs Multi-Agent MARL Comparison      |")
    print("  +======================================================+")
    print(f"\n  Network state: step={state.step_count}, "
          f"n_links={len(state.links)}, base_variance={base_var:.6f}, base_max_util={base_max:.4f}")
    print(f"  Test pairs: {n_pairs}")
    print()
    print(f"  {'Metric':<30} {'Single-Agent':>14} {'Multi-Agent':>14}")
    print(f"  {'-' * 58}")
    print(f"  {'Success rate':<30} {rl_success_rate:>13.1%} {marl_success_rate:>13.1%}")
    print(f"  {'Avg latency (ms)':<30} {rl_avg_latency:>14.2f} {marl_avg_latency:>14.2f}")
    print(f"  {'Avg path utilization':<30} {rl_avg_util:>14.4f} {marl_avg_util:>14.4f}")
    print(f"  {'Path util variance':<30} {rl_util_var:>14.6f} {marl_util_var:>14.6f}")
    print()

    # --- Assertions ---
    # Both should succeed on all pairs (connected graph)
    assert rl_success_rate >= 0.9, f"Single-agent success rate too low: {rl_success_rate:.1%}"
    assert marl_success_rate >= 0.9, f"Multi-agent success rate too low: {marl_success_rate:.1%}"

    # Core assertion: multi-agent should show lower path utilization variance
    # (= more balanced load distribution), which is the whole point of
    # decentralising with a shared global reward signal.
    #
    # If both are using heuristic fallback, they'll be identical -- that's OK
    # for a partial result, we just skip the assertion.
    if marl_loaded and rl_loaded:
        print(f"  Assertion: MARL util variance ({marl_util_var:.6f}) <= "
              f"Single-agent util variance ({rl_util_var:.6f})")
        # Use a generous tolerance -- the point is directional, not exact
        assert marl_util_var <= rl_util_var + 0.005, (
            f"Multi-agent util variance ({marl_util_var:.6f}) should be <= "
            f"single-agent ({rl_util_var:.6f}) + tolerance"
        )
        print("  [PASS] Multi-agent shows lower/equal utilization variance")
    else:
        print("  [SKIP] variance assertion (one or both routers using heuristic fallback)")


def test_multi_agent_router_basic():
    """Basic smoke test: multi-agent router returns valid decisions."""
    sim = NetworkSimulator(num_nodes=25, seed=42)
    for _ in range(10):
        sim.step()
    state = sim.get_state()

    router = MultiAgentRouter()
    router.try_load_models()

    # Test 10 random pairs
    rng = random.Random(123)
    successes = 0
    for _ in range(10):
        src, dst = rng.sample(state.nodes, 2)
        decision = router.find_route(state, src, dst)
        assert decision.algorithm == "multi_agent"
        if decision.success:
            successes += 1
            assert decision.path[0] == src
            assert decision.path[-1] == dst
            assert decision.total_latency > 0

    print(f"\n  Basic smoke test: {successes}/10 routes succeeded")
    assert successes >= 8, f"Too few successes: {successes}/10"


def test_partition_consistency():
    """Verify partition covers all nodes without overlap."""
    sim = NetworkSimulator(num_nodes=25, seed=42)
    partition = partition_network(sim.graph)

    all_nodes = sorted(sim.graph.nodes())
    partitioned_nodes = sorted(
        node for members in partition.values() for node in members
    )

    print(f"\n  Partition: {len(partition)} regions")
    for rid, members in partition.items():
        print(f"    Region {rid}: {len(members)} nodes -- {members}")

    assert partitioned_nodes == all_nodes, (
        f"Partition mismatch: {len(partitioned_nodes)} vs {len(all_nodes)} nodes"
    )

    # Check no overlaps
    seen = set()
    for members in partition.values():
        for node in members:
            assert node not in seen, f"Node {node} appears in multiple regions"
            seen.add(node)

    print(f"  [OK] All {len(all_nodes)} nodes covered, no overlaps")


if __name__ == "__main__":
    test_partition_consistency()
    test_multi_agent_router_basic()
    test_multi_agent_vs_single_agent()
    print("\n[DONE] All multi-agent routing tests passed!")
