"""Stress test the network simulator before Phase 2 wires it to FastAPI."""

import sys
from pathlib import Path
import time

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from simulator.network_sim import NetworkSimulator
from router.dijkstra import find_route as dijkstra_route
from router.bellman_ford import find_route as bellman_ford_route
from router.aco import AntColonyRouter
from router.rl_agent import RLRouter

def test_500_steps_no_crash() -> None:
    """
    Test that simulator runs 500 steps continuously without crashing.
    Simulates real usage: backend running continuously for hours.
    """
    print("[TEST] Running 500 continuous steps...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    
    start = time.time()
    for i in range(500):
        state = sim.step()
        
        # Every 50 steps, verify state consistency
        if i % 50 == 0:
            assert len(state.nodes) == 10, f"Unexpected node count at step {i}"
            assert len(state.links) > 0, f"No links at step {i}"
            assert all(0.0 <= link.utilization <= 1.0 for link in state.links), \
                f"Invalid utilization at step {i}"
    
    elapsed = time.time() - start
    print(f"  [OK] Completed 500 steps in {elapsed:.2f}s ({500/elapsed:.0f} steps/sec)")


def test_link_failure_recovery() -> None:
    """
    Test injecting and restoring link failures.
    Simulates network outages users might inject via dashboard.
    """
    print("[TEST] Testing link failure injection...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    
    # Run a few steps to get utilization data
    for _ in range(5):
        sim.step()
    
    state = sim.get_state()
    num_links_before = len(state.links)
    
    # Pick a random link and fail it
    link_to_fail = state.links[0]
    sim.inject_failure(link_to_fail.source, link_to_fail.target)
    
    state = sim.get_state()
    num_links_after = len(state.links)
    
    assert num_links_after == num_links_before - 1, "Link was not removed"
    print(f"  [OK] Link failed: {num_links_before} -> {num_links_after}")
    
    # Restore it
    sim.restore_link(link_to_fail.source, link_to_fail.target)
    
    state = sim.get_state()
    num_links_restored = len(state.links)
    
    assert num_links_restored == num_links_before, "Link was not restored"
    print(f"  [OK] Link restored: {num_links_after} -> {num_links_restored}")


def test_all_routers_under_load() -> None:
    """
    Call all 4 routers 100 times on the same state.
    Measures performance and verifies no crashes.
    """
    print("[TEST] Testing all routers under load (100 requests)...")
    sim = NetworkSimulator(num_nodes=10, seed=42)
    
    # Warm up a few steps
    for _ in range(10):
        sim.step()
    
    state = sim.get_state()
    
    aco = AntColonyRouter()
    rl = RLRouter()
    
    start = time.time()
    success_count = 0
    
    for i in range(100):
        # Pick random source and destination
        import random
        rng = random.Random(42 + i)
        src, dst = rng.sample(state.nodes, 2)
        
        # Call all 4 routers
        d1 = dijkstra_route(state, src, dst)
        d2 = bellman_ford_route(state, src, dst)
        d3 = aco.find_path(state, src, dst)
        d4 = rl.predict(state, src, dst)
        
        if all([d1.success, d2.success, d3.success, d4.success]):
            success_count += 1
    
    elapsed = time.time() - start
    success_rate = (success_count / 100) * 100
    
    print(f"  [OK] Completed 400 route lookups in {elapsed:.2f}s ({400/elapsed:.0f} lookups/sec)")
    print(f"  [OK] Success rate: {success_rate:.1f}%")
    assert success_rate >= 95, f"Success rate too low: {success_rate}%"


def test_reproducibility_with_seed() -> None:
    """
    Same seed should produce identical simulation outcomes.
    Critical for ML training: need deterministic data generation.
    """
    print("[TEST] Testing reproducibility with seed=42...")
    
    # Run 1
    sim1 = NetworkSimulator(num_nodes=10, seed=42)
    for _ in range(20):
        sim1.step()
    state1 = sim1.get_state()
    
    # Run 2 (should be identical)
    sim2 = NetworkSimulator(num_nodes=10, seed=42)
    for _ in range(20):
        sim2.step()
    state2 = sim2.get_state()
    
    # Compare link utilizations
    for link1, link2 in zip(state1.links, state2.links):
        assert link1.source == link2.source
        assert link1.target == link2.target
        assert abs(link1.utilization - link2.utilization) < 1e-10, \
            f"Utilization mismatch: {link1.utilization} vs {link2.utilization}"
    
    print(f"  [OK] Identical results across {len(state1.links)} links")


if __name__ == "__main__":
    try:
        test_500_steps_no_crash()
        test_link_failure_recovery()
        test_all_routers_under_load()
        test_reproducibility_with_seed()
        print("\n[SUCCESS] All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)