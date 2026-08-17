"""Proof that the simulation loop is actually closed.

This is the most important behavioural test in the repository. Link
utilization used to evolve as a random walk independent of routing, which
made per-path latency minimisation exactly optimal — Dijkstra
solves that exactly, so no learned policy could ever beat it. Every ML weakness
in the project traced back to this one property.

If these tests ever fail, the project's central premise has become untestable
again, and every benchmark number would need to be regenerated.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.simulator import NetworkSimulator


def _utilization_of(state, a: str, b: str) -> float:
    key = frozenset((a, b))
    return next(
        link.utilization
        for link in state.links
        if frozenset((link.source, link.target)) == key
    )


def test_routing_increases_utilization_on_the_chosen_path(quiet_sim):
    """A link carrying flows must get busier than it started."""
    path = quiet_sim.get_candidate_paths("R1", "R10", k=1)[0]
    before = _utilization_of(quiet_sim.get_state(), path[0], path[1])

    for _ in range(20):
        quiet_sim.register_flow(path)
        quiet_sim.step()

    after = _utilization_of(quiet_sim.get_state(), path[0], path[1])
    assert after > before, (
        "register_flow must raise utilization on the chosen path; without this "
        "the simulator is open-loop and Dijkstra is optimal by construction."
    )


def test_untouched_links_are_not_driven_by_our_routing(quiet_sim):
    """The effect has to be *local*, or it is not feedback, it is drift."""
    path = quiet_sim.get_candidate_paths("R1", "R10", k=1)[0]
    on_path = {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}

    for _ in range(25):
        quiet_sim.register_flow(path)
        quiet_sim.step()

    state = quiet_sim.get_state()
    on = [
        link.utilization
        for link in state.links
        if frozenset((link.source, link.target)) in on_path
    ]
    off = [
        link.utilization
        for link in state.links
        if frozenset((link.source, link.target)) not in on_path
    ]
    assert np.mean(on) > np.mean(off) + 0.1, (
        "Loaded links should be measurably busier than untouched ones."
    )


def test_load_balancing_beats_greedy_under_repeated_demand():
    """Spreading load must keep the worst link cooler than saturating one path.

    This is the property the whole project is about. Under a *non-reactive*
    network it is unmeasurable, because nothing anyone routes changes anything.
    """
    greedy = NetworkSimulator(num_nodes=25, seed=7, background_flows=0)
    spread = NetworkSimulator(num_nodes=25, seed=7, background_flows=0)

    greedy_paths = greedy.get_candidate_paths("R1", "R12", k=3)
    spread_paths = spread.get_candidate_paths("R1", "R12", k=3)
    assert len(greedy_paths) >= 2, "need alternatives for this test to mean anything"

    for step in range(60):
        # Greedy: always the single cheapest path.
        greedy.register_flow(greedy_paths[0])
        greedy.step()
        # Round-robin: alternate across the candidate set.
        spread.register_flow(spread_paths[step % len(spread_paths)])
        spread.step()

    greedy_max = max(link.utilization for link in greedy.get_state().links)
    spread_max = max(link.utilization for link in spread.get_state().links)

    assert spread_max < greedy_max, (
        f"Round-robin peak {spread_max:.3f} should be below greedy peak "
        f"{greedy_max:.3f}. If it is not, the closed loop is too weak for "
        f"load balancing to be demonstrable."
    )


@pytest.mark.parametrize("num_nodes", [25, 50, 100])
def test_topology_is_not_a_ring(num_nodes):
    """The 100-node topology used to be a degree-2 ring with diameter 50.

    Two simple paths between any pair means no algorithm can differentiate, and
    every one of them scored the same. It was a scale test mislabelled as a
    stress test.
    """
    stats = NetworkSimulator(num_nodes=num_nodes, seed=42).topology_stats()
    assert stats["is_connected"]
    assert stats["avg_degree"] >= 3.0, f"degree {stats['avg_degree']} is ring-like"
    assert stats["diameter"] is not None and stats["diameter"] < 12


def test_utilization_has_learnable_temporal_structure():
    """Utilization must not be a pure random walk.

    The Bayes-optimal one-step predictor for a random walk is the identity, so
    training a sequence model on one is unfalsifiable busywork. An AR(1) process
    around a diurnal baseline gives a forecaster something real to model.
    """
    sim = NetworkSimulator(num_nodes=25, seed=11, background_flows=0)
    series = np.array([[link.utilization for link in sim.step().links] for _ in range(400)])

    # Autocorrelation at the diurnal period should exceed autocorrelation at a
    # half-period offset, which a random walk would not produce.
    period = sim.diurnal_period
    column = series[:, 0] - series[:, 0].mean()
    at_period = float(np.corrcoef(column[:-period], column[period:])[0, 1])
    at_half = float(np.corrcoef(column[: -period // 2], column[period // 2 :])[0, 1])

    assert at_period > at_half, (
        f"Expected structure at the {period}-step cycle "
        f"(r={at_period:.3f}) to exceed the half-cycle (r={at_half:.3f})."
    )
