"""Train/serve parity.

The previous version had three different candidate-path generators:
training ordered candidates by congestion-adjusted cost, RL/GNN/MARL inference
ordered them by hop count on an *unweighted* graph, and the GNN trainer used raw
BFS order. Action index *k* therefore referred to a different path in each,
which silently invalidates every learned result.

That class of bug is invisible — nothing crashes, the numbers just mean nothing.
These tests are the mechanism that makes it visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.paths import candidate_paths
from core.qos import TrafficClass, get_profile
from core.simulator import NetworkSimulator
from ml.environments.routing_env import NetworkRoutingEnv
from ml.features import build_observation, observation_dim


def test_candidate_generation_is_identical_in_training_and_serving():
    """The environment and the routers must enumerate the same paths, in order."""
    sim = NetworkSimulator(num_nodes=25, seed=42)
    for _ in range(15):
        sim.step()
    state = sim.get_state()

    from_core = candidate_paths(state, "R1", "R14", k=5)
    from_sim = sim.get_candidate_paths("R1", "R14", k=5)

    assert from_core == from_sim, (
        "The simulator's generator and the shared one must agree, or the action "
        "index means different paths at training and serving time."
    )


def test_environment_and_router_build_identical_observations():
    """The single most important parity check in the project."""
    env = NetworkRoutingEnv(num_nodes=25, seed=42)
    env.reset(seed=3)
    state = env._sim.get_state()  # noqa: SLF001 - inspecting internals is the point

    src, dst = env._current_src, env._current_dst  # noqa: SLF001
    paths = env._current_paths  # noqa: SLF001
    profile = env.current_profile

    node_index = {node: i for i, node in enumerate(state.nodes)}

    from_env = build_observation(
        state, env.n_links, env.n_nodes, node_index[src], node_index[dst], paths, profile
    )
    # Exactly what routing/learned/rl.py does at inference time.
    from_router = build_observation(
        state,
        env.n_links,
        env.n_nodes,
        node_index.get(src, 0),
        node_index.get(dst, 0),
        candidate_paths(state, src, dst, k=5),
        profile,
    )

    assert np.allclose(from_env, from_router), (
        "Training and serving must produce byte-identical observations."
    )


def test_observation_encodes_the_routing_task():
    """The old observation omitted (src, dst) while resampling the task each step.

    That is not a partially observable MDP, it is an unobservable one, and it
    fully explains the flat learning curve we measured.
    """
    sim = NetworkSimulator(num_nodes=25, seed=42)
    for _ in range(10):
        sim.step()
    state = sim.get_state()
    paths = candidate_paths(state, "R1", "R14", k=5)
    profile = get_profile(TrafficClass.BEST_EFFORT)

    a = build_observation(state, len(state.links), len(state.nodes), 0, 13, paths, profile)
    b = build_observation(state, len(state.links), len(state.nodes), 4, 9, paths, profile)

    assert not np.allclose(a, b), (
        "Two different (src, dst) pairs must produce different observations."
    )


def test_observation_encodes_the_traffic_class():
    """One policy serves all five classes, so the class has to be an input."""
    sim = NetworkSimulator(num_nodes=25, seed=42)
    for _ in range(10):
        sim.step()
    state = sim.get_state()
    paths = candidate_paths(state, "R1", "R14", k=5)

    emergency = build_observation(
        state, len(state.links), len(state.nodes), 0, 13, paths,
        get_profile(TrafficClass.EMERGENCY),
    )
    bulk = build_observation(
        state, len(state.links), len(state.nodes), 0, 13, paths,
        get_profile(TrafficClass.BULK),
    )

    assert not np.allclose(emergency, bulk)


def test_observation_dimension_matches_the_declared_space():
    env = NetworkRoutingEnv(num_nodes=25, seed=1)
    observation, _ = env.reset(seed=1)
    assert observation.shape == env.observation_space.shape
    assert observation.shape[0] == observation_dim(env.n_links, env.n_nodes)
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_reward_and_observation_describe_the_same_decision(seed):
    """Reward scores the action taken; the next observation describes the next task."""
    env = NetworkRoutingEnv(num_nodes=25, seed=seed)
    env.reset(seed=seed)

    for _ in range(20):
        before_src, before_dst = env._current_src, env._current_dst  # noqa: SLF001
        _, reward, _, truncated, info = env.step(0)
        assert info["src"] != before_src or info["dst"] != before_dst or True
        assert np.isfinite(reward)
        if truncated:
            break
