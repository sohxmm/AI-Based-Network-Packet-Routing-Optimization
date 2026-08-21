"""The tests that make the multi-agent claim true rather than asserted.

The previous implementation gave every agent the full global link vector and had
each one emit a complete end-to-end path, with the acting agent chosen by a
lookup on the source node. That is a mixture-of-experts with a hardcoded gating
function. Calling it "centralized-critic, decentralized-execution multi-agent
RL" was the claim least likely to survive a viva.

Two properties have to hold for the current claim to be honest, and both are
checked here rather than described in a docstring:

1. **Decentralized execution** — an agent's observation must not grow with the
   size of the network, because if it can see everything it is not local.
2. **Centralized training** — the critic must use information the actor cannot,
   or "centralized critic" means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.simulator import NetworkSimulator
from ml.environments.partition import partition_network
from ml.environments.regional_env import RegionalRoutingEnv
from ml.local_features import GLOBAL_DIM, LOCAL_DIM, OBS_DIM


@pytest.mark.parametrize("num_nodes", [25, 50, 100])
def test_observation_width_is_independent_of_network_size(num_nodes):
    """The defining property of decentralized execution."""
    sim = NetworkSimulator(num_nodes=num_nodes, seed=42)
    env = RegionalRoutingEnv(
        region_id=0, partition=partition_network(sim.graph), num_nodes=num_nodes, seed=42
    )
    assert env.observation_space.shape == (OBS_DIM,), (
        f"A {num_nodes}-node network produced a {env.observation_space.shape} "
        f"observation. If the width tracks the network, the agent is not local."
    )


def test_action_is_a_single_next_hop_not_a_whole_path():
    """An agent that emits an end-to-end path is not decentralized."""
    sim = NetworkSimulator(num_nodes=25, seed=42)
    env = RegionalRoutingEnv(region_id=0, partition=partition_network(sim.graph), seed=42)
    from ml.local_features import MAX_DEGREE

    assert env.action_space.n == MAX_DEGREE, (
        "The action space must index the current node's neighbours, not a set "
        "of candidate end-to-end paths."
    )


def test_routes_never_revisit_a_node():
    """Hop-by-hop forwarding must not produce loops."""
    sim = NetworkSimulator(num_nodes=25, seed=42)
    env = RegionalRoutingEnv(region_id=0, partition=partition_network(sim.graph), seed=42)
    env.reset(seed=5)

    for _ in range(100):
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        path = env._path  # noqa: SLF001
        assert len(set(path)) == len(path), f"loop in {path}"
        if truncated:
            break


def test_critic_sees_the_global_summary_and_the_actor_does_not():
    """The actual centralized-training / decentralized-execution claim.

    Perturbing only the global block must leave the policy's action distribution
    untouched while moving the value estimate. If the actor moves, execution is
    not decentralized; if the critic does not, training is not centralized.
    """
    from stable_baselines3.common.monitor import Monitor

    from ml.training.train_regional import build_ctde_ppo

    sim = NetworkSimulator(num_nodes=25, seed=42)
    env = Monitor(
        RegionalRoutingEnv(region_id=0, partition=partition_network(sim.graph), seed=42)
    )
    model = build_ctde_ppo(env, seed=1)

    torch.manual_seed(0)
    observation = torch.zeros(1, OBS_DIM)
    observation[0, :LOCAL_DIM] = torch.rand(LOCAL_DIM)

    perturbed = observation.clone()
    perturbed[0, LOCAL_DIM:] = torch.rand(GLOBAL_DIM)

    policy = model.policy
    with torch.no_grad():
        probs_a = policy.get_distribution(observation).distribution.probs
        probs_b = policy.get_distribution(perturbed).distribution.probs
        value_a = policy.predict_values(observation)
        value_b = policy.predict_values(perturbed)

    assert torch.allclose(probs_a, probs_b), (
        "The actor's output changed when only the global block moved, so the "
        "policy is reading information it must not have at execution time."
    )
    assert not torch.allclose(value_a, value_b), (
        "The critic ignored the global block, so there is no centralized "
        "training happening and the CTDE claim would be false."
    )
    env.close()


def test_partition_follows_the_live_topology():
    """The router used to derive regions from a hardcoded 25-node simulator.

    On the 100-node scenario every node above R25 mapped to region -1 and forced
    a heuristic fallback; we measured fallback_rate = 0.75 there.
    """
    from routing.learned.multi_agent import MultiAgentRouter

    router = MultiAgentRouter(seed=42)

    small = NetworkSimulator(num_nodes=25, seed=42).get_state()
    router.ensure_partition(small)
    assert set(router._region_of) == set(small.nodes)  # noqa: SLF001

    large = NetworkSimulator(num_nodes=100, seed=42).get_state()
    router.ensure_partition(large)
    assigned = set(router._region_of)  # noqa: SLF001
    assert assigned == set(large.nodes), (
        f"{len(set(large.nodes) - assigned)} nodes have no region on the "
        f"100-node topology; those would all force a fallback."
    )


def test_global_summary_is_fixed_width_across_topologies():
    """A raw link vector could not transfer between topologies; a summary can."""
    from ml.local_features import build_global_summary

    for num_nodes in (25, 100):
        state = NetworkSimulator(num_nodes=num_nodes, seed=42).get_state()
        summary = build_global_summary(state)
        assert summary.shape == (GLOBAL_DIM,)
        assert np.all(np.isfinite(summary))
