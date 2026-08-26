"""Floors and ceilings, without which a reported score means nothing.

The problem with the previous ML work was not that the PPO agent failed.
Negative results are fine. It was that the reported number — "mean reward
improved from -77 to -61" — was uninterpretable, because nobody had measured
what a *random* policy scores or what an *oracle* scores on the same episodes.
Without those two numbers, -61 conveys no information at all.

The statistic this module produces is the **normalized score**::

    normalized = (policy_mean - random_mean) / (oracle_mean - random_mean)

    0.0  no better than picking at random
    1.0  matches the greedy oracle
    <0   actively worse than random

Reporting that instead of a raw return is the single change most likely to
convince a reader that the evaluation was done properly — and it is honest in
both directions, because it makes a bad result equally legible.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ml.environments.routing_env import NetworkRoutingEnv

logger = logging.getLogger(__name__)


@dataclass
class PolicyResult:
    """Episode-return statistics for one policy."""

    name: str
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    n_episodes: int

    @classmethod
    def from_returns(cls, name: str, returns: list[float]) -> PolicyResult:
        array = np.asarray(returns, dtype=float)
        n = max(1, len(array))
        mean = float(array.mean()) if len(array) else 0.0
        std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        return cls(name, mean, std, float(mean - half), float(mean + half), int(n))


def evaluate_policy(
    action_fn: Callable[[np.ndarray, NetworkRoutingEnv], int],
    name: str,
    n_episodes: int = 20,
    seed: int = 99,
    num_nodes: int = 25,
) -> PolicyResult:
    """Run *n_episodes* of the routing environment under *action_fn*.

    Every policy is evaluated on the *same* seeded episodes, so the comparison
    is paired and differences cannot be an artifact of episode difficulty.
    """
    env = NetworkRoutingEnv(num_nodes=num_nodes, seed=seed)
    returns: list[float] = []

    for episode in range(n_episodes):
        observation, _ = env.reset(seed=seed + episode)
        total = 0.0
        while True:
            action = action_fn(observation, env)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)

    env.close()
    return PolicyResult.from_returns(name, returns)


def random_policy(observation: np.ndarray, env: NetworkRoutingEnv) -> int:
    """Uniform choice over the action space — the floor."""
    return int(env.action_space.sample())


def oracle_policy(observation: np.ndarray, env: NetworkRoutingEnv) -> int:
    """Greedily maximise the immediate reward — the ceiling.

    Greedy, not optimal: it ignores how its own load affects later steps, so in
    a closed-loop network a policy that plans ahead can in principle exceed it.
    A normalized score above 1.0 is therefore meaningful rather than a bug.
    """
    return env.oracle_action()


def first_candidate_policy(observation: np.ndarray, env: NetworkRoutingEnv) -> int:
    """Always take the cheapest candidate — behaviourally equivalent to Dijkstra."""
    return 0


def normalized_score(policy: PolicyResult, floor: PolicyResult, ceiling: PolicyResult) -> float:
    """Where *policy* sits between the random floor and the oracle ceiling."""
    span = ceiling.mean - floor.mean
    if abs(span) < 1e-9:
        return 0.0
    return float((policy.mean - floor.mean) / span)


def run_full_evaluation(
    model=None,
    n_episodes: int = 20,
    seed: int = 99,
    output: Path | None = None,
) -> dict:
    """Evaluate random, Dijkstra-equivalent, oracle and (optionally) PPO."""
    results: dict[str, PolicyResult] = {
        "random": evaluate_policy(random_policy, "random", n_episodes, seed),
        "greedy_first_candidate": evaluate_policy(
            first_candidate_policy, "greedy_first_candidate", n_episodes, seed
        ),
        "oracle": evaluate_policy(oracle_policy, "oracle", n_episodes, seed),
    }

    if model is not None:
        def ppo_policy(observation: np.ndarray, env: NetworkRoutingEnv) -> int:
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

        results["ppo"] = evaluate_policy(ppo_policy, "ppo", n_episodes, seed)

    floor, ceiling = results["random"], results["oracle"]
    report = {
        "n_episodes": n_episodes,
        "seed": seed,
        "policies": {name: asdict(result) for name, result in results.items()},
        "normalized_scores": {
            name: normalized_score(result, floor, ceiling)
            for name, result in results.items()
        },
        "interpretation": (
            "normalized = (policy - random) / (oracle - random). 0.0 means no "
            "better than random; 1.0 means it matches the greedy oracle."
        ),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        logger.info("Wrote %s", output)

    return report


def format_table(report: dict) -> str:
    """Render the evaluation as a fixed-width table for the console and the docs."""
    lines = [
        f"{'policy':<24}{'mean':>10}{'std':>9}{'95% CI':>22}{'normalized':>12}",
        "-" * 77,
    ]
    for name, stats in report["policies"].items():
        normalized = report["normalized_scores"][name]
        interval = f"[{stats['ci95_low']:.2f}, {stats['ci95_high']:.2f}]"
        lines.append(
            f"{name:<24}{stats['mean']:>10.2f}{stats['std']:>9.2f}"
            f"{interval:>22}{normalized:>12.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "PolicyResult",
    "evaluate_policy",
    "first_candidate_policy",
    "format_table",
    "normalized_score",
    "oracle_policy",
    "random_policy",
    "run_full_evaluation",
]
