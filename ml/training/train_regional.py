"""Train the decentralized regional routing policies (CTDE).

Run from the repository root::

    python -m ml.training.train_regional

Centralized training, decentralized execution
---------------------------------------------
Stable-Baselines3 has no asymmetric actor-critic, so the observation carries
both blocks — ``[local | global_summary]`` — and the two feature extractors are
built with different views of it:

* the **actor** extractor slices off the global block, so the policy is a
  function of local information only;
* the **critic** extractor consumes the whole vector, so the value function can
  use the network-wide summary the policy never sees.

That asymmetry is the actual CTDE claim, and it is not taken on faith:
``tests/unit/ml/test_marl_locality.py`` perturbs the global block and asserts
the action distribution does not move, while the value estimate does.

What is *not* claimed: there is no explicit inter-agent messaging and no shared
critic across agents. These are independent learners trained in rotation against
frozen partners, cooperating through the shared environment and a shared team
reward term. That is a real MARL setting, and the model card says so plainly
rather than describing it as something stronger.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from core.simulator import NetworkSimulator
from ml.environments.partition import partition_network
from ml.environments.regional_env import RegionalRoutingEnv
from ml.local_features import GLOBAL_DIM, LOCAL_DIM
from ml.model_registry import RESULTS_DIR, regional_path

logger = logging.getLogger("train_regional")


class AsymmetricExtractor(BaseFeaturesExtractor):
    """Feature extractor that can be restricted to the local slice.

    ``use_global=False`` (the actor) reads only the first ``LOCAL_DIM`` entries.
    ``use_global=True`` (the critic) reads the whole observation.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        local_dim: int = LOCAL_DIM,
        use_global: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.local_dim = local_dim
        self.use_global = use_global
        input_dim = int(observation_space.shape[0]) if use_global else local_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        view = observations if self.use_global else observations[:, : self.local_dim]
        return self.net(view)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """A policy whose actor is local-only and whose critic sees everything.

    The asymmetry lives in the *class*, not in a patch applied after
    construction. That distinction is not cosmetic: ``PPO.load`` rebuilds the
    policy by calling this constructor with the saved ``policy_kwargs``, and SB3
    hands both feature extractors the same kwargs. A post-construction patch is
    therefore silently lost on reload and the checkpoint fails to restore with a
    shape mismatch — which is exactly what happened on the first attempt,
    leaving the multi-agent router on its heuristic fallback for an entire
    benchmark run while reporting ``fallback_rate = 1.00``.
    """

    def __init__(self, *args, local_dim: int = LOCAL_DIM, **kwargs):
        self.local_dim = local_dim
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = AsymmetricExtractor
        extractor_kwargs = dict(kwargs.get("features_extractor_kwargs") or {})
        extractor_kwargs.update({"local_dim": local_dim, "use_global": False})
        kwargs["features_extractor_kwargs"] = extractor_kwargs

        super().__init__(*args, **kwargs)

        # The base class built the critic's extractor from the actor's kwargs.
        # Swap in the global-aware variant, then rebuild the optimizer so it
        # owns the new parameters.
        self.vf_features_extractor = AsymmetricExtractor(
            self.observation_space,
            features_dim=self.features_extractor.features_dim,
            local_dim=local_dim,
            use_global=True,
        ).to(self.device)

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=current_lr, **self.optimizer_kwargs
        )


def build_ctde_ppo(env, seed: int, learning_rate: float = 3e-4) -> PPO:
    """Build a PPO whose critic sees the global summary and whose actor does not."""
    return PPO(
        policy=AsymmetricActorCriticPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        device="cpu",
        verbose=0,
    )


def evaluate_region(model: PPO, env: RegionalRoutingEnv, episodes: int, seed: int) -> float:
    """Mean episode return for one region's policy."""
    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return float(np.mean(returns)) if returns else 0.0


def evaluate_random(env: RegionalRoutingEnv, episodes: int, seed: int) -> float:
    """The floor for one region: uniform next-hop choice."""
    returns: list[float] = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return float(np.mean(returns)) if returns else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regional CTDE policies.")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timesteps-per-round", type=int, default=30_000)
    parser.add_argument("--num-nodes", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    # Each episode builds a fresh simulator, which logs its topology at INFO.
    # That is useful once and noise 10,000 times.
    logging.getLogger("core.simulator").setLevel(logging.WARNING)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    reference = NetworkSimulator(num_nodes=args.num_nodes, seed=args.seed)
    partition = partition_network(reference.graph)
    region_ids = sorted(partition)
    logger.info(
        "Partitioned %d nodes into %d regions: %s",
        args.num_nodes,
        len(region_ids),
        {rid: len(partition[rid]) for rid in region_ids},
    )
    logger.info(
        "Observation: %d local + %d global = %d (independent of network size)",
        LOCAL_DIM,
        GLOBAL_DIM,
        LOCAL_DIM + GLOBAL_DIM,
    )

    models: dict[int, PPO] = {}
    started = time.time()

    # Rotation: train one region at a time against the current partners, so
    # each agent adapts to the behaviour the others actually exhibit.
    for round_index in range(1, args.rounds + 1):
        for region_id in region_ids:
            partners = {rid: m for rid, m in models.items() if rid != region_id}
            env = Monitor(
                RegionalRoutingEnv(
                    region_id=region_id,
                    partition=partition,
                    num_nodes=args.num_nodes,
                    seed=args.seed,
                    partner_policies=partners,
                )
            )

            if region_id in models:
                model = models[region_id]
                model.set_env(env)
            else:
                model = build_ctde_ppo(env, seed=args.seed + region_id)
                models[region_id] = model

            model.learn(
                total_timesteps=args.timesteps_per_round,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            logger.info(
                "round %d/%d  region %d trained (+%s steps)",
                round_index,
                args.rounds,
                region_id,
                f"{args.timesteps_per_round:,}",
            )
            env.close()

    elapsed = time.time() - started
    logger.info("Training finished in %.1fs (%.1f min)", elapsed, elapsed / 60)

    scores: dict[str, dict[str, float]] = {}
    for region_id, model in models.items():
        eval_env = RegionalRoutingEnv(
            region_id=region_id,
            partition=partition,
            num_nodes=args.num_nodes,
            seed=args.seed + 500,
            partner_policies={rid: m for rid, m in models.items() if rid != region_id},
        )
        trained = evaluate_region(model, eval_env, args.eval_episodes, seed=args.seed + 700)
        chance = evaluate_random(eval_env, args.eval_episodes, seed=args.seed + 700)
        scores[str(region_id)] = {
            "trained_mean_return": trained,
            "random_mean_return": chance,
            "improvement": trained - chance,
            "beats_random": bool(trained > chance),
        }
        logger.info(
            "region %d: trained=%.2f  random=%.2f  %s",
            region_id,
            trained,
            chance,
            "better" if trained > chance else "NOT better than random",
        )
        eval_env.close()

        destination = regional_path(region_id).with_suffix("")
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(destination))
        logger.info("Saved region %d policy to %s.zip", region_id, destination)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "regional_ctde_policies",
        "architecture": "decentralized execution, centralized critic, independent learners",
        "seed": args.seed,
        "num_nodes": args.num_nodes,
        "regions": {str(rid): len(partition[rid]) for rid in region_ids},
        "local_obs_dim": LOCAL_DIM,
        "global_obs_dim": GLOBAL_DIM,
        "rounds": args.rounds,
        "timesteps_per_region": args.rounds * args.timesteps_per_round,
        "per_region": scores,
        "regions_beating_random": sum(1 for s in scores.values() if s["beats_random"]),
        "train_seconds": round(elapsed, 1),
    }
    (RESULTS_DIR / "marl_evaluation.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", RESULTS_DIR / "marl_evaluation.json")


if __name__ == "__main__":
    main()
