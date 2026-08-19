"""Train the single-agent PPO routing policy.

Run from the repository root::

    python -m ml.training.train_rl

Fixes applied here, on top of the environment repairs in
``ml/environments/routing_env.py``:

* **The save path comes from the model registry.** Training used to write
  ``rl_router_final`` while the router loaded ``ppo_routing_agent.zip``. Nothing
  connected the two, and the mismatch went unnoticed for the life of the project
  because the loader swallowed the resulting ``FileNotFoundError`` in silence.
* **Seeded.** ``PPO(...)`` was constructed without a seed, so no run was
  reproducible.
* **TensorBoard claims match reality.** The old script printed "TensorBoard logs
  -> ..." and told the user to run ``tensorboard --logdir``, but never passed
  ``tensorboard_log=``, so there were no event files to read. It is now wired up
  when ``tensorboard`` is installed, and the script says so explicitly when it
  is not, rather than pointing at a directory that will stay empty.
* **Baselines reported.** The run ends by evaluating random, Dijkstra-equivalent
  and oracle policies on the same seeded episodes and printing the normalized
  score, because a raw episode return conveys nothing on its own.
* **Observation layout stored in the checkpoint**, so the router can detect a
  stale or mismatched policy instead of reshaping into nonsense.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ml.environments.routing_env import NetworkRoutingEnv
from ml.evaluation.baselines import format_table, run_full_evaluation
from ml.model_registry import RESULTS_DIR, path_for

logger = logging.getLogger("train_rl")

LOG_DIR = Path(__file__).resolve().parents[2] / "experiments" / "runs" / "ppo_routing"


def _tensorboard_dir() -> str | None:
    """Return the log directory only if tensorboard can actually consume it.

    Passing tensorboard_log= without the package installed makes SB3 raise, and
    passing it while claiming logs exist when they do not is exactly the defect
    we are trying to avoid. Neither is acceptable, so we check.
    """
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return None
    return str(LOG_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PPO routing policy.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--num-nodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=15_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    # Each episode builds a fresh simulator, which logs its topology at INFO.
    # That is useful once and noise 10,000 times.
    logging.getLogger("core.simulator").setLevel(logging.WARNING)

    model_dir = path_for("rl").parent
    model_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[1/5] Validating the environment...")
    probe = NetworkRoutingEnv(num_nodes=args.num_nodes, seed=0)
    check_env(probe, warn=True)
    n_links, n_nodes = probe.n_links, probe.n_nodes
    obs_dim = probe.observation_space.shape[0]
    probe.close()
    logger.info(
        "      check_env passed. obs_dim=%d (%d links, %d nodes)", obs_dim, n_links, n_nodes
    )

    logger.info("[2/5] Building training and evaluation environments...")
    train_env = Monitor(NetworkRoutingEnv(num_nodes=args.num_nodes, seed=args.seed))
    eval_env = Monitor(NetworkRoutingEnv(num_nodes=args.num_nodes, seed=args.seed + 57))

    tensorboard_dir = _tensorboard_dir()
    if tensorboard_dir:
        logger.info("      TensorBoard logs -> %s", tensorboard_dir)
    else:
        logger.info(
            "      tensorboard is not installed, so no event files will be "
            "written. Install it with: pip install tensorboard"
        )

    logger.info("[3/5] Initialising PPO (seed=%d)...", args.seed)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        tensorboard_log=tensorboard_dir,
        device="cpu",
        verbose=0,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(10_000, args.timesteps // 6),
            save_path=str(model_dir / "checkpoints_rl"),
            name_prefix="ppo_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "checkpoints_rl"),
            log_path=str(LOG_DIR),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=0,
        ),
    ]

    logger.info("[4/5] Training for %s timesteps...", f"{args.timesteps:,}")
    started = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - started
    logger.info("      done in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # Record the observation layout alongside the weights so the serving router
    # can refuse a mismatched policy rather than silently misinterpreting it.
    model.custom_data = {"n_links": n_links, "n_nodes": n_nodes, "obs_dim": obs_dim}
    destination = path_for("rl").with_suffix("")
    model.save(str(destination))
    logger.info("[5/5] Saved policy to %s.zip", destination)

    logger.info("Evaluating against the random floor and the oracle ceiling...")
    report = run_full_evaluation(
        model=model,
        n_episodes=args.final_eval_episodes,
        seed=args.seed + 999,
        output=RESULTS_DIR / "rl_evaluation.json",
    )
    print()
    print(format_table(report))
    print()

    ppo_score = report["normalized_scores"].get("ppo")
    if ppo_score is not None:
        if ppo_score <= 0:
            logger.warning(
                "Normalized score %.3f: the policy is no better than random. "
                "That is the result and it must be reported as such.",
                ppo_score,
            )
        else:
            logger.info(
                "Normalized score %.3f — the policy closes %.0f%% of the gap "
                "between random and the greedy oracle.",
                ppo_score,
                100 * ppo_score,
            )

    report["training"] = {
        "timesteps": args.timesteps,
        "seed": args.seed,
        "train_seconds": round(elapsed, 1),
        "obs_dim": obs_dim,
        "num_nodes": args.num_nodes,
    }
    (RESULTS_DIR / "rl_evaluation.json").write_text(json.dumps(report, indent=2))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
