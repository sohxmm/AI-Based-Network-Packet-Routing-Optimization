"""train_rl.py â€“ Train a PPO agent on NetworkRoutingEnv.

Usage (from the backend/ directory):
    python -m ml.train_rl

What this script does:
1. Validates the environment with Stable-Baselines3 check_env().
2. Wraps the environment in a Monitor for episode statistics.
3. Initialises a PPO agent with an MlpPolicy on GPU (falls back to CPU).
4. Trains for 500,000 timesteps with periodic checkpoints every 50k steps.
5. Evaluates the final policy over 10 episodes.
6. Saves the final model to backend/ml/models/rl_router_final.
7. Logs training metrics to TensorBoard under runs/ppo_routing/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure backend root is importable when run as `python -m ml.train_rl`
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ml.rl_environment import NetworkRoutingEnv
from simulator.network_sim import NetworkSimulator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
_MODEL_DIR = _ROOT / "models"
_LOG_DIR = _ROOT.parent.parent / "runs" / "ppo_routing"
_FINAL_MODEL_PATH = _MODEL_DIR / "rl_router_final"
_CHECKPOINT_PREFIX = str(_MODEL_DIR / "ppo_checkpoint")


def main() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Environment validation                                           #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("PHASE 4 â€” PPO Routing Agent Training")
    print("=" * 60)
    print("\n[1/5] Validating environment with check_env()...")
    validation_env = NetworkRoutingEnv(seed=0)
    check_env(validation_env, warn=True)
    validation_env.close()
    print("      check_env() passed âœ“")

    # ------------------------------------------------------------------ #
    # 2. Training environment                                             #
    # ------------------------------------------------------------------ #
    print("\n[2/5] Creating training & evaluation environments...")
    train_env = Monitor(NetworkRoutingEnv(seed=42))
    eval_env = Monitor(NetworkRoutingEnv(seed=99))

    # ------------------------------------------------------------------ #
    # 3. Determine device                                                 #
    # ------------------------------------------------------------------ #
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"      Compute device: {device.upper()}")

    # ------------------------------------------------------------------ #
    # 4. PPO agent                                                        #
    # ------------------------------------------------------------------ #
    print("\n[3/5] Initialising PPO agent...")
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(_MODEL_DIR),
        name_prefix="ppo_checkpoint",
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(_MODEL_DIR),
        log_path=str(_LOG_DIR),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=1,
    )

    # ------------------------------------------------------------------ #
    # 5. Training                                                         #
    # ------------------------------------------------------------------ #
    total_timesteps = 500_000
    print(f"\n[4/5] Training PPO for {total_timesteps:,} timesteps...")
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )
    elapsed = time.time() - t0
    print(f"      Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min) âœ“")

    # ------------------------------------------------------------------ #
    # 6. Save final model                                                 #
    # ------------------------------------------------------------------ #
    model.save(str(_FINAL_MODEL_PATH))
    print(f"\n[5/5] Final model saved â†’ {_FINAL_MODEL_PATH}.zip âœ“")

    # ------------------------------------------------------------------ #
    # 7. Quick evaluation                                                 #
    # ------------------------------------------------------------------ #
    print("\n--- Post-training Evaluation (10 episodes) ---")
    obs, _ = eval_env.reset()
    episode_rewards: list[float] = []
    ep_reward = 0.0
    episodes_done = 0
    max_episodes = 10

    while episodes_done < max_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ep_reward += float(reward)
        if terminated or truncated:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            episodes_done += 1
            obs, _ = eval_env.reset()

    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))
    print(f"  Mean episode reward : {mean_r:.4f} Â± {std_r:.4f}")
    print(f"  Min / Max           : {min(episode_rewards):.4f} / {max(episode_rewards):.4f}")
    print("\nTensorBoard logs â†’", _LOG_DIR)
    print("Run: tensorboard --logdir", _LOG_DIR)

    train_env.close()
    eval_env.close()
    print("\nâœ… Phase 4 â€“ RL training complete.")


if __name__ == "__main__":
    main()

