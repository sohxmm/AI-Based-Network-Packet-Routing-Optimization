"""train_multi_agent.py — Train one PPO model per region using naive self-play.

Usage (from the backend/ directory):
    python -m ml.train_multi_agent

SIMPLIFICATION NOTE (by design):
    True simultaneous multi-agent training is out of scope.  We use a naive
    self-play rotation instead: train region A's policy for K iterations
    with all other regions' policies frozen, then rotate to region B, etc.
    This is repeated for N rounds.  The inter-region coordination emerges
    from the shared global reward signal (utilization variance and max-link
    penalty computed across ALL links), not from simultaneous policy updates.

    This is an honest documented simplification — NOT full MARL.

Training schedule:
    N_ROUNDS = 3
    K_STEPS  = 50_000  per region per round
    Total    = N_ROUNDS × n_regions × K_STEPS  ≈  450k–600k total steps

Output:
    backend/ml/models/multi_agent_region_{i}.zip  (one per region)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ml.multi_agent_rl_environment import RegionalRoutingEnv
from ml.network_partition import partition_network
from simulator.network_sim import NetworkSimulator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
_MODEL_DIR = _ROOT / "models"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
N_ROUNDS = 3           # number of self-play rotation rounds
K_STEPS = 15_000       # timesteps per region per round
NUM_NODES = 25
SEED = 42


def main() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent RL -- Naive Self-Play Training")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Partition the network                                            #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Partitioning the 25-node topology...")
    sim = NetworkSimulator(num_nodes=NUM_NODES, seed=SEED)
    partition = partition_network(sim.graph)
    n_regions = len(partition)
    print(f"       -> {n_regions} regions")
    for rid, members in partition.items():
        print(f"         Region {rid}: {members}")

    # ------------------------------------------------------------------ #
    # 2. Create environments and validate                                 #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Creating regional environments...")
    envs: dict[int, Monitor] = {}
    for rid, members in partition.items():
        env = RegionalRoutingEnv(
            region_id=rid,
            region_nodes=members,
            num_nodes=NUM_NODES,
            seed=SEED,
        )
        # Validate only the first one to save time
        if rid == 0:
            print("       Validating RegionalRoutingEnv with check_env()...")
            check_env(env, warn=True)
            print("       check_env() passed [OK]")
        envs[rid] = Monitor(env)

    # ------------------------------------------------------------------ #
    # 3. Determine device                                                 #
    # ------------------------------------------------------------------ #
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"       Compute device: {device.upper()}")

    # ------------------------------------------------------------------ #
    # 4. Initialise PPO models (one per region)                           #
    # ------------------------------------------------------------------ #
    print("\n[3/4] Initialising PPO agents...")
    models: dict[int, PPO] = {}
    for rid in partition:
        models[rid] = PPO(
            policy="MlpPolicy",
            env=envs[rid],
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
            verbose=0,
        )
        print(f"       Region {rid}: PPO initialised (obs_dim={models[rid].observation_space.shape[0]})")

    # ------------------------------------------------------------------ #
    # 5. Naive self-play rotation training                                #
    # ------------------------------------------------------------------ #
    print(f"\n[4/4] Training with naive self-play rotation")
    print(f"       N_ROUNDS={N_ROUNDS}, K_STEPS={K_STEPS:,}, n_regions={n_regions}")
    print(f"       Total steps: {N_ROUNDS * n_regions * K_STEPS:,}")

    t0 = time.time()
    for round_idx in range(N_ROUNDS):
        print(f"\n  -- Round {round_idx + 1}/{N_ROUNDS} --")
        for rid in partition:
            t_start = time.time()
            print(f"    Training Region {rid} for {K_STEPS:,} steps...", end=" ", flush=True)
            models[rid].learn(
                total_timesteps=K_STEPS,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            dt = time.time() - t_start
            print(f"done ({dt:.1f}s)")

    total_time = time.time() - t0
    print(f"\n  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    # ------------------------------------------------------------------ #
    # 6. Save models                                                      #
    # ------------------------------------------------------------------ #
    print("\n  Saving models:")
    for rid in partition:
        model_path = _MODEL_DIR / f"multi_agent_region_{rid}"
        models[rid].save(str(model_path))
        print(f"    Region {rid} -> {model_path}.zip")

    # ------------------------------------------------------------------ #
    # 7. Quick evaluation                                                 #
    # ------------------------------------------------------------------ #
    print("\n  --- Post-training Evaluation (5 episodes per region) ---")
    for rid in partition:
        obs, _ = envs[rid].reset()
        ep_rewards: list[float] = []
        ep_reward = 0.0
        episodes_done = 0

        while episodes_done < 5:
            action, _ = models[rid].predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = envs[rid].step(action)
            ep_reward += float(reward)
            if terminated or truncated:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                episodes_done += 1
                obs, _ = envs[rid].reset()

        mean_r = float(np.mean(ep_rewards))
        print(f"    Region {rid}: mean_reward={mean_r:.4f} (over 5 episodes)")

    # ------------------------------------------------------------------ #
    # 8. Cleanup                                                          #
    # ------------------------------------------------------------------ #
    for env in envs.values():
        env.close()

    print("\n[DONE] Multi-Agent RL training complete.")


if __name__ == "__main__":
    main()
