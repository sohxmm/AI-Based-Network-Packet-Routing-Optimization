# Model card: `ppo_routing_agent`

## Overview

**Task.** Choose one of *k* candidate paths for a demand, given the network
state, the demand itself, and the QoS class.

**Output.** A discrete action in `[0, 5)`, an index into the candidate list.

**Where it runs.** `routing/learned/rl.py`, on every `algorithm="rl"` request.

> This is the model behind the most serious bug we found. The loader looked
> for `rl_router_final.zip`; training saved, and the repository
> shipped, `ppo_routing_agent.zip`. The resulting `FileNotFoundError` was caught
> by `except (FileNotFoundError, ImportError, Exception): return False` with no
> logging, so the "RL router" served a congestion-weighted heuristic — in effect
> Dijkstra — for the entire life of the project. Paths now come from
> `ml/model_registry.py`, which training also reads, and every failure branch
> logs.

## Training data

On-policy, generated during training by `ml/environments/routing_env.py`. Each
episode builds a **fresh** simulator (25 nodes, 50 links, degree 4) and warms it
up 0–30 steps. The previous version never reset between episodes, so utilization
random-walked to the [0, 1] boundaries and the agent spent ~95% of its budget on
states it would never see at inference.

## Architecture

Stable-Baselines3 `MlpPolicy`, default 2 × 64 actor and critic.

**Observation, 286 dimensions:**

| Block | Width | Contents |
|---|---|---|
| Link state | 50 × 4 = 200 | utilization, queue, loss, base latency |
| **The task** | 25 × 2 = 50 | one-hot source, one-hot destination |
| **The choices** | 5 × 6 = 30 | validity, hops, cost, mean/max util, loss per candidate |
| **The class** | 6 | QoS weights and constraints |

The last three blocks are new. The old observation encoded per-link features
*only*, while the environment re-drew `(src, dst)` every step. The agent was
asked to pick "path index 2" without being told which pair it was routing, and
the meaning of index 2 changed completely between steps. That is not a partially
observable MDP; it is an unobservable one, and it fully explains the flat
learning curve.

## Hyperparameters

| | |
|---|---|
| Algorithm | PPO |
| Timesteps | 250,000 |
| lr / n_steps / batch | 3e-4 / 2048 / 64 |
| n_epochs / γ / λ | 10 / 0.99 / 0.95 |
| clip / ent_coef / vf_coef | 0.2 / 0.01 / 0.5 |
| Seed | 42 (the previous version passed none, so no run was reproducible) |
| Wall clock | 960 s on 4 CPU cores |

**Reward.** Negative class-weighted path cost, minus an infeasibility penalty,
minus a global load term measured on the state *after* the chosen flow is
registered. That last detail matters: the old reward computed its
"global load balancing" terms over all links independently of the action, which
in policy-gradient terms is a pure state-dependent baseline — it shifts returns
and adds variance while contributing **exactly zero** gradient to the policy.

## Metrics

A raw episode return conveys nothing. What is reported is the **normalized
score**: where the policy sits between a random floor and a greedy-oracle
ceiling, evaluated on 20 identical seeded episodes.

| Policy | Mean return | 95% CI | Normalized |
|---|---|---|---|
| Random | −48.01 | [−49.40, −46.62] | 0.000 |
| Greedy cheapest candidate (≈ Dijkstra) | −34.66 | [−36.01, −33.31] | 0.903 |
| **PPO** | **−35.20** | [−36.40, −34.01] | **0.867** |
| Greedy oracle | −33.23 | [−34.29, −32.17] | 1.000 |

**Read this honestly.** PPO closes 87% of the random-to-oracle gap, which is
real learning — but it does **not** beat simply taking the cheapest candidate,
which scores 0.903. On a largely additive objective there is only ~10% of
headroom above greedy, and the policy has not captured it. That is the result.

Learning curve, before and after the environment fixes:

| | slope / 100k steps | r² | p | best checkpoint |
|---|---|---|---|---|
| Before | −0.094 | 0.001 | 0.878 | the **first** one, at 25k |
| After | +0.742 | 0.195 | 0.087 | 180k of 250k |

The reward function changed between these runs, so the vertical scales are not
comparable. The *shape* is: a flat line against a rising one. Note that the
"after" slope is still not significant at p < 0.05 with 10 evaluation points.

## Known failure modes

- **Does not beat the greedy baseline.** See above.
- **Topology-locked.** The observation width is tied to 50 links and 25 nodes.
  On a different topology the router detects the mismatch and falls back, rather
  than reshaping into nonsense — and flags the decision as a fallback.
- **Restricted to the candidate set.**

## Reproduction

```bash
python -m ml.training.train_rl         # ~16 min, CPU only
```
