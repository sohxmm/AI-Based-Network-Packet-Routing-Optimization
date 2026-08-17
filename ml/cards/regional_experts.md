# Model card: `multi_agent_region_{0..3}`

## Overview

**Task.** Decentralized hop-by-hop packet forwarding. Each region owns one
policy; a policy acts only when the packet is at a node inside its region.

**Input.** A purely **local** 113-dimensional observation. **Output.** One next
hop, chosen from the current node's ordered neighbour list.

**Where it runs.** `routing/learned/multi_agent.py`. A single route is produced
by several agents in sequence, which is what makes it multi-agent at all.

## What this is, precisely

**Decentralized execution, centralized training, independent learners.**

The previous implementation was N independently trained PPO agents that each saw
the *entire* global link vector and each emitted a *complete* end-to-end path,
with the acting agent chosen by a lookup on the source node. That is a
mixture-of-experts with a hardcoded gating function. Describing it as
"centralized-critic, decentralized-execution multi-agent RL" was the claim in
this project least likely to survive a viva.

What holds now, and is asserted by `tests/unit/ml/test_marl_locality.py` rather
than merely described:

| Property | How it is enforced | How it is verified |
|---|---|---|
| Local observation | The observation contains only the current node, the destination expressed relatively, and that node's incident links | Its width is **129 on both a 25-node and a 100-node network** |
| Next-hop action | `Discrete(MAX_DEGREE=8)`, indexing neighbours | Action space is degree-sized, not candidate-set-sized |
| Centralized critic | `AsymmetricActorCriticPolicy`: the actor's feature extractor slices off the global block, the critic's does not | Perturbing only the global block leaves the action distribution unchanged and moves the value estimate |
| No loops | Visited set enforced during the walk | Asserted over 100 random walks |

**What is still not claimed.** There is no explicit inter-agent messaging and no
shared critic across agents. Agents are independent learners trained in rotation
against frozen partners, cooperating through the shared environment and a shared
team reward term. That is a real and standard MARL setting; it is not a
joint-action solver.

## Training data

On-policy, from `ml/environments/regional_env.py`. 25 nodes partitioned by
greedy modularity into **4 regions of 8, 7, 6 and 4 nodes**.

## Architecture

- `AsymmetricExtractor` → MLP(input → 128 → 128), then SB3's default heads.
- Actor input: 113 (local only). Critic input: 129 (local + global summary).
- Observation: `[region one-hot (40) | destination features (3) | 8 neighbours × 8 features | QoS class (6)] ‖ [global summary (16)]`.
- The global summary is a fixed-width *histogram-and-moments* digest, not a raw
  link vector — a raw vector could not transfer between topologies.

The asymmetry lives in the **policy class**, not in a patch applied after
construction. That is not cosmetic: `PPO.load` rebuilds the policy from the saved
`policy_kwargs` and hands both extractors the same kwargs, so a post-construction
patch is silently lost on reload. The first attempt did exactly that, the
checkpoints failed to restore with a shape mismatch, and the router spent a
whole benchmark run on its heuristic fallback reporting `fallback_rate = 1.00`.

## Hyperparameters

| | |
|---|---|
| Algorithm | PPO, one per region |
| Schedule | 2 rounds × 4 regions × 25,000 steps = **50,000 per agent** |
| lr / n_steps / batch | 3e-4 / 1024 / 64 |
| Reward | −hop cost + 0.10 × progress shaping + 1.0 arrival − 1.0 loop/dead-end + 0.5 × shared team term (change in network-wide utilization variance) |
| Seed | 42 + region id |
| Wall clock | ~590 s on 4 CPU cores |

## Metrics

Mean episode return against a uniform-random next-hop floor, 5 seeded episodes
per region:

| Region | Nodes | Trained | Random | Improvement |
|---|---|---|---|---|
| 0 | 8 | 70.5 | −5.0 | +75.5 |
| 1 | 7 | 78.5 | +1.7 | +76.8 |
| 2 | 6 | 69.9 | −4.9 | +74.8 |
| 3 | 4 | 108.3 | +38.6 | +69.7 |

**All 4 regions beat random by a wide margin.** Random next-hop forwarding
frequently dead-ends or loops, so the floor is genuinely low; the margin
demonstrates that the agents learned to make progress, not that they route
optimally.

## Known failure modes

- **Not evaluated against Dijkstra per-region.** The comparison above is against
  random. The end-to-end comparison lives in the benchmark, where the multi-agent
  router competes on equal terms.
- **Hop cap.** A walk exceeding `3 × max(4, n/4)` hops is abandoned and the
  router falls back, flagged.
- **Untrained regions on a new topology.** Repartitioning a 100-node network
  produces regions with no policy; those nodes take a greedy hop and the decision
  records that an untrained region participated.
- **MAX_DEGREE = 8.** A node with more neighbours exposes only its 8 cheapest
  incident links.
- **50,000 steps per agent is a thin budget** for a harder problem than the
  single-agent task, which gets 250,000.

## Reproduction

```bash
python -m ml.training.train_regional   # ~10 min, CPU only
```
