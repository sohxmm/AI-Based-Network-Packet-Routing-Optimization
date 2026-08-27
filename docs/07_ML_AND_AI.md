# ML & AI Components

Four models, four separate tasks, four separate evaluations. Each is measured
against a baseline on its **own** task before it is allowed into the routing
comparison, because a model that fails its own task cannot be diagnosed from a
routing table.

Every result below is read from `ml/results/*.json`, which `make train`
regenerates and `scripts/verify_claims.py` checks the documentation against.

| Model | Task | Baseline | Result |
|---|---|---|---|
| GNN router | Rank candidate paths | Random choice (0.227 top-1) | **0.978 top-1** |
| PPO agent | Choose a path per demand | Random / greedy / oracle | **0.866 normalized** |
| Congestion LSTM | Predict next-step utilisation | Persistence | **+0.1497 skill** |
| Regional CTDE policies | Balance local utilisation | Random per region | **4/4 beat random** |

No GPU is required. All four train on CPU in roughly 35 minutes total.

---

## 1. Graph Neural Network router

**Architecture**: `ml/architectures/gnn.py` · **Training**: `ml/training/train_gnn.py`
· **Inference**: `routing/learned/gnn.py` · **Card**: `ml/cards/gnn_router.md`

### What it does

Given a network state and a set of candidate paths, score each path and pick the
best one. It is a **ranker**, not a path generator — candidates come from
`core.paths.candidate_paths()`.

### Architecture

```
node features (2)          edge features (4)
      │                          │
      ▼                          ▼
┌──────────────────────────────────────────┐
│  3 x message-passing layer               │
│    msg_ij  = MLP([h_i, h_j, e_ij])       │
│    h_i'    = MLP([h_i, mean_j(msg_ij)])  │  <- mean, not sum
└──────────────┬───────────────────────────┘
               │  node embeddings + edge messages
               ▼
┌──────────────────────────────────────────┐
│  path pooling: mean over the path's edge │
│  messages, concatenated with 9 explicit  │
│  path features (3 structural + 6 QoS)    │
└──────────────┬───────────────────────────┘
               ▼
        Linear -> scalar score per path
```

Two architectural decisions carry most of the weight:

**Mean aggregation, not sum.** With sum aggregation a node's embedding scales
with its degree, so on a graph with degree varying from 2 to 8 the embedding
magnitude encodes degree and swamps everything else. Mean makes the
representation degree-invariant.

**Edge messages are pooled, not node embeddings.** A path is a sequence of
*links*, and link congestion is what the routing decision turns on. Pooling node
embeddings loses exactly the quantity being optimised.

**47,361 parameters.** It is a small model, deliberately: the task is ranking 5
candidates on a 25-node graph, and a larger model would memorise rather than
generalise.

### Training

```bash
python -m ml.training.train_gnn         # ~10 min on CPU
```

| Parameter | Default |
|---|---|
| Epochs | 40 (early stopping, patience 8) |
| Train / val / test samples | 2500 / 600 / 600 |
| Hidden dimension | 64 |
| Learning rate | 1e-3 |
| Batch size | 16 |
| Seed | 42 |
| Loss | Pairwise margin ranking |

**Pairwise ranking loss, not regression to cost.** Regressing the exact cost of
every path solves a harder problem than the one that matters: only the
*ordering* affects the decision. The margin loss says "the better path should
score higher by at least a margin" and says nothing about absolute values.

**The validation split is the part worth studying.** The original code split one
simulator run 70/30. Adjacent timesteps in that run are ~85% correlated, so a
"validation" sample was a near-copy of a training sample and validation accuracy
measured memorisation of neighbouring timesteps. Train, validation and test now
come from **three independently seeded simulators** (42, 1042, 2042), so
validation measures generalisation to a network the model has never seen.

### Result (`ml/results/gnn_evaluation.json`)

| | Top-1 accuracy | Mean regret |
|---|---|---|
| Trained GNN | **0.978** | 0.00064 |
| Random choice | 0.227 | 0.676 |

Regret of 0.00064 means the 2% of decisions it gets wrong are between paths that
are nearly tied.

### The finding

On best-effort traffic the GNN reproduces Dijkstra's path 96–98% of the time,
and the benchmark flags it as degenerate. **This is correct behaviour**, not a
defect: with additive non-negative costs Dijkstra is provably optimal, so a
correctly trained ranker must converge to it. See `docs/12_KNOWN_ISSUES.md` §2.1
and `LEARNING_GUIDE.md` §18.1.

The GNN is also the only learned router that survives a change of topology size
— it runs on the 100-node scenario with zero fallbacks, because message passing
does not depend on node count.

---

## 2. PPO reinforcement-learning agent

**Environment**: `ml/environments/routing_env.py` · **Training**:
`ml/training/train_rl.py` · **Inference**: `routing/learned/rl.py` · **Card**:
`ml/cards/ppo_routing.md`

### The environment

`NetworkRoutingEnv` frames routing as an episodic decision problem.

**Observation** — `Box`, `links x 4 + nodes x 2 + K_PATHS x 6 + QOS_FEATS`
(286-dimensional for 25 nodes / 50 links), built by `ml/features.py`.

**Action** — `Discrete(5)`: choose one of five candidate paths.

**Episode** — a fresh simulator per episode; each step routes one demand,
registers the resulting flow, then advances the simulator.

### The bug that mattered

The original observation contained link state only — **not the (source,
destination) pair being routed** — while the task resampled a new pair every
step. The agent was asked "which of these five paths is best?" without being
told where the packet was going. The MDP was **unobservable**, so no policy could
beat chance, and the measured learning curve was accordingly flat: slope
−0.094 per 100k steps, r² = 0.001, p = 0.878, with the best checkpoint being the
first one taken.

Once the observation included the demand, the curve rose: **+0.742 per 100k
steps, r² = 0.195**.

A second bug in the reward function: a term that depended only on global network
state, not on the action. A state-dependent term with no action dependence
contributes exactly zero to the policy gradient — it is pure variance.

### Training

```bash
python -m ml.training.train_rl          # ~16 min on CPU
```

| Parameter | Value |
|---|---|
| Algorithm | PPO (`MlpPolicy`) |
| Total timesteps | 300,000 |
| Learning rate | 3e-4 |
| n_steps / batch / epochs | 2048 / 64 / 10 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Clip range / entropy coeff | 0.2 / 0.01 |
| Eval frequency | every 15,000 steps |
| Final evaluation | 20 episodes |
| Seed | 42 (passed to `PPO(...)`, so runs reproduce) |

### Result (`ml/results/rl_evaluation.json`)

| Policy | Mean return | 95% CI | Normalized |
|---|---|---|---|
| Random | −48.01 | [−49.40, −46.62] | 0.000 |
| **PPO** | **−35.20** | [−36.40, −34.01] | **0.866** |
| Greedy first-candidate | −34.66 | [−36.01, −33.31] | 0.903 |
| Oracle | −33.23 | [−34.29, −32.17] | 1.000 |

`normalized = (policy − random) / (oracle − random)`.

**Read both halves.** PPO covers 87% of the distance from random to the oracle,
which is real learning on a task that was previously unlearnable. It also does
**not** beat the one-line greedy heuristic, whose CI overlaps its own. Reporting
only the 0.866 would be true and misleading.

### Known limitation

The observation is fixed-width, so a checkpoint trained on 25 nodes cannot run
on 100 nodes or on a topology with links removed. In those scenarios the router
detects the mismatch, falls back honestly, sets `is_fallback`, and the benchmark
emits a warning. Replacing the flat observation with the GNN's node embeddings
is the fix; it has not been done.

---

## 3. Congestion LSTM

**Architecture**: `ml/architectures/lstm.py` · **Training**:
`ml/training/train_lstm.py` · **Inference**: `routing/learned/forecaster.py` ·
**Card**: `ml/cards/congestion_lstm.md`

### The task

Predict next-step utilisation for every link, so routing can act on where
congestion is *going* rather than where it is.

### Why the first version scored −1.77

The model predicted the utilisation **level**. Utilisation is an AR(1) process
with `a = 0.85`, which means "copy the last value" — persistence — is right to
within the one-step noise almost every time. Predicting levels forces the network
to spend its capacity relearning the identity function, and it lost to
persistence by a factor of 2.8. Skill score −1.77.

### Predicting the residual

```python
def delta(self, inputs):
    differenced = inputs[:, 1:, :] - inputs[:, :-1, :]
    sequence_output, _ = self.lstm(differenced)
    return self.MAX_DELTA * torch.tanh(self.output(self.dropout(sequence_output[:, -1, :])))
```

The model now sees **changes** and predicts a **change**, bounded to ±0.5 by a
tanh. The identity function is removed from the problem, so every unit of
capacity goes to the part persistence cannot do.

### Training

```bash
python -m ml.training.train_lstm        # ~1 min on CPU
```

| Parameter | Default |
|---|---|
| Simulator steps collected | 6,000 |
| Sequence length | 20 |
| Epochs | 60 (early stopping, patience 8) |
| Hidden size / layers | 64 / 2 |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Split | **Chronological** 70/15/15 |
| Seed | 42 |

The split is chronological, not shuffled. A shuffled split on a time series puts
timestep *t+1* in train and *t* in test, which is leakage.

The training script **refuses to save a checkpoint that does not beat
persistence**. A forecaster worse than doing nothing is not a forecaster.

### Result (`ml/results/lstm_evaluation.json`)

| Predictor | Test MSE |
|---|---|
| Window mean | 0.01681 |
| Persistence | 0.00134 |
| **LSTM** | **0.00114** |

Skill score vs persistence: **+0.1497**. Modest and genuine — the LSTM extracts
about 15% of the variance persistence leaves behind.

### Predictive routing

`build_forecast_state()` produces a forecast `NetworkState` that `gnn_predictive`
and `rl_predictive` route against. Before this model existed the artifact was
absent, the builder returned `None`, the caller fell through to `or state`, and
`gnn_predictive` was **byte-identical** to `gnn` in every published result file.
Two of eight benchmarked "algorithms" were duplicate columns. A permanent
honesty gate now fails CI if they ever match again.

Multi-step forecasts are autoregressive and drift; treat anything past 3–5 steps
with scepticism.

---

## 4. Regional multi-agent RL (CTDE)

**Environment**: `ml/environments/regional_env.py` · **Partitioning**:
`ml/environments/partition.py` · **Training**: `ml/training/train_regional.py` ·
**Inference**: `routing/learned/multi_agent.py` · **Card**:
`ml/cards/regional_experts.md`

### What it is, and what it is not

The previous implementation trained one PPO agent per region on the **global**
observation and had each choose a **complete end-to-end path**. Every agent could
see the whole network and act on the whole network, so they were not regional
agents — they were N copies of the single-agent problem with different weights.
Nothing about the system was multi-agent except the file name.

### What it is now

**Decentralised execution.** Each agent observes only its own region — 113 local
features (`ml/local_features.py`) — and chooses only the **next hop**. Routing is
a hop-by-hop walk in which control passes between agents as the packet crosses
region boundaries. An agent physically cannot see outside its region.

**Centralised training.** The critic sees 16 additional global features during
training only. This is CTDE, and it is standard: the value function's job is
variance reduction, it is discarded at inference, and giving it information the
actor lacks is legitimate precisely because it never influences the action.

### The serialization bug worth remembering

Stable-Baselines3 rebuilds both the policy and value feature extractors from
identical kwargs, so `PPO.load` reconstructed the actor with the critic's
129-dimensional input and failed with

```
size mismatch for vf_features_extractor.net.0.weight: [128,129] vs [128,113]
```

The fix is to encode the asymmetry **in a policy class** —
`AsymmetricActorCriticPolicy(ActorCriticPolicy)` — so it is part of what gets
serialized rather than something applied after construction.

### Training

```bash
python -m ml.training.train_regional    # ~9 min on CPU
```

| Parameter | Default |
|---|---|
| Rounds | 2 |
| Timesteps per region per round | 30,000 |
| Nodes | 25 (partitioned into 4 regions) |
| Local / global observation | 113 / 16 dims |
| Seed | 42 |

### Result (`ml/results/marl_evaluation.json`)

| Region | Nodes | Trained return | Random return | Improvement |
|---|---|---|---|---|
| 0 | 8 | 70.49 | −6.58 | +77.07 |
| 1 | 7 | 78.48 | −2.73 | +81.21 |
| 2 | 6 | 69.90 | −2.85 | +72.75 |
| 3 | 4 | 108.35 | +39.32 | +69.02 |

4 of 4 regions beat random on their local objective.

### The gap between component and system

End-to-end, `multi_agent` routes **61% worse** than Dijkstra at 3.67 hops against
an optimum of 2.70, and under QoS constraints it satisfies the emergency class
only 61.5% of the time — statistically indistinguishable from random's 61.7%.

Both numbers are correct, and the gap is the finding: **each agent is good at its
local task, and the composition of locally-good decisions is a globally poor
path.** No agent optimises end-to-end latency because no agent can see an
end-to-end path. That is the cost of decentralised execution, not a training
failure, and it is the honest answer to "why not do everything with multi-agent
RL": decentralisation buys scalability and failure isolation and pays for them in
path quality.

---

## 5. The model registry

`ml/model_registry.py` is the single source of truth for artifact paths.

It exists because of a specific bug: the loader looked for
`rl_router_final.zip` while training wrote `ppo_routing_agent.zip`. The file was
never found, the router fell back to a heuristic, and every published "RL" result
for the life of the project was produced by five lines of Python. No error was
raised, because the fallback was the designed behaviour for a missing model.

The registry now declares, per model: filename, whether it ships in the repo, and
the command that trains it. `log_model_inventory()` runs at service startup and
logs what loaded and what did not. `missing_models()` powers the dashboard's
model-status banner. `scripts/verify_claims.py` fails CI if a model declared as
shipping is absent, or if one declared absent is present.

Committed checkpoints (`ml/checkpoints/`):

| File | Model |
|---|---|
| `gnn_router.pt` | GNN router |
| `ppo_routing_agent.zip` | PPO agent |
| `congestion_lstm.pt` | Congestion LSTM |
| `multi_agent_region_{0..3}.zip` | Regional CTDE policies |

All are committed, so `docker compose up` runs real models with no fallbacks.

---

## 6. No external AI APIs

This project uses **no** external model APIs — no OpenAI, Anthropic, Gemini or
Ollama. Every model is defined, trained and executed locally with PyTorch and
Stable-Baselines3 on self-generated simulator data. There are no API keys, no
network calls at inference, and no per-request cost.

---

## 7. Design decisions

**Why a ranker for the GNN rather than a generator?** Generating a path
end-to-end with a neural network means the network can emit invalid paths —
disconnected, cyclic, or through failed links. Ranking a classically generated
candidate set makes every output valid by construction. The cost is a ceiling:
no learned model can find a path outside the candidate set.

**Why PPO rather than DQN?** The action space is small and discrete, which suits
either. PPO's clipped objective bounds how far a single update can move the
policy, which matters when the environment is stochastic and episodes are short.
Stable-Baselines3's implementation is well-tested, and its `Monitor` and
`EvalCallback` gave the evaluation curve the project needed.

**Why a heuristic fallback at all?** The system must work on a fresh clone
before anything is trained, and the fallback (congestion-aware shortest path) is
itself a strong baseline. The danger is that a fallback is *silent*, which is
exactly what went wrong before. Every fallback now sets `is_fallback`, is
aggregated into `fallback_rate`, is flagged above 20% in the results file, is
daggered in the report tables, and is checked by an honesty gate.

**Why small models?** The largest is 66k parameters. The tasks are small — rank
5 paths, predict 50 numbers — and a bigger model would memorise the simulator
rather than learn the structure. Small models also mean the whole pipeline
retrains on a laptop CPU in 35 minutes, which is what makes the results
reproducible by anyone who clones the repository.
