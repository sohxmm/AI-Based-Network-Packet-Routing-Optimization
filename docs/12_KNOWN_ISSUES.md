# Known Issues & Limitations

This document is a **current** statement of what is broken, what is limited, and
what is deliberately out of scope. It is not a history.

Two rules govern it:

<!-- verify-claims: refuted -->

1. **Every number here must exist in an artifact.** `python scripts/verify_claims.py`
   re-reads `ml/results/`, `experiments/results/` and `runs/`, and fails CI if a
   figure written in the documentation cannot be found in them. The previous
   version of this file claimed a PPO reward improvement "from -77 to -61 (+21%),
   with the best evaluation reward at -45.81". None of those three numbers
   appeared anywhere in the committed artifact — the actual `evaluations.npz`
   ranged from **-86.57 to -99.67** with the best result at the *first*
   checkpoint. The claim came from a training run that was never committed. That
   is why the verifier exists.

   (The paragraph above quotes those figures in order to refute them, which the
   verifier would otherwise flag. The `<!-- verify-claims: refuted -->` marker
   above permits it until the next heading. It is deliberately an explicit
   marker and not a turn of phrase, so nobody re-introduces a bad number by
   writing the word "previously" near it.)
2. **Fixed issues are removed, not annotated.** This file states what is true
   now, not what used to be.

---

## 1. Modelling limitations

These are the honest ceiling on what any result in this repository means.

### 1.1 The simulator is not a network

**Impact**: High (on interpretation, not on operation)
**Location**: `core/simulator.py`, `core/cost.py`

There are no packets, no queues, no TCP, no protocol overhead. Congestion is
modelled as `link_cost = base_latency × (1 + 4·u²)` — a plausible convex
congestion curve, not a measured one. Traffic is an AR(1) process around a
diurnal sine.

Every quantitative result in `docs/14_RESULTS_AND_FINDINGS.md` is a statement
about **this model**. Generalisation to real networks is an assumption. Use
`TraceReplaySource` with your own recorded measurements if you need results that
are about a real network.

### 1.2 One topology family

Topologies are Watts–Strogatz small-world graphs with average degree 4. Real ISP
topologies are closer to scale-free with a distinct core/edge split;
data-centre fabrics are regular Clos networks. Conclusions about path diversity
and about how much room a learned router has to differ from shortest path are
likely sensitive to this choice.

### 1.3 The candidate set bounds every result

**Location**: `core/paths.py` — `candidate_paths()`

Every router except Bellman–Ford selects from the k best paths under a
congestion-weighted metric. No model can choose a path outside that set — not the
GNN, not PPO, not the oracle. Learned "improvements" are therefore always a
**re-ranking of a classically generated set**, which caps the achievable gain and
is a large part of why `constrained` is so hard to beat.

### 1.4 Single-flow objective

Every decision is made for one flow in isolation. The standard traffic-engineering
objective — minimise the *maximum* link utilisation across all simultaneous flows
— is a global optimisation, is not a shortest-path problem, and is not
implemented here. It is the highest-value extension and is listed as such in
`LEARNING_GUIDE.md` §21.2. **Explicitly out of scope for the current work.**

---

## 2. Machine-learning issues

### 2.1 The GNN is degenerate on best-effort traffic (by design, and declared)

**Impact**: This is a finding, not a defect
**Location**: `routing/learned/gnn.py`, flagged automatically by `experiments/runner.py`

The trained GNN reproduces Dijkstra's path 96–98% of the time on best-effort
traffic, and the benchmark writes a warning saying so into every results file:

> `gnn: chooses the same path as Dijkstra 98% of the time, so it is degenerate
> and adds no information.`

This is the **correct** outcome. With non-negative additive edge costs, Dijkstra
returns the provably minimum-cost path; there is nothing better to find. The GNN
reaching 0.978 top-1 accuracy and 0.00064 mean regret means it learned the cost
structure well enough to rediscover the optimum from data.

The honesty gate (`tests/honesty/test_honesty_gates.py`) does **not** require the
absence of degeneracy — that would be unsatisfiable. It requires that degeneracy
be *declared* in `warnings`. The project is allowed a negative result; it is not
allowed an undisclosed one.

The regime where a learned router has room to win is QoS-constrained routing
(`qos_mixed_traffic`), where the problem is NP-hard and every method is a
heuristic. See `LEARNING_GUIDE.md` §18.1–18.2.

### 2.2 Fixed-width observations break the PPO agent on changed topologies

**Impact**: Medium
**Location**: `ml/features.py` — `build_observation()`; `routing/learned/rl.py` — `_observation_fits()`

The PPO observation is `links × 4 + nodes × 2 + K_PATHS × 6 + QOS_FEATS` = 286
for a 25-node / 50-link graph. Removing links changes that width, so a
checkpoint trained on the intact topology cannot be used.

This is observable in the committed results: under `link_failures_persistent`,
`rl` reports `fallback_rate = 1.00` and the results file states

> `rl: 100% of decisions came from the heuristic fallback, not a trained model.
> This row is a heuristic, not rl.`

The agent falls back honestly rather than crashing or pretending. The row's
latency is a heuristic's latency and is labelled as such.

**Fix path**: replace the flat observation with the GNN's node embeddings, which
are structurally size-agnostic. Not done.

### 2.3 PPO does not beat the greedy heuristic

**Impact**: Medium (on the project's thesis)
**Location**: `ml/results/rl_evaluation.json`

| Policy | Mean return | 95% CI | Normalized |
|---|---|---|---|
| Random | −48.01 | [−49.40, −46.62] | 0.000 |
| PPO | −35.20 | [−36.40, −34.01] | 0.866 |
| Greedy first-candidate | −34.66 | [−36.01, −33.31] | 0.903 |
| Oracle | −33.23 | [−34.29, −32.17] | 1.000 |

PPO covers 87% of the distance from random to the oracle — real learning on a
task that was previously unlearnable — but its CI overlaps the one-line greedy
heuristic's. The correct statement is: *the agent learned the task, and the task
is one a trivial heuristic already solves.*

### 2.4 The multi-agent system is good locally and poor globally

**Impact**: Medium
**Location**: `routing/learned/multi_agent.py`, `ml/cards/regional_experts.md`

All four regional policies beat random by 69–81 return points on their own local
objective (`ml/results/marl_evaluation.json`). End-to-end, `multi_agent` routes
**61% worse** than Dijkstra at 3.67 hops against an optimum of 2.70.

Both numbers are correct. Each agent optimises the utilisation variance of its
own region; no agent can see an end-to-end path, because that is what
decentralised execution means. Composing four locally-good myopic policies gives
a globally poor path. This is the price of the constraint, not a training bug.

### 2.5 ACO is not competitive at this iteration budget

**Impact**: Low
**Location**: `routing/classical/aco.py`

ACO lands 62% worse than Dijkstra with `diversity_index = 0.70` and 4.24 mean
hops — the signature of a pheromone table that has not converged. ACO's quality
is a function of iteration count and the benchmark's 40-step horizon does not
give it enough. The result should always be quoted **with the budget**:
"not competitive at 40 steps", not "does not work".

### 2.6 Autoregressive forecast drift

**Impact**: Low
**Location**: `routing/learned/forecaster.py`

Multi-step forecasts feed each prediction back as input for the next step, so
errors compound. Forecasts beyond 3–5 steps should be treated with scepticism.
The LSTM's measured skill score against persistence is **+0.1497** at one step
(`ml/results/lstm_evaluation.json`); no multi-step skill score is claimed
because none has been measured.

### 2.7 Trained and evaluated on the same generator

Models see topologies from the same process that generates evaluation
topologies. Seeds differ, so this is not leakage, but neither is it distribution
shift — and real deployment is nothing but distribution shift.

---

## 3. Statistical limitations

### 3.1 n = 15 runs

**Location**: `experiments/runner.py`, every `replication` block in `experiments/results/`

The unit of replication is one independently seeded run, and there are 15 of
them. This is correct — the previous design treated 20,000 autocorrelated
decisions as independent samples and produced p-values as low as `0.0` — but it
is a small sample. A true 1% latency difference would not be detectable.

Where the documentation says "indistinguishable", read **"indistinguishable at
n = 15"**, not "equal". The bootstrap CI printed beside every p-value is the
number to read.

### 3.2 Inference cost is never measured

The argument that a learned router wins on *amortised computation* (one forward
pass vs. `O(E log V)` per query) is asserted in `LEARNING_GUIDE.md` §18.2 and
**not demonstrated**. No wall-clock comparison exists. This would be cheap to add.

---

## 4. Operational issues

### 4.1 Live probing produces a star topology and cannot benchmark routing

**Impact**: Medium (on expectations)
**Location**: `core/sources.py` — `LiveProbeSource`

Probing *n* hosts from one machine gives one centre and *n* leaves: exactly one
path per destination. With one candidate path there is nothing to route between.

Live mode is a **measurement and visualisation** feature — real latency, real
loss, real graph — and it **cannot** compare routing algorithms. The UI says so
rather than rendering a comparison table of identical rows. Use
`TraceReplaySource` for anything that needs real *routing* data.

Live probing is opt-in (`LIVE_PROBE_ENABLED=1`), read-only, uses the
unprivileged system `ping`, never runs as root, and only ever contacts hosts the
operator lists explicitly. No scanning, no host enumeration, no traffic
injection.

### 4.2 Single-worker deployment only

**Impact**: Low
**Location**: `service/state.py`

`AppState` is a module-level singleton holding the network source, the routers
(including ACO's pheromone table) and loaded torch weights. Running Uvicorn with
`--workers > 1` creates one per process and the dashboard's state becomes
inconsistent.

The singleton is correctly motivated — pheromone tables and model weights
genuinely must persist across requests — so the fix is shared state (Redis, or
the database), not removing it. Not done.

### 4.3 No authentication or authorization

**Impact**: High (for any public deployment)

All REST endpoints and the WebSocket are open. This is acceptable for local
development and for the Docker Compose stack on a trusted network. It must be
addressed before exposing the service anywhere else.

### 4.4 Docker is not exercised in CI

**Impact**: Low
**Location**: `docker-compose.yml`, `service/Dockerfile`, `web/Dockerfile`

The images build and the compose file is correct by inspection, but CI runs the
test suite directly rather than through the containers, and the development
environment used to produce this work had no Docker daemon available. The compose
path is therefore **documented but not machine-verified**. If `docker compose up`
fails on your machine, that is a bug worth reporting rather than a known
limitation.

### 4.5 The retention window is fixed, not configurable

**Impact**: Low
**Location**: `service/db/retention.py`, `service/api/simulator.py`

`prune_snapshots()` runs every 10 minutes and keeps the newest
`MAX_SNAPSHOTS = 10_000` rows — a **row-count** cap, not an age cap. Combined
with `SNAPSHOT_EVERY_N_STEPS = 10`, that is roughly 28 hours of history at the
default 1 Hz tick.

Both numbers are module constants rather than settings, so changing the retention
window means editing code. Raising `TICK_SECONDS` shortens the window in wall
time rather than growing the table, which is the safe direction to fail, but it
is not adjustable without a redeploy.

The original schema had **no retention at all** and wrote a snapshot every tick:
about 860 MB a day, growing without bound, written from inside the 1 Hz
simulator loop.

---

## 5. Deliberately out of scope

Listed so that their absence reads as a decision rather than an oversight.

| Item | Why |
|---|---|
| Min–max utilisation objective | The highest-value extension; a different problem class requiring a multi-flow formulation. |
| Online / continual learning | Needs a safety story this project does not have. A self-updating model in a control plane is a much larger commitment. |
| Multi-path / load splitting | Every decision here chooses exactly one path. |
| Device programming (PCEP, OpenFlow, P4) | The layer that turns a computed path into a configured one. Entirely absent; see `LEARNING_GUIDE.md` §20.2. |
| Packet-level simulation (ns-3, OMNeT++) | Would replace §1.1 with a real answer, at a large cost in scope. |

---

## 6. How to check this document is still true

```bash
python scripts/verify_claims.py     # every number here, against the artifacts
pytest tests/honesty -v             # the gates that keep results honest
make benchmark && make report       # regenerate the results this file cites
```

`verify_claims.py` runs in CI on every push. If it passes, no figure in this
document has drifted from the artifact it came from.
