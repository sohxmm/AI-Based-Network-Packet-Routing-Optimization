# Future Improvements

This is a roadmap, ordered by **value**, not by effort. Items already delivered
have been removed rather than annotated — Alembic migrations, CI, retention,
environment externalisation and the observation fix were all on the previous
version of this list and are all done.

Two items were explicitly considered and **scoped out** of the current work.
They are marked as such below so their absence reads as a decision.

---

## 1. The highest-value change

### 1.1 A min–max utilisation objective

**Why this is first.** Everything else on this list improves the system.
This one changes what the system can *conclude*.

The project's central result is that on an additive, non-negative cost, Dijkstra
is provably optimal, so no learned router can beat it — and the benchmark
confirms it, with the GNN reproducing Dijkstra's path 96–98% of the time. The
learned routers are competing for a tie.

Minimising the **maximum link utilisation across all simultaneous flows** is a
different problem:

- It is not a shortest-path problem. `max` is not a sum.
- Greedy per-flow routing is provably suboptimal for it.
- It is the objective real traffic engineering actually optimises (Hedera, CONGA,
  SR-TE).

That is a regime where a learned heuristic is competing against other heuristics
rather than against a theorem. **Explicitly out of scope for the current work;**
it requires a multi-flow formulation, a new reward, and a new scenario family.

### 1.2 Constraint-aware training objectives

The cheaper half of the same idea, and the most direct experiment this repository
leaves undone.

`qos_mixed_traffic` shows the classical `constrained` router winning every
traffic class — 92.4% emergency-class satisfaction against the GNN's 82.3% and
PPO's 78.5%. That is not evidence that learned routing fails here. The GNN was
trained to rank paths by **additive cost** and PPO was rewarded for **latency**;
neither has ever been asked to satisfy a constraint. They behave exactly as
trained.

Two concrete changes:

- A **feasibility term in the PPO reward**, penalising constraint violations
  rather than only latency.
- A **constraint-aware ranking loss** for the GNN: rank feasible paths above
  infeasible ones before ranking by cost.

The arena is already built. This is training a model that can compete in it.

### 1.3 A graph encoder for the RL agent

The PPO observation is a flat 286-dimensional vector sized for one topology.
`fallback_rate` is **1.00** on both `large_topology_100_nodes` and
`link_failures_persistent`: not one decision came from the trained policy,
because a 100-node graph does not fit a vector shaped for 25 nodes.

The GNN has no such problem — it runs on the 100-node scenario with zero
fallbacks, because message passing does not depend on node count.

Replacing the flat observation with the GNN's node embeddings would make one
checkpoint work across every scenario. This is an architectural argument the
benchmark has already made empirically.

---

## 2. Cheap and worth doing

### 2.1 Measure inference latency

`LEARNING_GUIDE.md` §18.2 argues that a learned router wins on **amortised
computation**: one forward pass versus `O(E log V)` per query, at 10⁴–10⁵ queries
per second. That argument is currently **asserted, not demonstrated**. No
wall-clock comparison exists.

A benchmark comparing a forward pass against a Dijkstra run at 25, 100, 1,000 and
10,000 nodes would take an afternoon and would convert the project's most
practically relevant claim from an argument into a result.

### 2.2 Model-quality regression floors

The evaluation scripts measure quality and write it to `ml/results/`, but no test
asserts a floor. Retraining with worse hyperparameters would produce a worse
model and a green build.

`train_lstm.py` already does this correctly — it refuses to save a checkpoint
that loses to persistence. Extending the same rule to the other three (GNN top-1
above some threshold, PPO normalized score above random) is a small change to the
honesty gates.

### 2.3 More seeds

`n_runs = 15`. Correct, and small: a true 1% latency difference is undetectable.
Raising it to 50 costs about an hour of CPU and materially narrows every
confidence interval. There is nothing clever here; it is just compute.

---

## 3. Bigger research directions

### 3.1 Real topologies and real traces

The single largest threat to validity is that the simulator is a caricature: no
packets, no queues, no TCP, and a congestion curve (`1 + 4u²`) that is plausible
rather than measured.

Two ways to attack it:

- **Public topologies.** The Internet Topology Zoo has hundreds of real ISP
  topologies. Real structure is closer to scale-free with a distinct core/edge
  split; conclusions about path diversity are likely sensitive to that.
- **Recorded traces.** `TraceReplaySource` already exists and the schema is
  documented in `datasets/README.md`. What is missing is a published dataset
  recorded from a real network.

### 3.2 Packet-level simulation

Integrating ns-3 or OMNeT++ would replace the modelling assumption with an answer,
at a large cost in scope. It is the honest long-term fix for the limitation in
`12_KNOWN_ISSUES.md` §1.1.

### 3.3 Multi-path routing and load splitting

Every decision here chooses exactly one path. Real networks split a flow across
several (ECMP, weighted-ECMP, flowlet switching). This changes the action space
from "pick a path" to "pick a distribution over paths" and makes the min–max
objective in §1.1 far more tractable.

### 3.4 Online / continual learning

**Explicitly out of scope for the current work.** Every model here is frozen
after offline training, so it cannot adapt to drift.

The reason it is out of scope is not difficulty — it is that a model which
updates itself inside a control plane is a much larger commitment than a model
that does not, and this project does not have the safety story to justify it. A
learned router that changes its own behaviour without a human in the loop needs
bounded-update guarantees, a rollback path and a way to detect that adaptation
has gone wrong. None of that exists here.

Approached properly, the first step is a **serve-time bandit over a small set of
fixed policies** rather than online gradient updates: bounded, interpretable, and
reversible.

### 3.5 Distribution shift and transfer

Models are trained and evaluated on topologies from the same generator. Seeds
differ, so this is not leakage, but it is not distribution shift either — and
real deployment is nothing but distribution shift. Train on one topology family,
evaluate on another, and report the drop.

---

## 4. Engineering

### 4.1 Authentication

The blocker for any non-local deployment. Every endpoint and the WebSocket are
open. JWT on the REST API, token validation on the WebSocket handshake, and a
read-only role for the dashboard.

### 4.2 Horizontal scaling

`AppState` is a module-level singleton holding the network source, the router set
(with ACO's stateful pheromone table) and loaded torch weights. Running more than
one worker gives each a divergent copy of the network.

The singleton is correctly motivated, so the fix is **shared state** — Redis for
the pheromone table and the live network, the database for history — not removing
it.

### 4.3 Configurable retention

`prune_snapshots()` keeps the newest `MAX_SNAPSHOTS = 10_000` rows, and
`SNAPSHOT_EVERY_N_STEPS = 10` decides how many ticks each row covers. Both are
module constants, so changing how much history the dashboard can show means
editing code and redeploying. They should be environment variables with the
current values as defaults.

### 4.4 Dashboard additions

Ordered by how much they would help someone using the tool:

- **Historical trend charts** for latency and utilisation. The data is already in
  `network_snapshots`; nothing renders it over time.
- **A topology editor** — add and remove routers and links interactively. This
  would make the "try it on your own network" story far more usable than editing
  a trace file.
- **Alerting** when utilisation crosses a threshold, reusing the reserved status
  palette.
- **A convergence view** for failover, plotting recovery time distributions
  rather than single events.

### 4.5 Packet-level visualisation

There is no packet-level anything: the simulator models link utilisation, not
packets. A `packet_logs` table was declared in an earlier version and never
written to, so the refactor dropped it rather than shipping an empty table as
though it meant something.

Animating packets along chosen paths, and recording per-packet delivery, would
make the closed loop visible rather than merely measured — and would need a
packet model to exist first (§3.2).

---

## 5. What is deliberately not on this list

| Not planned | Why |
|---|---|
| More routing algorithms | Eight is already more than the comparison can meaningfully separate. Another heuristic adds a row, not a result. |
| More themes | Four is enough to keep the theming engine honest. Ten was padding. |
| A bigger GNN | The largest model is 66k parameters because the tasks are small. A bigger model would memorise the simulator. |
| Beating Dijkstra on additive costs | Impossible, not merely hard. See `LEARNING_GUIDE.md` §18.1. Any roadmap item promising it is a roadmap item promising to violate a theorem. |
