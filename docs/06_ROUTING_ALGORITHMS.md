# Routing Algorithms

Eight strategies, one interface, one cost function. Every one of them takes a
`NetworkState`, a source, a destination and a `QoSProfile`, and returns a
`RoutingDecision`.

---

## 1. The shared foundations

### 1.1 The cost function

```python
# core/cost.py — the only definition in the repository
CONGESTION_EXPONENT = 2
CONGESTION_PENALTY_FACTOR = 4.0

def link_cost(link):
    return link.base_latency * (1 + 4 * link.utilization ** 2)
```

Quadratic in utilisation, so a link at 80% load costs 3.6x its idle latency and
a link at 100% costs 5x. This is what makes every algorithm congestion-aware.

This formula previously existed at **fourteen sites across eleven files**, with
three different exponents and two different penalty factors among them. That is
not a style problem: the RL agent was trained against one version of the cost and
served against another, so its policy was optimised for a network that did not
exist at inference time. There is now exactly one definition and everything
imports it.

### 1.2 The `Router` contract

```python
class Router(Protocol):
    name: str
    def find_route(
        self, state: NetworkState, src: str, dst: str,
        profile: QoSProfile | None = None,
    ) -> RoutingDecision: ...
```

Before this there were three different method names across six routers
(`find_route`, `predict`, `route`), so every call site needed a dispatch table
and adding an algorithm meant editing five files.

### 1.3 The candidate set

```python
core.paths.candidate_paths(state, src, dst, k=5, weighted=True)
```

k-shortest paths under the **congestion-weighted** metric. Every router except
Bellman-Ford chooses from this set.

`weighted=True` is the default, and it matters: training generated candidates one
way and serving generated them another, so the model was ranking a different set
of paths at inference than it had ever been trained on — a textbook train/serve
skew, hidden behind a boolean default.

**This is also a hard ceiling.** Nothing can select a path outside the candidate
set, including the oracle. Every learned "improvement" is a re-ranking of a
classically generated set.

### 1.4 `RoutingDecision`

```python
@dataclass
class RoutingDecision:
    source: str
    destination: str
    path: list[str]
    algorithm: str
    total_latency: float
    avg_utilization: float
    success: bool
    is_fallback: bool = False        # did a model decide this, or the heuristic?
    diagnostics: dict = field(default_factory=dict)
```

`is_fallback` is part of the **domain model**, not an API detail. A learned
router that quietly served a heuristic answer has to be distinguishable from one
that ran its model, everywhere in the system — in the API response, in the
benchmark aggregation, in the honesty gates and in the dashboard.

Failed routes return `path=[]`, `total_latency=inf`, `success=False`.

---

## 2. The eight algorithms

| Algorithm | File | Type | Role in the comparison |
|---|---|---|---|
| Dijkstra | `routing/classical/dijkstra.py` | Exact | The optimum for additive costs |
| Bellman-Ford | `routing/classical/bellman_ford.py` | Exact | Correctness cross-check |
| Constrained | `routing/classical/constrained.py` | Exact, filtered | **The honest QoS ceiling** |
| ACO | `routing/heuristic/aco.py` | Metaheuristic | Stochastic explorer |
| GNN | `routing/learned/gnn.py` | Learned ranker | Size-agnostic learned router |
| RL (PPO) | `routing/learned/rl.py` | Learned policy | Sequential decision maker |
| Multi-agent | `routing/learned/multi_agent.py` | CTDE | Decentralised routing |
| Random | `routing/random_baseline.py` | Uniform | **The honest floor** |

The floor and the ceiling are first-class rows in the results table, not
footnotes. Without a floor you cannot tell whether a model learned anything;
without a ceiling you cannot tell whether the remaining gap is worth chasing.

---

## 3. Dijkstra

Binary-heap shortest path over the congestion-weighted graph. `O((V+E) log V)`.

**It is optimal, and that is the single most important fact in this project.**
For non-negative additive edge costs Dijkstra returns the minimum-cost path — a
theorem, not a heuristic. Both conditions hold for `link_cost`: it is positive
for all utilisations, and a path's cost is the sum of its links'.

The consequence is that **no algorithm here can beat Dijkstra on best-effort
traffic.** Not with more training, not with a better architecture. The benchmark
confirms this: Dijkstra, Bellman-Ford, constrained, the GNN and PPO are
statistically indistinguishable on every best-effort scenario, with p > 0.5 and
negligible Cliff's delta throughout.

This is why the project's interesting regime is QoS constraints (§5), and why
`LEARNING_GUIDE.md` §18.1 spends several pages on it.

---

## 4. Bellman-Ford

Edge relaxation, `V-1` passes, with early termination and a negative-cycle check.
`O(VE)`.

**It is not an independent baseline, and it is not presented as one.** With
identical non-negative weights, Dijkstra and Bellman-Ford are both exact, so they
necessarily return the same cost. The measured `dijkstra_match_rate` is 1.00 in
every scenario, exactly as the theory requires.

It is here for two honest reasons: as a correctness cross-check on the cost
computation, and because it models how distance-vector protocols (RIP) actually
work — routers exchanging distance vectors with neighbours rather than computing
a global shortest path.

It is exempt from the degeneracy guardrail. Flagging Bellman-Ford for agreeing
with Dijkstra would be flagging it for being correct.

---

## 5. Constrained k-shortest paths

**This is the honest ceiling, and the most important classical algorithm here
after Dijkstra.**

```
1. Generate k candidate paths (congestion-weighted k-shortest)
2. Filter to those satisfying every hard constraint of the traffic class
3. Score the survivors by the class's weighted objective
4. Return the best; if none is feasible, return the least-infeasible
```

Under additive costs it agrees with Dijkstra. Under **QoS constraints** it does
not, and that difference is the entire point.

Dijkstra minimises a sum and is constraint-blind: it will happily return a path
whose bottleneck link is at 95% utilisation for a class whose hard limit is 70%.
The constrained router will not.

The module also exposes `qos_oracle()` and `qos_floor()` — the best and worst
feasible path in the candidate set — so a learned router's QoS performance can be
normalised the same way the PPO agent's return is.

### Measured result

Per-class QoS satisfaction on `qos_mixed_traffic`, 15 runs:

| Algorithm | emergency | interactive | gaming | bulk | best-effort |
|---|---|---|---|---|---|
| **Constrained** | **92.4%** | **97.1%** | **98.1%** | **99.9%** | 100% |
| GNN | 82.3% | 94.2% | 96.2% | 99.3% | 100% |
| Dijkstra | 84.0% | 92.7% | 95.1% | 99.2% | 100% |
| RL (PPO) | 78.5% | 90.8% | 92.8% | 98.6% | 100% |
| Random | 61.7% | 80.8% | 87.0% | 96.0% | 100% |

It wins every class, and it pays 1.6% in mean latency to do so — the trade you
want a constraint-aware router to make. No learned router beats it, and the
reason is that none of them was ever trained to satisfy a constraint.

---

## 6. Ant Colony Optimization

Probabilistic path construction with pheromone reinforcement.

```
        tau(i,j)^alpha  x  eta(i,j)^beta
P(j) = ----------------------------------
        sum_k tau(i,k)^alpha x eta(i,k)^beta
```

`tau` is pheromone on the edge, `eta = 1/cost` is heuristic desirability.

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | 1.0 | Pheromone influence |
| `beta` | 2.0 | Cost influence |
| `evaporation_rate` | 0.2 | Fraction evaporating per iteration |
| `Q` | 100 | Deposit constant |
| `n_ants` | 20 | Ants per iteration |
| `n_iterations` | 30 | Iterations per query |

Seeded (`Random(42)`) so results reproduce.

### Measured result, stated with its budget

ACO lands 62% worse than Dijkstra, with `diversity_index = 0.70` against
Dijkstra's 0.27 and 4.24 mean hops against 2.70. Those diagnostics say something
specific: it is exploring widely and has **not converged**. ACO is an anytime
algorithm whose quality is a function of iteration count, and 40 benchmark steps
is not enough for the pheromone distribution to sharpen.

At 100 nodes it degrades further — 8.60 hops against an optimum of 4.37, and
diversity 0.86, which is essentially random.

The honest statement is *"ACO is not competitive at this iteration budget"*, and
the budget belongs in the sentence.

The pheromone table is stateful and persists across requests, which is why
`service/state.py` keeps routers as singletons and why the benchmark harness
builds its **own** isolated router set — a sandbox experiment used to permanently
shift the live dashboard's pheromones.

---

## 7. GNN router

Ranks the candidate set with a 3-layer message-passing network. Full detail in
[`07_ML_AND_AI.md`](07_ML_AND_AI.md) §1.

Two things worth knowing from a routing perspective:

**It converges to Dijkstra on best-effort traffic** — 96–98% path agreement —
and the benchmark declares it. That is the correct outcome (§3), and the
declaration is enforced by an honesty gate.

**It is the only learned router that survives a change of topology size.** On the
100-node scenario it runs with **zero fallbacks**, because message passing does
not depend on node count. The same weights, trained on 25 nodes, apply
unmodified.

A loading bug worth remembering: `load_model` assigned `self._model` *before*
`load_state_dict`, so a failed load left an untrained model reporting
`is_trained=True` and quietly serving random weights. The model is now built into
a local variable and assigned only on success.

---

## 8. RL (PPO) router

Selects one of five candidate paths from a 286-dimensional observation. Full
detail in [`07_ML_AND_AI.md`](07_ML_AND_AI.md) §2.

The routing-relevant limitation is the **fixed observation width**. The vector is
sized `links x 4 + nodes x 2 + K_PATHS x 6 + QOS_FEATS` for a specific topology,
so a checkpoint trained on 25 nodes / 50 links cannot run on 100 nodes, or on a
topology with links removed.

`_observation_fits()` compares the built observation's width against
`model.observation_space.shape[0]`. On a mismatch it falls back to the heuristic,
sets `is_fallback=True`, and logs a WARNING naming the retrain command.

This fires for real: `fallback_rate` is **1.00** in both
`large_topology_100_nodes` and `link_failures_persistent`. The results files say
so in words, the report tables dagger the row, and an honesty gate fails CI if
the declaration ever disappears.

The old code path returned the same heuristic answer under the label `rl`,
with a latency number that looked competitive — and in `link_failures_persistent`
it was *better than Dijkstra's*. Someone would have written "the RL agent
outperforms Dijkstra under link failures". That sentence would have been about
five lines of Python.

---

## 9. Multi-agent (CTDE) router

Hop-by-hop routing, one PPO policy per region, each seeing only its own region's
113 local features. Control passes between agents as the packet crosses region
boundaries. Full detail in [`07_ML_AND_AI.md`](07_ML_AND_AI.md) §4.

`ensure_partition()` re-derives regions from the **live** topology whenever the
node set changes, rather than assuming the partition it was trained with.

The result is instructive: every region beats random on its own local objective,
and the composed end-to-end path is 61% worse than Dijkstra's. No agent optimises
end-to-end latency because no agent can see an end-to-end path. That is the price
of decentralised execution, and it is stated in the model card so it cannot be
quietly forgotten.

---

## 10. Random baseline

Uniform choice from the candidate set. **The floor.**

It exists because "the GNN gets 56.7 ms" means nothing on its own. Random gets
84.4 ms and Dijkstra gets 56.7 ms, so the interesting statement is that the GNN
has closed essentially all of that gap. Without the floor in the same table, a
reader has to take the claim on trust.

It also calibrates the statistics. Random against Dijkstra gives
`Cliff's delta = 1.0` and `p = 6.1e-05` — the smallest a 15-pair Wilcoxon can
produce — which shows the test detects real differences easily. That is what
makes its *failure* to detect one among the top five informative rather than
merely underpowered.

---

## 11. Predictive mode

Not an algorithm. A **mode** that swaps the state a router sees.

```
LSTM forecast  ->  build_forecast_state()  ->  gnn / rl routes on the forecast
```

`build_forecast_state()` returns `None` when no forecast is available. The
previous version ended `... or state`, so a missing forecaster silently meant
"route on the present while calling it a forecast" — and since the LSTM artifact
had never existed, `gnn_predictive` was **byte-identical** to `gnn` and
`rl_predictive` to `rl` in every published result. Two of eight benchmarked
"algorithms" were duplicate columns.

A permanent honesty gate fails CI if they ever match again.

---

## 12. Failover and convergence

`routing/failover.py` measures something no latency column can: how many ticks a
router needs to restore a **QoS-satisfying** route after a link on an active flow
is cut.

```python
measure_convergence(simulator, router, src, dst, failed_link, traffic_class)
# -> {"converged": bool, "convergence_steps": int | None,
#     "latency_before": float, "latency_after": float, ...}
```

It returns latency before and after alongside the step count, so a
fast-but-worse recovery is distinguishable from a slow-but-better one — a
distinction a single "convergence time" destroys.

`FailoverMonitor` runs the same logic continuously against watched flows and
emits `RerouteEvent`s to the dashboard. It detects breakage by checking
`path_is_intact()` **and** re-evaluating the QoS profile: a path whose links all
still exist but whose bottleneck has crossed the class's threshold is broken *for
that class*, even though every link is up.

---

## 13. Comparing them

```bash
curl -X POST http://localhost:8000/network/route/compare \
     -H "Content-Type: application/json" \
     -d '{"source": "R1", "destination": "R14", "traffic_class": "emergency"}'
```

Returns one `RoutingDecision` per algorithm against the same state, each with its
own `is_fallback` flag and diagnostics.

For statistically meaningful comparison use the benchmark, not this endpoint:

```bash
make bench      # 7 scenarios x 8 algorithms x 15 independent seeded runs
make report     # tables, effect sizes, CIs and figures
```

A single comparison on a single state is an anecdote. `docs/14_RESULTS_AND_FINDINGS.md`
is the result.
