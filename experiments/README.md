# Benchmark methodology and known limitations

This file is **required**, not decorative. `service/api/benchmark.py` reads it to
populate the dashboard's *Known Limitations* panel. It did not exist before, so a
feature built specifically to be honest rendered empty.

Regenerate all results with:

```bash
make bench          # or: python -m experiments.runner --scenario all
```

---

## What is measured

Per algorithm, per scenario, across **N independently seeded runs**:

| Metric | Meaning |
|---|---|
| `mean_latency` | Mean congestion-adjusted path latency, ms. Lower is better. |
| `p95_latency` | 95th percentile latency — the tail a user actually notices. |
| `qos_satisfaction_rate` | Fraction of decisions whose path met every constraint of its traffic class. **This is the headline metric for QoS scenarios**, not latency. |
| `success_rate` | Fraction of demands for which any path was returned. |
| `fallback_rate` | Fraction of decisions made by the heuristic fallback rather than a trained model. |
| `dijkstra_match_rate` | Fraction of decisions identical to the baseline's, measured on a shared open-loop trajectory. |
| `diversity_index` | Normalised entropy of the paths chosen per (source, destination) pair. |
| `mean/p95_path_max_utilization` | The bottleneck link on the chosen path — how close to saturation the route runs. |
| `comparison_vs_dijkstra` | Wilcoxon p-value, Cliff's delta and a bootstrap 95% CI, computed **across runs**. |

## The unit of replication

**One independently seeded run.** Not one routing decision.

This is the single most important methodological point in the project. An
earlier version of this harness ran a paired Wilcoxon test over 1,000 steps ×
20 pairs = 20,000 observations *from a single trajectory*. Successive simulator
steps are heavily autocorrelated — step *t* and step *t+1* are almost the same
network — so those 20,000 numbers are approximately one independent observation
repeated. With that much pseudo-replication **any** difference becomes
"significant", which is why every result file reported `wilcoxon_p_value: 0.0`.
A literal zero is numerical underflow, not a p-value.

Each algorithm now runs its **own closed-loop trajectory** per seed. That is not
a stylistic choice: once routing decisions change the network, algorithms cannot
share a trajectory, because whichever ran first would pollute the state the
others observe. Within a seed, every algorithm sees the identical topology, the
identical background traffic and the identical demand schedule, so the
comparison stays paired at the level that matters.

## Guardrails

- **`fallback_rate`** — the fraction of decisions made by the heuristic rather
  than a trained model. A high value means the reported "AI" result is not AI.
- **`dijkstra_match_rate`** — near 1.0 means the algorithm is degenerate and adds
  nothing over the baseline. Measured on a *separate shared-state pass*, because
  in the closed-loop benchmark the algorithms' networks legitimately diverge and
  comparing their paths would measure trajectory drift instead of similarity.
- **`random_baseline`** — picks uniformly among the candidate paths. Any learned
  method that does not beat this has not learned anything.
- **`warnings`** — emitted into every result file. If an algorithm was degenerate
  or ran mostly on its fallback, the artifact says so.

---

## **Limitation**: Known limitations of these results

1. **The network is synthetic.** Every published number comes from
   `core/simulator.py`, not from real traffic. The platform can also consume a
   recorded trace or live ICMP measurements, but no benchmark result is produced
   from either — live probing yields a star topology with one path per
   destination, where no routing algorithm can differentiate.

2. **A good learned ranker on a single additive objective converges to
   Dijkstra, and that is correct.** Dijkstra is provably optimal for any additive
   edge cost. On best-effort traffic the GNN reproduces its path almost every
   time, and the degeneracy warning fires. This is not a failure of the model; it
   is what optimality means. Learned policies can only differentiate where the
   objective is *not* a single additive cost — which is why the QoS scenarios
   exist.

3. **Bellman-Ford is not an independent baseline.** With identical non-negative
   weights, it and Dijkstra are both exact, so they necessarily return the same
   cost. It is reported as a correctness cross-check and is excluded from the
   degeneracy guardrail for that reason.

4. **The oracle is greedy, not optimal.** The QoS ceiling is the best candidate
   in a *k*=5 shortest-path set, chosen greedily per decision. It ignores the
   downstream consequences of its own load, so a policy that plans ahead can in
   principle exceed it; a normalized score above 1.0 is meaningful rather than a
   bug.

5. **ACO's cost is dominated by its search budget.** 30 iterations × 20 ants per
   decision makes it roughly ten times slower than every other algorithm here,
   and its wall-clock cost is not reflected in the latency metric, which measures
   the *path* rather than the time taken to find it.

6. **Results are reproducible only for a fixed model set.** Retraining changes
   the learned rows. Trained artifacts are committed, so a fresh clone reproduces
   the published numbers exactly; `make train` will not.

7. **Sample sizes are modest.** The shipped results use 15 runs × 40 steps × 8
   demands per scenario, chosen so the full suite completes in about half an
   hour on a laptop CPU. Wilcoxon is reported only when at least 6 independent
   runs are available.

---

## Reproducing

```bash
# Exactly the shipped numbers (uses the committed model checkpoints):
python -m experiments.runner --scenario all --runs 15 --steps 40 --pairs 8

# One scenario, more replication:
python -m experiments.runner --scenario qos_mixed_traffic --runs 30

# Check that nothing published contradicts an artifact:
make verify
```

Two runs of the same command produce byte-identical output. The previous harness
used the unseeded global `random` module in seven places and had no `--seed`
flag, so it did not.
