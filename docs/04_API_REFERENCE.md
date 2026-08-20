# API Reference

**The authoritative, always-current reference is the OpenAPI UI at
http://localhost:8000/docs** (or `/redoc`, or the raw schema at
`/openapi.json`). It is generated from the Pydantic models, so it cannot drift
from the code.

This document is deliberately *not* an exhaustive schema dump — a hand-copied
one would go stale the first time a field changed, which is the failure mode this
whole revision was written to prevent. It covers the endpoints whose **semantics
are not obvious from their signature**, and summarises the rest.

Base URL: `http://localhost:8000`

---

## 1. Endpoint map

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness **including simulator tick age** |
| GET | `/health/models` | Which model artifacts exist and which are loaded |
| GET | `/health/gnn` | GNN-specific load diagnostics |
| GET | `/network/state` | Current nodes, links, utilisation |
| GET | `/network/topology` | Structural stats: degree, diameter, connectivity |
| GET | `/network/source` | Which source is driving the state |
| GET | `/network/algorithms` | Available algorithms and their trained status |
| GET | `/network/candidates` | The candidate path set for a pair |
| POST | `/network/route` | Route one demand with one algorithm |
| POST | `/network/route/compare` | Route the same demand with every algorithm |
| GET | `/network/congestion-forecast` | LSTM forecast of link utilisation |
| GET | `/network/failover` | Watched flows and recent reroute events |
| POST | `/network/failover/watch` | Start watching a flow |
| DELETE | `/network/failover/watch` | Stop watching a flow |
| POST | `/network/failover/convergence` | Cut a link, measure recovery |
| POST | `/sim/step` | Advance one tick |
| POST | `/sim/reset` | Reset to step 0 |
| POST | `/sim/inject-failure` | Fail a link |
| POST | `/sim/restore-link` | Restore a failed link |
| POST | `/sim/source` | **Switch between simulator / trace / live** |
| GET | `/sim/source/health` | Per-target reachability (live mode only) |
| GET | `/metrics/summary` | Aggregate network metrics |
| GET | `/metrics/history` | Historical snapshots |
| GET | `/metrics/benchmark-history` | Stored benchmark runs |
| GET | `/metrics/algorithm-comparison` | Per-algorithm aggregates |
| GET | `/benchmark/results` | All committed scenario results |
| GET | `/benchmark/results/{scenario}` | One scenario |
| GET | `/benchmark/scenarios` | Available scenario names |
| POST | `/experiments` | Submit a sandbox benchmark job |
| GET | `/experiments/{job_id}/status` | Poll a job |
| GET | `/experiments/{job_id}/results` | Retrieve a finished job |
| WS | `/ws/stream` | Live state broadcast |

---

## 2. `GET /health` — read this one carefully

```json
{
  "status": "ok",
  "simulator_last_tick_age_s": 0.4,
  "consecutive_tick_failures": 0
}
```

`status` is `"degraded"`, not `"ok"`, once the last successful simulator tick is
older than 10 seconds.

This is not cosmetic. The background loop previously caught only
`CancelledError`, so any other exception killed the task while the application
stayed up. `/health` kept returning a bare `{"status": "ok"}`, every orchestrator
health check stayed green, and the dashboard simply froze with no error anywhere
in the system. A liveness endpoint that cannot report the liveness of the thing
that matters is worse than none.

Use `simulator_last_tick_age_s` as your readiness signal, not `status` alone.

---

## 3. `GET /health/models` — is the AI actually running?

```json
{
  "status": "ok",
  "models": {
    "gnn": {
      "file": "gnn_router.pt",
      "file_present": true,
      "expected_in_repo": true,
      "train_command": "python -m ml.training.train_gnn",
      "description": "Graph neural network path ranker",
      "loaded_in_memory": true
    },
    "rl": { "...": "..." },
    "lstm": { "...": "..." },
    "multi_agent": { "...": "..." }
  }
}
```

Two separate facts per model, because they can differ:

- `file_present` — the artifact exists on disk
- `loaded_in_memory` — it was successfully loaded into a router

A file that exists but failed to load is the interesting case, and it used to be
invisible. This endpoint backs the dashboard's `ModelStatusBanner`, so a user
looking at results always knows whether a model was behind them.

---

## 4. `POST /network/route`

```bash
curl -X POST http://localhost:8000/network/route \
     -H "Content-Type: application/json" \
     -d '{"source": "R1", "destination": "R14",
          "algorithm": "gnn", "traffic_class": "emergency"}'
```

```json
{
  "source": "R1",
  "destination": "R14",
  "path": ["R1", "R25", "R24", "R13", "R14"],
  "algorithm": "gnn",
  "total_latency": 63.47,
  "avg_utilization": 0.194,
  "success": true,
  "is_fallback": false,
  "diagnostics": {
    "qos": {
      "feasible": true,
      "score": 0.7509,
      "total_loss": 0.0,
      "bottleneck_utilization": 0.2357,
      "hops": 4,
      "violations": []
    },
    "candidates_considered": 5
  },
  "hops": [
    {"from": "R1",  "to": "R25", "base_latency": 23.0, "utilization": 0.205, "cost": 26.87},
    {"from": "R25", "to": "R24", "base_latency": 12.0, "utilization": 0.206, "cost": 14.04},
    {"from": "R24", "to": "R13", "base_latency":  8.0, "utilization": 0.236, "cost":  9.78}
  ]
}
```

Three fields deserve attention.

**`is_fallback`.** `true` means the answer came from the heuristic, not the named
algorithm — a missing checkpoint, an observation-width mismatch, a failed forward
pass. Always check it before attributing a result to a model. This is why it
lives in the domain model rather than being an API afterthought.

**`diagnostics.qos`.** The chosen path evaluated against the traffic class's hard
constraints. `feasible: false` with a populated `violations` array means the
router returned the least-infeasible path it could find rather than failing —
which is usually what an operator wants, but you have to be told.

**`hops`.** The per-link cost breakdown, so the total is auditable rather than
asserted. `cost = base_latency * (1 + 4 * utilization^2)`; the numbers above
multiply out.

`traffic_class` accepts `emergency`, `interactive`, `gaming`, `bulk`,
`best_effort`. Omitted, it is `best_effort`.

---

## 5. `POST /sim/source` — point it at a real network

```json
{"kind": "simulated", "num_nodes": 25, "seed": 42}
{"kind": "trace", "trace_path": "datasets/my_network.jsonl"}
{"kind": "live", "targets": ["192.168.1.1", "8.8.8.8"]}
```

Returns the new source's `describe()` plus the first state.

| Status | Meaning |
|---|---|
| 403 | `kind: "live"` without `LIVE_PROBE_ENABLED=1` |
| 404 | Trace file not found |
| 422 | `trace_path` missing for `kind: "trace"`, or an unparseable trace |

**Live mode caveat, repeated because it matters.** Probing *n* hosts from one
machine produces a **star** topology: one centre, *n* leaves, exactly one path per
destination. With one candidate path there is nothing to route between. Live mode
shows real latency and loss on a real graph; it **cannot** compare routing
algorithms. The dashboard says so rather than rendering a table of identical rows.

It is read-only throughout: unprivileged system `ping`, never root, no scanning,
no host enumeration, no traffic injection, and only the hosts you list.

Switching sources clears the forecaster's history and the failover monitor —
carrying utilisation history across a topology change would be meaningless.

---

## 6. `POST /network/failover/convergence`

```json
{"source": "R1", "destination": "R14",
 "failed_link": ["R25", "R24"],
 "algorithm": "gnn", "traffic_class": "emergency"}
```

Records the route, cuts the link, then steps the simulator until the router
produces a path that is both successful **and** feasible under the traffic
class's profile.

```json
{
  "converged": true,
  "convergence_steps": 1,
  "failed_link": ["R25", "R24"],
  "algorithm": "gnn",
  "latency_before": 63.47,
  "latency_after": 71.02,
  "path_before": ["R1", "R25", "R24", "R13", "R14"],
  "path_after":  ["R1", "R2", "R13", "R14"]
}
```

Latency before and after is returned alongside the step count on purpose: a
single "convergence time" cannot distinguish a fast recovery onto a worse path
from a slow recovery onto a better one.

---

## 7. `GET /benchmark/results`

Returns every committed scenario, each with per-algorithm metrics, the
`replication` block, `models_loaded`, and the `warnings` array.

Per algorithm you get `mean_latency`, `p95_latency`, `success_rate`,
`fallback_rate`, `qos_satisfaction_rate` (plus `qos_satisfaction_rate__<class>`
for mixed-traffic scenarios), `mean_path_max_utilization`,
`p95_path_max_utilization`, `diversity_index`, `mean_hops`,
`dijkstra_match_rate`, and — for everything except Dijkstra —
`comparison_vs_dijkstra`:

```json
"comparison_vs_dijkstra": {
  "n_runs": 15,
  "mean_diff": 0.42, "pct_diff": 0.74,
  "wilcoxon_p_value": 0.561,
  "cliffs_delta": 0.031, "effect_magnitude": "negligible",
  "ci95_low": -0.64, "ci95_high": 1.48
}
```

Read that block as a unit. `p = 0.56` alone tempts you to write "no difference".
The CI is the defensible statement: the difference is somewhere between 0.64 ms
better and 1.48 ms worse, and at n = 15 it cannot be distinguished from zero.

**`warnings` is not decoration.** It is where the system reports its own
problems:

```json
"warnings": [
  "gnn: chooses the same path as Dijkstra 98% of the time, so it is degenerate and adds no information.",
  "rl: 38% of decisions came from the heuristic fallback, not a trained model. This row is a heuristic, not rl."
]
```

`known_limitations` in the response is read from `experiments/README.md`.

---

## 8. `POST /experiments` — the sandbox

Submit a custom benchmark; poll; retrieve.

```json
{
  "topology_size": 25,
  "congestion_profile": "normal",
  "failure_rate": 0,
  "failure_pattern": "none",
  "steps": 40,
  "pairs_per_step": 8,
  "runs": 5,
  "algorithms": ["dijkstra", "gnn", "rl"],
  "traffic_classes": ["emergency", "best_effort"]
}
```

Hard caps, which **reject** rather than clamp:

| Cap | Value |
|---|---|
| `steps` | 300 |
| `pairs_per_step` | 10 |
| `runs` | 10 |
| `steps x pairs_per_step x runs` | 6000 per algorithm |
| `topology_size` | one of 25, 50, 100 |

Silently clamping an over-budget request would return results that do not match
what was asked for, which is worse than an error.

`runs` defaults to **3**, not 1, and counts against the budget. A single
trajectory cannot support the statistics this API reports back — see
`LEARNING_GUIDE.md` §16.1 on pseudo-replication. A sandbox run therefore reports
the same `comparison_vs_dijkstra` block as the committed benchmark, so a user
cannot accidentally compare a rigorous number against a casual one.

| Endpoint | Status codes |
|---|---|
| `GET /experiments/{id}/status` | 200, 404 |
| `GET /experiments/{id}/results` | 200, 404, **409** (still queued or running), 500 (job failed) |

The job store holds 50 jobs and evicts the oldest finished ones.

---

## 9. `WS /ws/stream`

Connect and receive the simulator state once per tick:

```json
{"type": "state_update", "data": {"step_count": 142, "nodes": [...], "links": [...]}}
```

Broadcast uses `asyncio.gather(..., return_exceptions=True)`, so one dead client
cannot stall the others, and disconnected sockets are pruned from the set.

---

## 10. Errors

Standard FastAPI shape:

```json
{"detail": "Live probing is disabled. Set LIVE_PROBE_ENABLED=1 to enable it, and only point it at networks you are authorised to measure."}
```

| Code | Used for |
|---|---|
| 400 | Malformed request the schema cannot express (e.g. source equals destination) |
| 403 | Live probing disabled |
| 404 | Unknown scenario, job, node, link or trace file |
| 409 | Experiment results requested before completion |
| 422 | Schema or cap violation (Pydantic) |
| 500 | A failed experiment job |

Error messages name the fix, not just the fault. That is a deliberate convention
throughout the service.

---

## 11. Authentication

There is none. Every endpoint and the WebSocket are open.

This is acceptable for local development and for the Compose stack on a trusted
network. It must be addressed before exposing the service anywhere else — see
`12_KNOWN_ISSUES.md` §4.3.
