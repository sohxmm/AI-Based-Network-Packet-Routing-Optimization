# System Architecture

## 1. The organising principle

The system is layered **by responsibility**, and the dependency direction is
strictly one-way:

```
core  <--  routing  <--  ml  <--  experiments
  ^           ^          ^
  |           |          |
  +-----------+----------+------  service  <--  web
```

Arrows point from a package to the packages that depend on it. `core` imports
nothing else in the project. `service` imports from everything and is imported by
nothing.

This matters more than it sounds. The previous layout was `backend/` +
`frontend/`, which put the domain model, the routing algorithms, the training
pipelines and the HTTP handlers in one tree with no rule about who could import
whom. The concrete consequences:

- The cost function was redefined at **14 sites in 11 files**, with three
  different exponents among them. The RL agent trained against one and served
  against another.
- The benchmark harness reached into the API's singleton router set and mutated
  it, so running a sandbox experiment permanently shifted the live dashboard's
  ACO pheromone table.
- Nothing could be tested without importing FastAPI.

With the layering in place, `core` is importable with only NetworkX and NumPy,
the benchmark builds its own isolated router set, and there is exactly one
definition of the cost.

---

## 2. Runtime architecture

```
+---------------------------------------------------------------+
|                     WEB  (React + Vite, :5173)                 |
|                                                                |
|  Live | Compare | Benchmark | Experiment      4 tabs, 4 themes |
|                                                                |
|  TopologyGraph (D3)   PathDivergenceView   FailoverPanel       |
|  PathCostBreakdown    ModelStatusBanner    NetworkSourcePanel  |
|                                                                |
|  hooks: useNetworkStream (WS) | useRouteRequest | useModelHealth|
+-------------------------------+--------------------------------+
                                |
       ws://:8000/ws/stream     |     http://:8000/*
                                |
+-------------------------------+--------------------------------+
|                   SERVICE  (FastAPI + Uvicorn, :8000)          |
|                                                                |
|  api/  network | simulator | metrics | benchmark | experiments |
|        websocket | dispatch                                    |
|                                                                |
|  lifespan: advance_simulator_forever()  <- crash-proof, 1 Hz   |
|            prune_snapshots_forever()    <- retention, 10 min   |
|                                                                |
|  state.py  AppState singleton:                                 |
|     NetworkSource (swappable) | router set | forecaster |      |
|     FailoverMonitor                                            |
+---------+--------------------------------------+---------------+
          |                                      |
          v                                      v
+---------------------------+       +---------------------------+
|   core / routing / ml     |       |   PostgreSQL 16  (:5433)  |
|                           |       |                           |
|   NetworkSource           |       |   routing_events          |
|    - SimulatedSource      |       |   network_snapshots       |
|    - TraceReplaySource    |       |   algorithm_metrics       |
|    - LiveProbeSource      |       |                           |
|                           |       |                           |
|   8 routers, 4 models     |       |   Alembic migrations      |
+---------------------------+       +---------------------------+
```

---

## 3. The layers

### 3.1 `core` — the domain

Depends on NetworkX and NumPy. Nothing else in the project.

| Module | Responsibility |
|---|---|
| `models.py` | `LinkState`, `NetworkState`, `RoutingDecision` |
| `cost.py` | **The** cost function. One definition. |
| `paths.py` | `path_cost`, `path_links`, `candidate_paths`, `max_path_utilization` |
| `qos.py` | 5 traffic classes, profiles, `evaluate_path`, `select_best_path` |
| `simulator.py` | Closed-loop small-world network simulator |
| `sources.py` | `NetworkSource` ABC + simulated / trace / live implementations |

Two decisions in `paths.py` are deliberate corrections rather than ports:
`path_cost` returns `inf` for an invalid path (the old copies filtered out
missing edges and under-reported the cost of a broken path), and
`candidate_paths` is congestion-weighted by default (fixing the train/serve
skew).

### 3.2 `routing` — the algorithms

One `Router` protocol, one `build_router_set()`, eight implementations across
`classical/`, `heuristic/` and `learned/`. Plus `failover.py` for reroute
detection and convergence measurement, and `random_baseline.py` for the floor.

Learned routers import torch **lazily**, so `import routing` costs milliseconds
and works on a machine with no ML stack installed.

### 3.3 `ml` — training

`model_registry.py` is the single source of truth for artifact paths. It exists
because the loader looked for `rl_router_final.zip` while training wrote
`ppo_routing_agent.zip`; the file was never found, the router silently fell back,
and every published "RL" result was produced by a heuristic.

`features.py` and `local_features.py` build observations. `architectures/` holds
the networks, `environments/` the Gym environments, `training/` the pipelines,
`evaluation/` the random/greedy/oracle baselines.

### 3.4 `experiments` — measurement

`scenarios.py` (7 declarative scenarios), `runner.py` (per-algorithm closed-loop
trajectories plus a separate open-loop degeneracy probe), `statistics.py`
(Cliff's delta, Wilcoxon, bootstrap CI, path entropy), `report.py` (generates the
results document, the README table and the figures).

### 3.5 `service` — HTTP and WebSocket

Routers per concern, Pydantic schemas in `schemas/`, database access in `db/`,
Alembic in `migrations/`. `dispatch.py` maps an algorithm name to a router so
every endpoint uses the same resolution.

### 3.6 `web` — the dashboard

React 18 + Vite + Tailwind + D3 + Recharts. Four tabs, four themes, all state
from three hooks.

---

## 4. Data flow

### 4.1 The simulator tick (1 Hz)

```
advance_simulator_forever()
  |
  +-- get_source().step()            advance the network one tick
  +-- handle_simulator_step(state)
  |     +-- broadcast over WebSocket to every connected client
  |     +-- persist a NetworkSnapshot (best-effort; failures are logged)
  +-- record the tick time            <- what /health reports on
  +-- sleep(TICK_SECONDS)
```

The loop used to catch only `CancelledError`, so any other exception killed the
task while the app stayed up, `/health` kept returning `{"status": "ok"}`, and
the dashboard froze with no error anywhere. It now survives failures, logs them
with a stack trace, backs off after 10 consecutive failures, and — critically —
`/health` reports the **age of the last successful tick**, so a dead loop is
externally detectable:

```json
{"status": "degraded", "simulator_last_tick_age_s": 47.3, "consecutive_tick_failures": 12}
```

### 4.2 A routing request

```
POST /network/route  {source, destination, algorithm, traffic_class}
  |
  +-- get_source().get_state()             current NetworkState
  +-- get_profile(traffic_class)           QoS weights + hard constraints
  +-- resolve_router(algorithm)            service/api/dispatch.py
  +-- router.find_route(state, src, dst, profile)
  |     +-- candidate_paths(...)           k-shortest, congestion-weighted
  |     +-- score / rank / select
  |     +-- set is_fallback if a model could not decide
  +-- evaluate_path(state, path, profile)  feasibility against the constraints
  +-- persist a RoutingEvent (best-effort)
  +-- return RoutingDecision + QoS evaluation + diagnostics
```

### 4.3 The closed loop

The single most consequential design decision. In the benchmark:

```python
decision = router.find_route(state, src, dst, profile)
if decision.success:
    sim.register_flow(decision.path, demand=0.5)   # <- the loop
```

The chosen path **loads the links it uses**, so the next decision observes a
network the previous decision changed.

Without this, routing is open-loop: every algorithm is scored against a movie it
cannot affect, and "congestion-aware routing" has nothing to be aware of. Every
comparison in the original benchmark was made under that condition.

Closing the loop forces a second decision: each algorithm gets its **own
simulator instance** per seed. Sharing one would mean ACO's wasteful paths load
the links Dijkstra then has to avoid, so Dijkstra's score would depend on which
competitors happened to be running.

---

## 5. The `NetworkSource` abstraction

```python
class NetworkSource(ABC):
    def get_state(self) -> NetworkState: ...
    def step(self) -> NetworkState: ...
    def reset(self) -> NetworkState: ...
    def register_flow(self, path, demand=1.0) -> None: ...
    def describe(self) -> dict: ...
```

| Implementation | Source of truth | Can benchmark routing? |
|---|---|---|
| `SimulatedSource` | Watts-Strogatz + AR(1) traffic | Yes |
| `TraceReplaySource` | Recorded JSONL/CSV measurements | Yes |
| `LiveProbeSource` | Unprivileged ICMP to named hosts | **No** — star topology |

This is the seam along which a research prototype becomes a controller: replace
the simulator with real telemetry and everything above it — featurisation,
inference, path selection — is unchanged.

Live probing is opt-in (`LIVE_PROBE_ENABLED=1`), read-only, never root, and only
contacts hosts listed explicitly. Its limitation is structural and documented in
the UI: *n* hosts probed from one machine gives one path per destination, so
there is nothing to route between.

The source is swappable at runtime (`POST /sim/source`), which is why
`state.py` clears the forecaster history and the failover monitor on a swap —
carrying utilisation history across a topology change would be meaningless.

---

## 6. Concurrency

| Concern | Approach |
|---|---|
| Simulator advance | One background asyncio task, 1 Hz, exception-safe |
| Snapshot retention | Separate background task, every 10 minutes |
| WebSocket broadcast | `asyncio.gather(..., return_exceptions=True)`, dead sockets pruned |
| Database writes | Async SQLAlchemy; failures logged, never fatal |
| Long benchmarks | `asyncio.to_thread`, so the event loop keeps ticking |
| Experiment jobs | `BackgroundTasks` with a bounded in-memory store (50 jobs, oldest finished evicted) |

**Single worker only.** `AppState` is a module-level singleton holding the
network source, the router set (including ACO's stateful pheromone table) and
loaded torch weights. A second Uvicorn worker would own a divergent copy of the
network and the dashboard would flicker between two realities. The Dockerfile
pins `--workers 1` with a comment saying why.

---

## 7. Failure handling

The design principle: **degrade visibly, never silently.**

| Failure | Behaviour |
|---|---|
| Database unavailable | API starts; history endpoints fall back to live-state estimates; a warning is logged |
| Alembic migration fails | Falls back to `create_all` with a warning that existing tables will **not** be altered |
| Model checkpoint missing | Router falls back to the heuristic, sets `is_fallback`, logs the retrain command, and the dashboard shows a model-status banner |
| Observation width mismatch | Same, with the mismatch in the log line |
| Simulator tick raises | Logged with a stack trace, loop continues, backs off after 10 failures, `/health` goes `degraded` |
| WebSocket client dies | Pruned from the broadcast set; other clients unaffected |
| No feasible QoS path | Returns the least-infeasible path with `qos_feasible: false`, rather than failing |

Every one of these is observable from outside the process: in `/health`, in
`/health/models`, in `fallback_rate`, or in the `warnings` block of a results
file.

---

## 8. Persistence

Three tables (`routing_events`, `network_snapshots`, `algorithm_metrics`), full
detail in [`05_DATABASE_SCHEMA.md`](05_DATABASE_SCHEMA.md).

Schema changes go through **Alembic**. `create_all` remains only as a fallback
for tests and throwaway databases: it can CREATE but never ALTER, so on an
existing deployment adding a column silently does nothing.

`network_snapshots` used to grow at one row per second — 86,400 a day, roughly
860 MB — with no retention at all. Two changes: only one tick in ten is
persisted (`SNAPSHOT_EVERY_N_STEPS = 10`), and `prune_snapshots()` runs every 10
minutes keeping the newest `MAX_SNAPSHOTS = 10_000` rows, about 28 hours of
history. The cap is on **row count**, not age, so a faster tick rate shortens
the window rather than growing the table.

---

## 9. Where the honesty machinery lives

Not one module, deliberately — it is threaded through the layers:

| Layer | Mechanism |
|---|---|
| `core` | `RoutingDecision.is_fallback` in the domain model |
| `routing` | Every learned router sets it; `failed_decision()` sets it |
| `ml` | `model_registry` declares what should exist and logs what did load |
| `experiments` | `fallback_rate`, `dijkstra_match_rate`, the `warnings` block, the degeneracy probe |
| `service` | `/health/models`, `warnings` in the benchmark response |
| `web` | `ModelStatusBanner`, `WarningsCallout`, `GuardrailBadge` |
| `tests` | `tests/honesty/` turns each guardrail into a gate |
| CI | `scripts/verify_claims.py` on every push |

The project already had good instincts here — `is_fallback` existed, so did
`dijkstra_match_rate`, so did a UI badge for "matches Dijkstra". What it did not
have was **enforcement**: the guardrails reported problems and nothing acted on
them. The gates are the difference.
