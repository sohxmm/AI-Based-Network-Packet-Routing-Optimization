# Project Overview & Structure

## 1. Introduction

**AI-Based Network Packet Routing Optimization** is a controlled benchmarking
platform for congestion-aware routing. It simulates a dynamic network, runs eight
routing strategies head-to-head under reproducible scenarios, and reports the
comparison with statistical rigour — effect sizes, confidence intervals, a
random floor, an optimal ceiling, and automatic detection of models that have
silently collapsed onto a classical algorithm.

The framing matters, so it is stated up front. This project does **not** claim
that AI routing beats Dijkstra. Under additive, non-negative link costs Dijkstra
returns the provably optimal path, so no method can beat it — and the benchmark
confirms exactly that, with the trained GNN reproducing Dijkstra's path 96–98%
of the time. The interesting question is what happens **outside** that regime:
under multi-class QoS constraints, where the problem is NP-hard and every method
is a heuristic. That is where the project's contribution lives, and
`LEARNING_GUIDE.md` §18 works through the argument in full.

### Project goals

- Simulate a dynamic, **closed-loop** network where routing decisions change
  future congestion
- Implement and compare eight routing strategies: Dijkstra, Bellman-Ford,
  constrained k-shortest paths, ACO, GNN, PPO, regional multi-agent RL, and a
  random floor
- Model **five QoS traffic classes** with hard constraints, so routing is not a
  single-objective shortest-path problem
- Forecast link congestion with an LSTM and feed the forecast into routing
- Measure **failover convergence** — how fast each router restores a
  QoS-satisfying path after a link is cut
- Report every comparison with a paired non-parametric test, an effect size and
  a bootstrap CI, at a correctly stated unit of replication
- Run the same stack against the simulator, a **recorded trace**, or **live
  measurements** of a network you own
- Stream live network state to a React dashboard over WebSocket, with history in
  PostgreSQL

---

## 2. Tech stack

| Layer | Technology |
|---|---|
| Service | Python 3.11+, FastAPI, Uvicorn |
| Web | React 18, Vite 5, Tailwind CSS 3 |
| ML / AI | PyTorch 2.x (CPU), Stable-Baselines3, Gymnasium |
| Database | PostgreSQL 16, SQLAlchemy 2.0 (async), Alembic |
| Graphs | NetworkX 3.x |
| Visualisation | D3.js 7, Recharts 2 |
| Real-time | WebSocket (FastAPI / browser native) |
| Statistics | SciPy, NumPy |
| Testing | pytest, Vitest, Testing Library |
| DevOps | Docker Compose, GitHub Actions, ruff, pre-commit |

No GPU is required at any point. All four models train on CPU in about 35
minutes total and their checkpoints are committed.

---

## 3. Repository structure

The layout is **layered by responsibility**, not by technology. The previous
`backend/` + `frontend/` split forced the domain model, the routing algorithms,
the training pipelines and the HTTP layer into one directory, which is why the
cost formula ended up duplicated at fourteen sites.

```
AI-Based-Network-Packet-Routing-Optimization/
│
├── README.md                    # Quick start and the results table
├── LEARNING_GUIDE.md            # The long-form study document
├── CONTRIBUTING.md  LICENSE  Makefile
├── docker-compose.yml  .env.example  ruff.toml  pytest.ini
├── WORKLOG_Soham.md  WORKLOG_Sneha.md
│
├── core/                        # Domain model — depends on nothing
│   ├── models.py                # LinkState, NetworkState, RoutingDecision
│   ├── cost.py                  # THE cost function. One definition, one file.
│   ├── paths.py                 # path_cost, path_links, candidate_paths
│   ├── qos.py                   # 5 traffic classes, profiles, feasibility
│   ├── simulator.py             # Closed-loop small-world simulator
│   └── sources.py               # NetworkSource: simulated / trace / live probe
│
├── routing/                     # Routing algorithms — depends on core
│   ├── base.py                  # The Router contract
│   ├── registry.py              # build_router_set(), ALGORITHM_NAMES
│   ├── random_baseline.py       # The floor
│   ├── failover.py              # FailoverMonitor, measure_convergence
│   ├── classical/               # dijkstra, bellman_ford, constrained
│   ├── heuristic/               # aco
│   └── learned/                 # gnn, rl, multi_agent, forecaster
│
├── ml/                          # Training — depends on core + routing
│   ├── model_registry.py        # Single source of truth for artifact paths
│   ├── features.py              # Graph tensors + PPO observation
│   ├── local_features.py        # Region-local observation for MARL
│   ├── architectures/           # gnn.py, lstm.py
│   ├── environments/            # routing_env, regional_env, partition
│   ├── training/                # train_gnn, train_rl, train_lstm, train_regional
│   ├── evaluation/              # baselines.py — random / greedy / oracle
│   ├── checkpoints/             # Committed model weights
│   ├── results/                 # Committed evaluation JSON
│   └── cards/                   # One model card per model
│
├── experiments/                 # Benchmarking — depends on everything above
│   ├── scenarios.py             # 7 declarative scenarios
│   ├── runner.py                # Closed-loop trajectories + degeneracy probe
│   ├── statistics.py            # Cliff's delta, Wilcoxon, bootstrap, entropy
│   ├── report.py                # Generates the results document and figures
│   └── results/                 # Committed benchmark output (JSON)
│
├── service/                     # HTTP/WS layer — depends on everything
│   ├── main.py                  # App, lifespan, background loop, /health
│   ├── state.py                 # Singletons; runtime-swappable network source
│   ├── api/                     # network, simulator, metrics, benchmark,
│   │                            #   experiments, websocket, dispatch
│   ├── schemas/                 # Pydantic request/response models
│   ├── db/                      # database, models, writes, retention
│   └── migrations/              # Alembic
│
├── web/                         # React + Vite dashboard
│   └── src/
│       ├── App.jsx              # 4 tabs, 4 themes
│       ├── components/          # Topology, path views, panels
│       │   └── benchmark/       # Scenario selector, metrics table, charts
│       ├── hooks/               # useNetworkStream, useRouteRequest, useModelHealth
│       └── utils/               # colorScales (CVD-validated), apiError
│
├── tests/
│   ├── unit/{core,routing,ml}/
│   ├── integration/api/
│   └── honesty/                 # Gates that keep the results honest
│
├── scripts/
│   ├── verify_claims.py         # Docs vs. artifacts. Runs in CI.
│   ├── plot_eval_curve.py
│   └── build_final_report.py
│
├── docs/                        # This documentation set
└── datasets/                    # Trace format and storage (README committed)
```

### Dependency direction

```
core  ←  routing  ←  ml  ←  experiments
  ↑         ↑        ↑
  └─────────┴────────┴──────  service  ←  web
```

Arrows point from dependency to dependent. `core` imports nothing from the rest
of the project; `service` imports from all of them and is imported by none. This
is enforced by convention and checked by review, and it is why the benchmark
harness can build its own isolated router set without touching the live
dashboard's state.

---

## 4. What changed in the current revision

The headline items:

| Area | Before | After |
|---|---|---|
| AI features actually running | 1 of 4 | 4 of 4 |
| Cost function definitions | 14 sites, 11 files | 1 (`core/cost.py`) |
| Statistical unit of replication | 20,000 correlated decisions | 15 independent runs |
| Simulation loop | Open loop (routing did not affect state) | Closed loop |
| Topology | Ring, degree 2, diameter 50 | Small-world, degree 4, diameter 8 |
| PPO training slope | −0.094/100k, r²=0.001 | +0.742/100k, r²=0.195 |
| LSTM skill vs persistence | −1.77 | **+0.1497** |
| GNN top-1 accuracy | Not measured | **0.978** (random: 0.227) |
| Test suite | Not installable | 60+ tests passing |
| Frontend lint | Broken config | Clean at `--max-warnings 0` |
| Traffic classes | 1 | 5, with hard constraints |
| Network sources | Simulator only | Simulator, trace replay, live probe |

---

## 5. Phase history

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Project scaffold & planning | Complete |
| Phase 1 | Network simulator | Complete |
| Phase 2 | Routing algorithms | Complete |
| Phase 3 | FastAPI service & database | Complete |
| Phase 4 | LSTM congestion predictor | Complete |
| Phase 5 | RL environment & PPO training | Complete |
| Phase 6 | React dashboard | Complete |
| Phase 7 | Integration, polish & theming | Complete |
| Phase 8 | Advanced algorithms & benchmarking | Complete |
| Phase 9 | Review and rebuild: honest measurement, QoS, failover, real-network sources | Complete |

---

## 6. Team

| Name | Role |
|---|---|
| Soham | Simulator core, routing algorithms, ML pipelines, benchmarking harness, service layer, dashboard integration, correctness rebuild |
| Sneha | Database (SQLAlchemy models, PostgreSQL/Docker setup), backend endpoints, algorithms (Dijkstra, ACO, PPO training), frontend components (React/D3), stress testing and infrastructure debugging |

---

## 7. Where to read next

| If you want | Read |
|---|---|
| To run it | `README.md` |
| To learn the subject in depth | `LEARNING_GUIDE.md` |
| The measured results | `docs/14_RESULTS_AND_FINDINGS.md` |
| What is still broken | `docs/12_KNOWN_ISSUES.md` |
| Traps that will cost you an afternoon | `docs/15_GOTCHAS.md` |
