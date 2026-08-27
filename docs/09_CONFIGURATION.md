# Configuration

Everything has a working default. A fresh clone runs with no configuration at
all — `make up` creates `.env` from `.env.example` if it is missing, and every
compose variable has an inline fallback.

That used to be false. Three services declared `env_file: .env`, `.env` is
gitignored, and nothing created it, so the first command a reviewer ran died
with `env file .env not found`.

---

## 1. Environment variables

One file: `.env` at the repository root. It is gitignored; `.env.example` is
the committed template.

### 1.1 Service

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | unset | Async PostgreSQL DSN. Unset means no persistence — the API runs and history endpoints degrade to live-state estimates. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed origins |
| `TICK_SECONDS` | `1.0` | Simulator tick interval |
| `LOG_LEVEL` | `INFO` | Root log level |
| `LIVE_PROBE_ENABLED` | `0` | `1` allows live ICMP probing |
| `BENCHMARK_RESULTS_DIR` | `experiments/results` | Where the API reads committed results from |

CORS is read from the environment, not hardcoded:

```python
_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(CORSMiddleware,
                   allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
                   ...)
```

### 1.2 Database

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER` | `routinguser` | Compose database user |
| `POSTGRES_PASSWORD` | `change-me-before-deploying` | Compose database password |
| `POSTGRES_DB` | `routing_db` | Compose database name |

The template ships `change-me-before-deploying` rather than a plausible-looking
password, because a plausible-looking default is one people keep.

### 1.3 pgAdmin (dev profile only)

| Variable | Default |
|---|---|
| `PGADMIN_DEFAULT_EMAIL` | `admin@example.com` |
| `PGADMIN_DEFAULT_PASSWORD` | `change-me-before-deploying` |

pgAdmin is behind `profiles: ["dev"]` and does **not** start with
`docker compose up`. A database admin UI with a default password should not
appear on a port because someone ran the default command:

```bash
docker compose --profile dev up      # adds pgAdmin on :5050
```

### 1.4 Frontend

`VITE_*` variables are read at **build time**, not runtime.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_URL` | same-origin | Backend base URL |

Same-origin by default means the nginx image works with no configuration, and
the Vite dev server works through its proxy. The previous version hardcoded
`http://localhost:8000` in a hook, so any non-localhost deployment required a
source edit.

---

## 2. Ports

| Service | Port | Change it in |
|---|---|---|
| Service (FastAPI) | 8000 | `uvicorn` args / `docker-compose.yml` |
| Dashboard (Vite dev) | 5173 | `vite.config.js` / CLI |
| Dashboard (nginx image) | 5173 -> 80 | `docker-compose.yml` |
| PostgreSQL | 5433 -> 5432 | `docker-compose.yml` |
| pgAdmin | 5050 | `docker-compose.yml` |

PostgreSQL is published on **5433** so it does not collide with a locally
installed PostgreSQL on 5432.

---

## 3. Simulator parameters

`core/simulator.py`. These are constructor arguments, not environment
variables — they are experiment parameters, and an experiment parameter that can
be changed by an environment variable is one that can be changed without anyone
noticing.

| Parameter | Default | Meaning |
|---|---|---|
| `num_nodes` | 25 | Router count |
| `seed` | 42 | All randomness derives from this |
| `avg_degree` | 4 | Target average node degree |
| `background_flows` | 3 | Synthetic flows loading the network each tick |
| `load_per_flow` | 0.06 | Utilisation added per registered flow |
| `flow_decay` | 0.90 | Per-tick decay of accumulated flow load |
| `ar_coefficient` | 0.85 | AR(1) persistence of utilisation |
| `diurnal_period` | 40 | Steps per simulated day |
| `noise_sigma` | 0.03 | Per-tick Gaussian noise |
| `DIURNAL_CENTRE` | 0.30 | Mean offered load |
| `DIURNAL_AMPLITUDE` | 0.18 | Peak-to-trough swing |

Three of these were changed for specific measured reasons.

**`avg_degree = 4`.** The topology was previously a ring: degree 2, diameter 50
at 100 nodes. A ring has exactly two paths between any pair — clockwise and
anticlockwise — so there was nothing for a routing algorithm to be clever about,
and the benchmark could not have detected a difference between algorithms even if
one existed. It is now Watts-Strogatz small-world: 100 nodes gives 200 edges,
degree 4.0, diameter 8.

**Flow load folded into the AR(1) baseline.** Adding registered flow load
*outside* the AR(1) update amplified it by `1/(1-a)` at steady state — roughly
10x — driving mean utilisation to 0.68 and p95 to 1.0, i.e. every link saturated
and every path equally bad. It is now part of the offered load:

```python
offered = (DIURNAL_CENTRE + DIURNAL_AMPLITUDE * sin(phase)
           + congestion_bias + flow_load)
utilization = (a * previous + (1 - a) * offered + gauss(0, sigma))
```

Mean utilisation is now 0.39.

**`DIURNAL_CENTRE = 0.30`, `DIURNAL_AMPLITUDE = 0.18`** put the network in the
0.12–0.48 band where the quadratic congestion penalty actually discriminates
between paths. Near 0 every path is equally good; near 1 every path is equally
bad.

---

## 4. Cost function

`core/cost.py`. The only place this is defined.

```python
CONGESTION_EXPONENT = 2
CONGESTION_PENALTY_FACTOR = 4.0
```

Changing either changes every algorithm's behaviour simultaneously, which is the
point. It was previously redefined at 14 sites in 11 files with three different
exponents among them, which meant the RL agent trained against one cost and
served against another.

---

## 5. QoS classes

`core/qos.py`. Five classes, each with objective weights and **hard
constraints**:

| Class | Priority | Constrains |
|---|---|---|
| `emergency` | Highest | Tight latency, loss and bottleneck-utilisation budgets |
| `interactive` | High | Latency and jitter (voice/video) |
| `gaming` | High | Jitter above all |
| `bulk` | Low | Throughput; tolerates latency |
| `best_effort` | Default | No hard constraints |

The hard constraints are what make the routing problem interesting. Re-weighting
an additive cost per class changes nothing that matters, because Dijkstra solves
the re-weighted problem exactly too. A **constraint** — particularly the
non-additive bottleneck limit — changes the complexity class.

`PROFILE_VECTOR_DIM = 6`, and the profile vector is part of both the GNN's path
features and the PPO observation, so one model serves all five classes.

---

## 6. Benchmark parameters

CLI arguments to `python -m experiments.runner`:

| Argument | Default | Committed results used | Meaning |
|---|---|---|---|
| `--scenario` | `all` | `all` | Scenario name, or `all` |
| `--runs` | 30 | **15** | Independent seeded replications |
| `--steps` | 100 | **40** | Simulator steps per run |
| `--pairs` | 20 | **8** | Demands routed per step |
| `--seed` | 1000 | 1000 | Base seed; run *i* uses `1000 + i` |
| `--algorithms` | all | all | Restrict to a subset |
| `--persist` | off | off | Also write metrics to the database |

The committed results in `experiments/results/` were generated with the smaller
settings, which is recorded in each file's `replication` block rather than
inferred:

```bash
python -m experiments.runner --scenario all --runs 15 --steps 40 --pairs 8
```

`--runs` is the sample size in every reported statistic, and it is the unit of
replication. The previous harness ran once and treated 20,000 autocorrelated
decisions as independent samples, producing p-values as low as `0.0`. See
`LEARNING_GUIDE.md` §16.1.

---

## 7. Training parameters

Every training script takes CLI arguments and is seeded at 42. Full tables in
[`07_ML_AND_AI.md`](07_ML_AND_AI.md).

```bash
python -m ml.training.train_gnn --epochs 40 --hidden-dim 64 --seed 42
python -m ml.training.train_rl --timesteps 300000 --learning-rate 3e-4
python -m ml.training.train_lstm --seq-len 20 --epochs 60
python -m ml.training.train_regional --rounds 2 --timesteps-per-round 30000
```

---

## 8. Experiment sandbox caps

`service/api/experiments.py`. These **reject** over-budget requests rather than
clamping them, because silently clamping returns results that do not match what
was asked for.

| Cap | Value |
|---|---|
| `MAX_STEPS` | 300 |
| `MAX_PAIRS_PER_STEP` | 10 |
| `MAX_RUNS` | 10 |
| `MAX_TOTAL_DECISIONS` | 6000 per algorithm |
| `MAX_JOBS` | 50 (oldest finished evicted) |
| Valid topology sizes | 25, 50, 100 |

---

## 9. Deployment constraints

**One worker.** `service/state.py` holds a module-level singleton network source
and router set — including ACO's stateful pheromone table and loaded torch
weights. A second Uvicorn worker owns a divergent copy of the network and the
dashboard flickers between two realities. The Dockerfile pins `--workers 1` with
a comment explaining why.

**Container resource limits.** The backend service caps at 2 CPUs and 4 GB. An
unbounded ACO benchmark can otherwise consume the whole host.
