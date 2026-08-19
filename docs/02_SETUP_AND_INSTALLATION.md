# Setup & Installation Guide

Three ways to run this, from least to most work. Pick one.

| Path | Time | You get |
|---|---|---|
| [Docker](#1-docker-one-command) | ~5 min | Everything, including PostgreSQL |
| [Local development](#2-local-development) | ~10 min | Hot reload, a debugger, the test suite |
| [Retrain from scratch](#4-retraining-the-models-optional) | +35 min | Every model rebuilt on your machine |

**No GPU is needed for any of them.** All four models train on CPU, and their
checkpoints are committed, so nothing has to be trained before the dashboard
works.

---

## Prerequisites

| Tool | Minimum | Needed for |
|---|---|---|
| Docker Desktop / Engine | 24.x | Path 1 |
| Python | 3.11 | Paths 2 and 3 |
| Node.js | 18 | Path 2 (the dashboard) |
| Git | 2.x | All |

---

## 1. Docker: one command

```bash
git clone https://github.com/sohxmm/AI-Based-Network-Packet-Routing-Optimization.git
cd AI-Based-Network-Packet-Routing-Optimization
make up
```

That is the whole thing. `make up` creates `.env` from `.env.example` if it is
missing, then runs `docker compose up --build`.

| URL | What |
|---|---|
| http://localhost:5173 | The dashboard |
| http://localhost:8000/docs | Interactive API reference |
| http://localhost:8000/health | Liveness, including simulator tick age |
| localhost:5433 | PostgreSQL (mapped off 5432 to avoid clashing with a local install) |

`make up` used to fail on a fresh clone. Three services declared
`env_file: .env`, `.env` is gitignored, and nothing created it, so the first
command a reviewer ran died with `env file .env not found`. The compose file now
marks it `required: false` and every variable has a default.

pgAdmin is behind a profile, because a database admin UI with a default password
should not start just because someone ran the default command:

```bash
docker compose --profile dev up      # adds pgAdmin on :5050
```

Stop and clean up:

```bash
make down       # stops everything and removes volumes
make logs       # follow the backend logs
```

> **Note.** The Docker path is correct by inspection and by the healthchecks in
> both Dockerfiles, but the environment this revision was developed in had no
> Docker daemon, so `docker compose up` was **not** executed end-to-end there.
> If it fails on your machine, treat that as a bug worth reporting rather than a
> known limitation.

---

## 2. Local development

### 2.1 Python environment

From the repository root — **not** from a subdirectory. The service imports
`core`, `routing`, `ml` and `experiments`, which are top-level packages.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r service/requirements.txt
pip install -r service/requirements-dev.txt   # tests, ruff, pre-commit
```

If `torch` is slow to install, the CPU wheel is smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2.2 Database (optional)

The API runs without a database. History and metrics endpoints degrade to
live-state estimates and say so in the logs. To get persistence:

```bash
cp .env.example .env
docker compose up -d db
```

Schema is applied by Alembic at startup. `create_all` remains only as a fallback
for tests and throwaway databases — it can CREATE but never ALTER, so it is not
a migration path.

### 2.3 Run the service

```bash
uvicorn service.main:app --reload --port 8000
```

`--workers 1` in production. `service/state.py` holds a module-level singleton
network source and router set; a second worker would own a divergent copy of the
network and the dashboard would flicker between two realities.

Check it:

```bash
curl http://localhost:8000/health
# {"status":"ok","simulator_last_tick_age_s":0.4,"consecutive_tick_failures":0}
```

`status` is `degraded`, not `ok`, when the simulator loop has stopped ticking.
A bare `{"status":"ok"}` used to be returned even when the background task had
died, so the dashboard froze while health checks stayed green.

### 2.4 Run the dashboard

```bash
cd web
npm install
npm run dev            # http://localhost:5173
```

The header shows **Live stream connected** in green once the WebSocket is up.

---

## 3. Verify the installation

```bash
make test        # pytest + vitest
make lint        # ruff + eslint, both at zero tolerance
make verify      # honesty gates + documented claims vs. artifacts
```

Expected: 120 Python tests and 18 frontend tests passing, both linters clean,
and the verifier reporting no violations.

`make verify` is the interesting one. It re-reads `ml/results/`,
`experiments/results/` and `runs/`, and fails if a number written in the
documentation is not present in the artifacts, or if a model declared as
shipping is missing from disk.

---

## 4. Retraining the models (optional)

Checkpoints are committed, so this is never required. To rebuild them:

```bash
make train       # all four, ~35 min on a laptop CPU
```

Or individually:

```bash
python -m ml.training.train_gnn        # ~10 min -> ml/checkpoints/gnn_router.pt
python -m ml.training.train_rl         # ~16 min -> ml/checkpoints/ppo_routing_agent.zip
python -m ml.training.train_lstm       # ~1 min  -> ml/checkpoints/congestion_lstm.pt
python -m ml.training.train_regional   # ~9 min  -> ml/checkpoints/multi_agent_region_*.zip
```

Every script is seeded (42) and writes a JSON evaluation to `ml/results/`
alongside the checkpoint. `train_lstm.py` **refuses to save** a model that does
not beat persistence.

TensorBoard is optional. If it is not installed, `train_rl.py` says so and
carries on rather than pointing you at an empty log directory.

### Regenerating the benchmark

```bash
make bench       # 7 scenarios x 8 algorithms x 15 runs, ~20 min
make report      # results document, README table, per-scenario figures
```

---

## 5. Running it against a real network

Two options beyond the simulator, both documented in
`LEARNING_GUIDE.md` §20.5 and `datasets/README.md`.

**Trace replay (recommended).** Record per-link latency and utilisation from
your own monitoring into the documented JSONL or CSV schema, then:

```bash
curl -X POST http://localhost:8000/sim/source \
     -H "Content-Type: application/json" \
     -d '{"kind": "trace", "trace_path": "datasets/my_network.jsonl"}'
```

Or use the **Network Source** panel in the dashboard, which does the same thing.

Every algorithm then runs against your measurements. No privileges, no probing,
fully reproducible.

**Live probing (opt-in).** Measures RTT and loss to hosts you name, using the
unprivileged system `ping`. Disabled unless you set it:

```bash
LIVE_PROBE_ENABLED=1 uvicorn service.main:app --port 8000

curl -X POST http://localhost:8000/sim/source \
     -H "Content-Type: application/json" \
     -d '{"kind": "live", "targets": ["192.168.1.1", "8.8.8.8"]}'
```

Without `LIVE_PROBE_ENABLED=1` the endpoint returns **403** and says why.

It is read-only: no scanning, no host enumeration, no traffic injection, never
root, and it only contacts hosts you list explicitly. **Only point it at
networks you are authorised to measure.**

Its honest limitation: probing *n* hosts from one machine produces a **star**
topology — one path per destination — so there is nothing to route between. Live
mode visualises a real network; it cannot benchmark routing algorithms, and the
UI says so.

---

## 6. Environment variables

Everything has a working default. `.env` is optional.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | (unset) | Async PostgreSQL DSN. Unset means no persistence. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed origins |
| `TICK_SECONDS` | `1.0` | Simulator tick interval |
| `LOG_LEVEL` | `INFO` | Root log level |
| `LIVE_PROBE_ENABLED` | `0` | Set to `1` to allow live ICMP probing |
| `BENCHMARK_RESULTS_DIR` | `experiments/results` | Where the API reads results from |
| `POSTGRES_USER` / `_PASSWORD` / `_DB` | `routinguser` / `routingpass` / `routing_db` | Compose database |

The frontend reads `VITE_API_URL` at build time (`web/src/config.js`), defaulting
to same-origin so the nginx proxy in the container image works with no
configuration.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `env file .env not found` | An old checkout. `make up` creates it; or `cp .env.example .env`. |
| Port 5433 in use | Change the host-side port in `docker-compose.yml`. |
| `ModuleNotFoundError: core` | You ran from a subdirectory. Run from the repository root. |
| Health returns `degraded` | The simulator loop stopped. Check the logs — the exception is there with a stack trace. |
| Dashboard says "Waiting for backend" | The service is not on :8000, or `CORS_ORIGINS` does not include the dashboard's origin. |
| `torch` install is slow or fails | Use the CPU wheel index shown in §2.1. |
| Model-status banner shows a missing model | Expected on a checkout with checkpoints stripped. `make train`, or accept the fallback — it is labelled everywhere it is used. |
| `npm run lint` fails | Run `npm install` first; the flat-config ESLint packages are dev dependencies. |
| PowerShell blocks the venv activation | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 8. Where to go next

| If you want | Read |
|---|---|
| The results | [`docs/14_RESULTS_AND_FINDINGS.md`](14_RESULTS_AND_FINDINGS.md) |
| To learn the subject properly | [`LEARNING_GUIDE.md`](../LEARNING_GUIDE.md) |
| Traps that will cost you an afternoon | [`docs/15_GOTCHAS.md`](15_GOTCHAS.md) |
| What is still broken | [`docs/12_KNOWN_ISSUES.md`](12_KNOWN_ISSUES.md) |
