# Deployment

This is a research and benchmarking platform, not a product. It is designed for
local use, demos and CI. §5 says plainly what would have to change before it
went anywhere else.

---

## 1. Docker Compose (the intended path)

```bash
git clone https://github.com/sohxmm/AI-Based-Network-Packet-Routing-Optimization.git
cd AI-Based-Network-Packet-Routing-Optimization
make up
```

`make up` creates `.env` from `.env.example` if it is missing, then runs
`docker compose up --build`.

| Service | Container | Port | Notes |
|---|---|---|---|
| `db` | `routing-db` | 5433 -> 5432 | PostgreSQL 16, healthchecked |
| `backend` | `routing-backend` | 8000 | FastAPI, healthchecked, capped at 2 CPU / 4 GB |
| `frontend` | `routing-frontend` | 5173 -> 80 | nginx serving the built bundle |
| `pgadmin` | `routing-pgadmin` | 5050 | **`dev` profile only** |

```bash
make down                        # stop, remove volumes
make logs                        # follow backend logs
docker compose --profile dev up  # adds pgAdmin
```

### What was wrong before

**A fresh clone could not start.** Three services declared `env_file: .env`,
`.env` is gitignored, and nothing created it, so the very first command a
reviewer ran failed with `env file .env not found`. Every service now uses
`env_file: {path: .env, required: false}` and every variable has an inline
default.

**pgAdmin started by default with a default password.** A database admin UI
should not appear on a port because someone ran the default command. It is behind
`profiles: ["dev"]`.

**Nothing limited the backend.** An unbounded ACO benchmark could consume the
whole host. The service is capped at 2 CPUs and 4 GB.

### Image design

Both images are multi-stage.

**`service/Dockerfile`** — the build context is the repository **root**, not
`service/`, because the service imports `core`, `routing`, `ml` and
`experiments`. Those are the scientific core of the project and deliberately do
not live inside the web application.

- Builder stage installs with `gcc`/`g++`; the runtime image never sees a
  compiler.
- Runs as `appuser` (uid 10001), not root.
- `HEALTHCHECK` polls `/health` every 15 s with a 40 s start period.
- `CMD` pins `--workers 1`, with a comment saying why (see §2).

**`web/Dockerfile`** — Node builder, nginx runtime, `HEALTHCHECK` via `wget`.
`nginx.conf` proxies `/api` to the backend, so the bundle can use same-origin
URLs and needs no build-time configuration.

---

## 2. Single worker is a requirement, not a default

`service/state.py` holds a module-level `AppState` singleton containing the
network source, the router set (including ACO's **stateful** pheromone table) and
loaded torch weights.

A second Uvicorn worker owns a divergent copy of the network. The dashboard's
WebSocket would connect to one worker and its REST calls land on the other, so it
would flicker between two realities.

The singleton is correctly motivated — pheromone tables and model weights
genuinely must persist across requests, and rebuilding a router per request would
both discard learned state and reload weights from disk on every call. The fix
for horizontal scaling is therefore **shared state** (Redis, or the database),
not removing the singleton. It has not been done; see `12_KNOWN_ISSUES.md` §4.2.

---

## 3. Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r service/requirements.txt
uvicorn service.main:app --host 0.0.0.0 --port 8000 --workers 1
```

From the repository root. The service imports top-level packages, so running
from a subdirectory fails with `ModuleNotFoundError: core`.

The database is optional. Without `DATABASE_URL` the API starts, logs a warning,
and history and metrics endpoints degrade to live-state estimates.

Frontend:

```bash
cd web
npm ci
npm run build            # -> web/dist, static files for any web server
```

---

## 4. Verifying a deployment

| Check | Command | Expected |
|---|---|---|
| Containers | `docker compose ps` | all `running`, `healthy` |
| Liveness | `curl :8000/health` | `"status": "ok"` **and** a small `simulator_last_tick_age_s` |
| Models | `curl :8000/health/models` | `"status": "ok"`, every model `loaded_in_memory: true` |
| State | `curl :8000/network/state` | 25 nodes, 50 links |
| Routing | `POST :8000/network/route` | `"success": true`, `"is_fallback": false` |
| Dashboard | `http://localhost:5173` | "Live stream connected" in green |

**Do not accept `/health` responding as proof of health.** `status` is
`"degraded"` when the simulator loop has stopped ticking, and a container in
trouble reports exactly that. The CI `docker` job greps for `"status": "ok"`
specifically, for this reason.

**`/health/models` is the check that matters most.** It is what would have caught
the original bug where the loader looked for `rl_router_final.zip` while training
wrote `ppo_routing_agent.zip` — the file was never found, the router silently fell
back to a heuristic, and every published "RL" result was produced by five lines
of Python with no error anywhere.

---

## 5. What is missing for production

Stated as a list rather than a caveat, because these are decisions someone would
have to make, not risks to be aware of.

| Gap | What it needs |
|---|---|
| **No authentication** | Every endpoint and the WebSocket are open. This is the blocker. |
| **No TLS** | Terminate at a reverse proxy. |
| **Single worker** | Shared state in Redis or PostgreSQL before scaling out. |
| **No rate limiting** | The experiment endpoint has hard caps, but nothing limits request volume. |
| **Secrets in `.env`** | Fine locally; production wants a secret manager. `.env.example` ships `change-me-before-deploying` rather than a plausible password, because a plausible default is one people keep. |
| **No structured log shipping** | `service/logging_config.py` logs to stdout; nothing aggregates it. |
| **Retention is time-based only** | `prune_snapshots()` trims by age with no size cap. |
| **No backup policy** | The `postgres_data` volume is not backed up. |

None of these is hard. All of them are unaddressed, and none should be discovered
by whoever deploys it.

---

## 6. Resource usage

Measured on the 4-CPU machine this revision was developed on.

| Workload | Time | Peak memory |
|---|---|---|
| Service idle (1 Hz tick) | — | ~400 MB |
| Full training (4 models) | ~35 min | ~1.5 GB |
| Full benchmark (7 scenarios) | ~24 min | ~600 MB |
| Frontend build | ~20 s | ~500 MB |

No GPU at any point. The largest committed model is 66k parameters.

The one genuinely large cost is the initial `pip install`: the torch CPU wheel is
about 200 MB. Everything after that is small.

---

## 7. Deploying the *idea* rather than the code

If what you actually want is a learned router in a real control plane, this
repository is the wrong artifact to deploy — but `LEARNING_GUIDE.md` §20 is the
design. In brief:

- The learned router belongs in the **control plane**, never the data plane. It
  computes paths; it does not touch packets.
- `NetworkSource` is the seam: replace the simulator with gNMI/sFlow telemetry
  and BGP-LS topology, and everything above it (featurisation, inference, path
  selection) is unchanged.
- Ship it in **shadow mode** first — logging the path it would have chosen,
  programming nothing — for weeks, and watch the agreement rate with the
  incumbent.
- Then one traffic class in one region, behind guardrails: never exceed the
  classical path's cost by more than *x*%, never violate the class's QoS
  constraints, never flap faster than once per *T* seconds.
- Always keep the classical fallback, and **monitor its rate**. `fallback_rate`
  climbing is the earliest signal that the model has gone stale or the network
  has drifted outside its training distribution.
- A kill switch that reverts to pure classical routing, tested regularly rather
  than assumed to work.

The model is the easy part. The safety envelope is the deployment.
