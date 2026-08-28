# Database Schema

PostgreSQL 16 via SQLAlchemy 2.0 async (`asyncpg`). Schema changes go through
**Alembic**.

**The database is optional.** Without `DATABASE_URL` the service starts, logs a
warning, and the history and metrics endpoints degrade to live-state estimates.
Nothing in the routing or benchmarking path requires it.

---

## 1. Connection

| Property | Value |
|---|---|
| Driver | `postgresql+asyncpg` |
| Port | 5433 on the host, 5432 in the container |
| Database | `routing_db` |
| Configured by | `DATABASE_URL` |

The host-side port is 5433 so it does not collide with a locally installed
PostgreSQL. Inside the Compose network the backend connects to `db:5432`
directly.

---

## 2. Migrations

Schema is applied by Alembic at startup (`service/main.py:_migrate_or_create`):

```python
config = Config("service/alembic.ini")
await asyncio.to_thread(command.upgrade, config, "head")
```

`create_all` remains only as a **fallback** for tests and throwaway databases,
and the log says so when it is used:

> Alembic upgrade failed (...); falling back to create_all. Existing tables will
> NOT be altered.

That distinction is the whole point. `create_all` can only CREATE, never ALTER,
so on an existing deployment adding a column silently does nothing — the app
starts, the query fails at runtime, and the startup logs are clean. The previous
version used `create_all` as its only mechanism and described it as the migration
path.

```bash
cd service
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
alembic downgrade -1
```

---

## 3. Tables

Three. There were four; see §5.

### 3.1 `routing_events`

Every routing decision made through the API.

| Column | Type | Index | Notes |
|---|---|---|---|
| `id` | String | PK | |
| `timestamp` | DateTime(tz) | yes | UTC-aware |
| `source` | String | yes | |
| `destination` | String | yes | |
| `algorithm` | String | yes | |
| `traffic_class` | String | yes | One of the five QoS classes |
| `path` | JSON | | The full node sequence |
| `total_latency` | Float | | |
| `avg_utilization` | Float | | |
| `success` | Boolean | | |
| `is_fallback` | Boolean | | **Was this a model decision or the heuristic?** |
| `qos_feasible` | Boolean | | Did the path satisfy the class's hard constraints? |
| `step_count` | Integer | yes | Simulator step |

Three columns are newer than the rest and each exists for a reason:

**`is_fallback`.** Without it, a stored routing event cannot tell you whether the
named algorithm actually ran. Historical analysis over a table where "gnn" might
mean the GNN or might mean a five-line heuristic is worthless.

**`qos_feasible`.** Latency alone cannot tell you whether a decision was
*correct*. A path 3 ms faster that violates the emergency class's loss budget is
not a better answer.

**`avg_utilization`.** This one was a genuine bug. The metrics endpoint computed
`congestion_events` from `getattr(event, "avg_utilization", 0.0)` — but the
column did not exist, so the `getattr` default applied on every row and the
metric was permanently, silently zero. A `getattr` with a default against your
own ORM model is a bug waiting for someone to add the attribute.

### 3.2 `network_snapshots`

Periodic captures of the whole network state.

| Column | Type | Index | Notes |
|---|---|---|---|
| `id` | String | PK | |
| `timestamp` | DateTime(tz) | yes | |
| `state_json` | JSON | | Full serialised `NetworkState` |
| `avg_utilization` | Float | | Denormalised for cheap queries |
| `congested_links` | Integer | | Denormalised |
| `step_count` | Integer | yes | |

**Write rate and retention.** One snapshot per **ten** ticks
(`SNAPSHOT_EVERY_N_STEPS = 10`), and `prune_snapshots()` keeps the newest
`MAX_SNAPSHOTS = 10_000` rows, running every 10 minutes. That is roughly 28 hours
of history at the default 1 Hz tick.

Previously a snapshot was written **every tick** with no retention at all: about
860 MB a day, growing without bound, written from inside the simulator loop.

The write is fire-and-forget (`asyncio.create_task`), so a slow or failed insert
cannot stall the tick. Failures are logged, never fatal.

### 3.3 `algorithm_metrics`

Aggregated benchmark results, one row per algorithm per scenario window.

| Column | Type | Index |
|---|---|---|
| `id` | String | PK |
| `timestamp` | DateTime(tz) | yes |
| `algorithm` | String | yes |
| `scenario` | String | yes |
| `window_start_step` | Integer | |
| `window_end_step` | Integer | |
| `avg_latency` | Float | |
| `success_rate` | Float | |
| `num_decisions` | Integer | |

This table was **dead code that looked live**: the benchmark built these rows and
then discarded them behind a commented-out commit. It is now written when the
runner is invoked with `--persist`, and read by
`GET /metrics/benchmark-history`.

---

## 4. Timestamps

```python
def _utcnow() -> datetime:
    return datetime.now(UTC)
```

Every timestamp column is `DateTime(timezone=True)` with this default.
`datetime.utcnow()` is deprecated in Python 3.12+ **and** returns a naive
datetime, which produces silent comparison bugs the moment anything
timezone-aware enters the same query.

---

## 5. What was removed

`packet_logs` is gone. It was declared, indexed, documented with a full column
table, and **never written to by anything** — zero references in the entire
codebase. Infrastructure for its own sake reads as padding, so the refactor
dropped it.

`scripts/verify_claims.py` now fails CI if any document describes a table that
is not declared in `service/db/models.py`, because three separate documents went
on describing `packet_logs` after it stopped existing. A table is a thing with a
name; that is exactly the kind of claim a script can check and a reader cannot.

---

## 6. Useful queries

Fallback rate per algorithm — the first thing to check on any stored result:

```sql
SELECT algorithm,
       COUNT(*)                                             AS decisions,
       AVG(CASE WHEN is_fallback THEN 1.0 ELSE 0.0 END)     AS fallback_rate,
       AVG(total_latency) FILTER (WHERE success)            AS avg_latency
FROM routing_events
GROUP BY algorithm
ORDER BY fallback_rate DESC;
```

QoS satisfaction per traffic class:

```sql
SELECT traffic_class,
       algorithm,
       AVG(CASE WHEN qos_feasible THEN 1.0 ELSE 0.0 END) AS satisfaction
FROM routing_events
WHERE success
GROUP BY traffic_class, algorithm
ORDER BY traffic_class, satisfaction DESC;
```

Congestion over time:

```sql
SELECT date_trunc('minute', timestamp) AS minute,
       AVG(avg_utilization)            AS mean_util,
       MAX(congested_links)            AS peak_congested
FROM network_snapshots
GROUP BY minute
ORDER BY minute DESC
LIMIT 60;
```

---

## 7. Inspecting it

```bash
docker compose --profile dev up          # pgAdmin on :5050
psql -h localhost -p 5433 -U routinguser -d routing_db
```

pgAdmin is behind the `dev` profile deliberately: a database admin UI with a
default password should not start because someone ran the default command.
