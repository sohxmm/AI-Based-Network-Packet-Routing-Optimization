# Database Schema

The application uses **PostgreSQL 16** via SQLAlchemy 2.0 with async support (`asyncpg` driver). Tables are automatically created on backend startup via `init_db()`.

---

## 1. Connection Details

| Property     | Value                                                        |
|-------------|--------------------------------------------------------------|
| Engine      | `postgresql+asyncpg`                                          |
| Host        | `localhost`                                                   |
| Port        | `5433` (mapped from container's 5432)                         |
| Database    | `routing_db`                                                  |
| User        | `routinguser`                                                 |
| Password    | `routingpass`                                                 |
| Full URL    | `postgresql+asyncpg://routinguser:routingpass@localhost:5433/routing_db` |

Configuration is read from the `DATABASE_URL` environment variable, with the above as the default fallback.

---

## 2. Tables

### 2.1 `routing_events`

Logs every routing decision made through the API. This is the primary table for performance analysis.

| Column        | Type       | Constraints        | Description                                   |
|---------------|------------|--------------------|-----------------------------------------------|
| `id`          | `String`   | **Primary Key**    | UUID v4 string                                |
| `timestamp`   | `DateTime` | Indexed, default=now | When the decision was made                  |
| `source`      | `String`   | Indexed            | Source router ID (e.g., "R1")                 |
| `destination` | `String`   | Indexed            | Destination router ID (e.g., "R10")           |
| `algorithm`   | `String`   | Indexed            | Algorithm used: `dijkstra` \| `bellman_ford` \| `aco` \| `rl` |
| `path`        | `JSON`     |                    | Ordered list of router IDs: `["R1","R3","R5"]`|
| `total_latency`| `Float`   |                    | Total path latency in ms (NULL if inf/NaN)    |
| `success`     | `Boolean`  |                    | Whether a valid path was found                |
| `step_count`  | `Integer`  | Indexed            | Simulator step when decision was made         |

**Usage**: Queried by `GET /metrics/summary` and `GET /metrics/history`.

---

### 2.2 `network_snapshots`

Complete network state captured on every simulator tick (once per second).

| Column           | Type       | Constraints        | Description                                |
|------------------|------------|--------------------|---------------------------------------------|
| `id`             | `String`   | **Primary Key**    | UUID v4 string                              |
| `timestamp`      | `DateTime` | Indexed, default=now | When the snapshot was taken               |
| `state_json`     | `JSON`     |                    | Full serialized `NetworkState` as JSON      |
| `avg_utilization` | `Float`   |                    | Mean link utilization across all links (0–1)|
| `congested_links`| `Integer`  |                    | Count of links with utilization ≥ 0.7       |
| `step_count`     | `Integer`  | Indexed            | Simulator step number                       |

**Usage**: Saved automatically by `handle_simulator_step()` on every background tick.

---

### 2.3 `algorithm_metrics`

Aggregate performance metrics per algorithm over a time window. Designed for dashboard analytics.

| Column             | Type       | Constraints        | Description                                |
|--------------------|------------|--------------------|--------------------------------------------|
| `id`               | `String`   | **Primary Key**    | UUID v4 string                             |
| `timestamp`        | `DateTime` | Indexed, default=now | When the metric was computed             |
| `algorithm`        | `String`   | Indexed            | Algorithm name                             |
| `window_start_step`| `Integer`  |                    | First step of the aggregation window       |
| `window_end_step`  | `Integer`  |                    | Last step of the aggregation window        |
| `avg_latency`      | `Float`    |                    | Average latency across decisions in window |
| `success_rate`     | `Float`    |                    | % of decisions that found a valid path     |
| `num_decisions`    | `Integer`  |                    | Number of decisions in the window          |

**Status**: Table schema defined; full aggregation pipeline planned for Phase 2.

---

### 2.4 `packet_logs`

Log of simulated packet transmissions for detailed packet-level analysis.

| Column         | Type       | Constraints        | Description                               |
|----------------|------------|--------------------|--------------------------------------------|
| `id`           | `String`   | **Primary Key**    | UUID v4 string                             |
| `timestamp`    | `DateTime` | Indexed, default=now | When the packet was transmitted          |
| `source`       | `String`   | Indexed            | Source router ID                           |
| `destination`  | `String`   | Indexed            | Destination router ID                      |
| `path`         | `JSON`     |                    | Route taken: `["R1","R3","R10"]`           |
| `success`      | `Boolean`  |                    | Whether packet reached its destination     |
| `arrival_time` | `Float`    |                    | Simulated arrival time                     |

**Status**: Table schema defined; packet simulation pipeline planned for Phase 2.

---

## 3. Entity Relationship Diagram

```
┌─────────────────────┐       ┌──────────────────────┐
│   routing_events    │       │  network_snapshots   │
├─────────────────────┤       ├──────────────────────┤
│ PK  id (UUID)       │       │ PK  id (UUID)        │
│     timestamp       │       │     timestamp         │
│ IDX source          │       │     state_json (JSON) │
│ IDX destination     │       │     avg_utilization   │
│ IDX algorithm       │       │     congested_links   │
│     path (JSON)     │       │ IDX step_count        │
│     total_latency   │       └──────────────────────┘
│     success         │
│ IDX step_count      │       ┌──────────────────────┐
└─────────────────────┘       │  algorithm_metrics   │
                              ├──────────────────────┤
┌─────────────────────┐       │ PK  id (UUID)        │
│    packet_logs      │       │     timestamp         │
├─────────────────────┤       │ IDX algorithm         │
│ PK  id (UUID)       │       │     window_start_step │
│     timestamp       │       │     window_end_step   │
│ IDX source          │       │     avg_latency       │
│ IDX destination     │       │     success_rate      │
│     path (JSON)     │       │     num_decisions     │
│     success         │       └──────────────────────┘
│     arrival_time    │
└─────────────────────┘
```

> **Note**: These tables are independent — there are no foreign key relationships between them. Each table operates as a standalone event/snapshot log.

---

## 4. Database Access Patterns

### Session Management

```python
# FastAPI dependency injection
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Direct usage (for background tasks)
async with AsyncSessionLocal() as session:
    session.add(entity)
    await session.commit()
```

### Key Queries Used

| Endpoint                     | Query Description                                    |
|-------------------------------|-----------------------------------------------------|
| `GET /metrics/summary`        | Last 100 routing events, optionally filtered by algorithm |
| `GET /metrics/history`        | Last N routing events ordered by timestamp DESC      |
| `POST /network/route`         | INSERT one routing event per decision                |
| `POST /network/route/compare` | INSERT one routing event per algorithm               |
| Background simulator loop     | INSERT one network snapshot per tick                  |

---

## 5. Database Management

### pgAdmin Access

| Property | Value                    |
|----------|--------------------------|
| URL      | `http://localhost:5050`   |
| Email    | `admin@example.com`      |
| Password | `admin`                  |

### Useful Docker Commands

```bash
# Start containers
docker compose up -d

# Stop containers
docker compose down

# View logs
docker compose logs db

# Reset database (deletes all data)
docker compose down -v
docker compose up -d

# Connect via psql
docker exec -it routing-db psql -U routinguser -d routing_db
```

### Manual Table Initialization

```bash
cd backend
python -m db.init_db
```
