# Configuration & Environment Variables

---

## 1. Environment Files

The project uses two `.env` files:

| File              | Location           | Purpose                               |
|-------------------|--------------------|---------------------------------------|
| `.env`            | Project root       | Docker Compose + shared variables     |
| `backend/.env`    | Backend directory  | Backend-specific overrides            |

Both are gitignored. A `.env.example` template is provided in the project root.

---

## 2. Environment Variables

### 2.1 Database Configuration

| Variable              | Default Value                                                   | Description                              |
|-----------------------|-----------------------------------------------------------------|------------------------------------------|
| `DATABASE_URL`        | `postgresql+asyncpg://routinguser:routingpass@localhost:5433/routing_db` | Full async connection string    |
| `POSTGRES_USER`       | `routinguser`                                                   | PostgreSQL username                      |
| `POSTGRES_PASSWORD`   | `routingpass`                                                   | PostgreSQL password                      |
| `POSTGRES_DB`         | `routing_db`                                                    | PostgreSQL database name                 |

### 2.2 pgAdmin Configuration

| Variable                   | Default Value        | Description                     |
|----------------------------|----------------------|---------------------------------|
| `PGADMIN_DEFAULT_EMAIL`    | `admin@example.com`  | pgAdmin login email             |
| `PGADMIN_DEFAULT_PASSWORD` | `admin`              | pgAdmin login password          |

---

## 3. Port Configuration

| Service           | Default Port | Configurable In              |
|-------------------|--------------|-----------------------------|
| FastAPI backend   | 8000         | `uvicorn` CLI args           |
| Vite dev server   | 5173         | `vite.config.js` or CLI      |
| PostgreSQL        | 5433         | `docker-compose.yml`         |
| pgAdmin           | 5050         | `docker-compose.yml`         |

> **Note**: PostgreSQL uses port **5433** (not the default 5432) to avoid conflicts with any locally installed PostgreSQL.

---

## 4. CORS Configuration

Defined in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

If the frontend runs on a different port or domain, update the `allow_origins` list.

---

## 5. Simulator Configuration

Hardcoded defaults in `simulator/network_sim.py`:

| Parameter              | Default | Description                              |
|------------------------|---------|------------------------------------------|
| `num_nodes`            | 10      | Number of router nodes (R1–R10)          |
| `seed`                 | 42      | Random seed for reproducibility          |
| Congestion burst every | 50 steps| Steps between random congestion events   |
| Congestion burst length| 10 steps| Duration of each congestion burst        |
| Base latency range     | 5–25 ms | Random base latency per link             |
| Bandwidth options      | 100, 500, 1000 | Random bandwidth per link         |
| Initial utilization    | 0.1–0.5 | Random initial link utilization          |

---

## 6. ML Configuration

### LSTM Predictor (`ml/congestion_lstm.py`)

| Parameter        | Default | Location                    |
|------------------|---------|-----------------------------|
| `seq_len`        | 10      | Constructor                 |
| `hidden_size`    | 64      | `CongestionLSTM.__init__`   |
| `num_layers`     | 2       | `CongestionLSTM.__init__`   |
| `dropout`        | 0.2     | `CongestionLSTM.__init__`   |
| Training steps   | 2,000   | `collect_data()` call       |
| Training epochs  | 30      | `train()` call              |
| Learning rate    | 0.001   | `train()` call              |

### PPO Agent (`ml/train_rl.py`)

| Parameter          | Default     | Location                   |
|--------------------|-------------|----------------------------|
| `total_timesteps`  | 500,000     | `main()` function          |
| `learning_rate`    | 3×10⁻⁴      | PPO constructor            |
| `n_steps`          | 2,048       | PPO constructor            |
| `batch_size`       | 64          | PPO constructor            |
| `n_epochs`         | 10          | PPO constructor            |
| `gamma`            | 0.99        | PPO constructor            |
| `checkpoint_freq`  | 50,000      | CheckpointCallback         |
| `eval_freq`        | 25,000      | EvalCallback               |

### GNN Agent (`ml/train_gnn.py`)

| Parameter          | Default     | Location                   |
|--------------------|-------------|----------------------------|
| `num_samples`      | 1500        | `generate_dataset()`       |
| Loss components    | Latency(0.5), Max Util(0.3), Imbalance(0.2)| `get_path_cost()`     |

### MARL Agent (`ml/train_multi_agent.py`)

| Parameter          | Default     | Location                   |
|--------------------|-------------|----------------------------|
| `N_ROUNDS`         | 3           | Module constant            |
| `K_STEPS`          | 15,000      | Module constant            |
| `NUM_NODES`        | 25          | Module constant            |

### RL Environment (`ml/rl_environment.py`)

| Parameter              | Default | Location                   |
|------------------------|---------|----------------------------|
| `K_PATHS`              | 5       | Module constant            |
| `EPISODE_STEPS`        | 200     | Module constant            |
| `W_LATENCY`            | 0.5     | Class attribute            |
| `W_UTIL`               | 0.3     | Class attribute            |
| `W_LOSS`               | 0.2     | Class attribute            |
| `CONGESTION_THRESHOLD` | 0.85    | Class attribute            |
| `CONGESTION_PENALTY`   | 2.0     | Class attribute            |

---

## 7. Frontend Configuration

### Vite (`frontend/vite.config.js`)

```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true
      }
    }
  }
});
```

### API Base URL

Defined in `frontend/src/hooks/useRouteRequest.js`:

```javascript
const API_BASE_URL = "http://localhost:8000";
```

### WebSocket URL

Computed in `frontend/src/hooks/useNetworkStream.js`:

```javascript
const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
return `${protocol}//${window.location.hostname}:8000/ws/stream`;
```

---

## 8. Docker Compose Configuration

See `docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:16
    container_name: routing-db
    ports: ["5433:5432"]        # Host:Container
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready ..."]
      interval: 10s

  pgadmin:
    image: dpage/pgadmin4:8
    container_name: routing-pgadmin
    ports: ["5050:80"]          # Host:Container
    depends_on:
      db:
        condition: service_healthy

volumes:
  postgres_data:                # Persistent data volume
```
