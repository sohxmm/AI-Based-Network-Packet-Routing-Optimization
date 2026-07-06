# Known Issues & Limitations

---

## 1. Backend Issues

### 1.1 Database Connection Failures Are Silently Logged

**Impact**: Low  
**Location**: `backend/api/routes.py` — `save_routing_event()`, `handle_simulator_step()`

If the PostgreSQL database is unavailable, routing events and network snapshots fail to save. The error is printed to console (`[DB] Error saving...`) but the API continues operating normally. This means the dashboard works without a database, but no historical data is recorded.

**Workaround**: Ensure Docker containers are running before starting the backend.

---

### 1.2 Singleton Simulator Is Not Thread-Safe

**Impact**: Low (single-worker mode)  
**Location**: `backend/api/state.py`

The `NetworkSimulator` is a module-level singleton shared across all requests. Running Uvicorn with multiple workers (`--workers > 1`) would create separate simulator instances per process, leading to inconsistent state.

**Workaround**: Run with a single worker (the default). Multi-worker deployment requires shared state via Redis or a database.

---

### 1.3 Network Snapshot Table Grows Unbounded

**Impact**: Medium (over time)  
**Location**: `backend/api/routes.py` — `handle_simulator_step()`

A new `NetworkSnapshot` row is inserted every second (one per simulator tick). After 24 hours of continuous operation, this creates ~86,400 rows. There is no cleanup or retention policy.

**Workaround**: Manually truncate the `network_snapshots` table, or add a retention policy in Phase 2.

---

### 1.4 `datetime.utcnow()` Deprecation Warning

**Impact**: Low  
**Location**: `backend/db/models.py`

`datetime.utcnow()` is deprecated in Python 3.12+. The ORM models use it as a column default. Should be migrated to `datetime.now(timezone.utc)`.

---

### 1.5 RL Router Creates New Instance Per Request

**Impact**: Low (performance)  
**Location**: `backend/api/routes.py` — `_run_algorithm()`

Both `AntColonyRouter()` and `RLRouter()` are instantiated fresh on every API call. The RL router re-loads the PPO model each time. Consider caching these as singletons.

---

## 2. Frontend Issues

### 2.1 Hardcoded Backend URL

**Impact**: Medium  
**Location**: `frontend/src/hooks/useRouteRequest.js`

The REST API base URL is hardcoded to `http://localhost:8000`. The WebSocket URL in `useNetworkStream.js` dynamically constructs the host but hardcodes port 8000. This must be updated for any non-localhost deployment.

**Workaround**: Use environment variables via Vite's `import.meta.env` system.

---

### 2.2 RightPanel References Undefined `capacity` Field

**Impact**: Low (displays NaN/0)  
**Location**: `frontend/src/components/RightPanel.jsx`

The component accesses `link.capacity` which is not a field on the `LinkState` data model. It shows 0 or NaN for "Avg Link Capacity". Should reference `link.bandwidth` instead.

---

### 2.3 PacketAnimator Is a Placeholder

**Impact**: None (unused)  
**Location**: `frontend/src/components/PacketAnimator.jsx`

This component file exists but contains no implementation. It was planned for animated packet flow visualization but was not completed in Phase 1.

---

### 2.4 No Error Recovery for Failed Route Comparisons

**Impact**: Low  
**Location**: `frontend/src/components/RouteComparison.jsx`

If all algorithms fail (e.g., source equals destination), the API returns a 400 error. The error message is shown briefly but the previous comparison results remain in the table, which can be confusing.

---

## 3. ML / AI Issues

### 3.1 LSTM Model May Not Be Loaded on Fresh Clone

**Impact**: Low  
**Location**: `backend/ml/congestion_lstm.py`

The LSTM model file (`congestion_lstm.pt`) is gitignored. On a fresh clone, the congestion forecast endpoint falls back to returning the last known snapshot. This is by design but should be documented for new developers.

**Workaround**: Run `python -m ml.congestion_lstm` from the `backend/` directory to train.

---

### 3.2 RL Agent Performance Plateaus

**Impact**: Medium  
**Location**: `backend/ml/train_rl.py`

The PPO agent's mean reward improved from -77 to -61 (+21%) over 500k steps, with the best evaluation reward at -45.81. Performance improvements were modest, suggesting:
- The observation space may not capture enough routing context
- The reward function could be further tuned
- Longer training may yield diminishing returns without architectural changes

---

### 3.3 Autoregressive Forecast Drift

**Impact**: Low  
**Location**: `backend/api/routes.py` — `get_congestion_forecast()`

Multi-step forecasts feed each prediction back as input for the next step. Over several steps, prediction errors compound, causing drift from reality. Forecasts beyond 3–5 steps should be treated with skepticism.

---

## 4. Infrastructure Issues

### 4.1 No Database Migration System

**Impact**: Medium (for schema changes)

Tables are created via `CREATE TABLE IF NOT EXISTS` on startup. There is no versioned migration system (like Alembic). Schema changes require manual table drops or migration scripts.

**Recommendation**: Add Alembic for Phase 2.

---

### 4.2 No CI/CD Pipeline

**Impact**: Medium  

There are no automated tests, linting, or deployment pipelines. All testing is manual.

---

### 4.3 No Authentication or Authorization

**Impact**: High (for production)

All API endpoints and WebSocket connections are publicly accessible without authentication. This is acceptable for local development but must be addressed before any public deployment.
