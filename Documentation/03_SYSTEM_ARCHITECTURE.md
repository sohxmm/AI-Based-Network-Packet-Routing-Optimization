# System Architecture

## 1. High-Level Architecture

The system follows a three-tier architecture with a real-time data pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│  Vite Dev Server :5173                                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ TopologyGraph│  │ Congestion   │  │ RouteComparison        ││
│  │ (D3.js)      │  │ Heatmap      │  │ (Recharts)             ││
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────────┘│
│         │                 │                    │                │
│  ┌──────┴─────────────────┴────────────────────┴───────────────┐│
│  │            useNetworkStream (WebSocket)                      ││
│  │            useRouteRequest (REST API)                        ││
│  └──────┬──────────────────────────────────────────────────────┘│
└─────────┼───────────────────────────────────────────────────────┘
          │
    WebSocket :8000/ws/stream         REST API :8000/*
          │                                    │
┌─────────┴────────────────────────────────────┴──────────────────┐
│                        BACKEND (FastAPI)                         │
│  Uvicorn ASGI Server :8000                                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ WebSocket    │  │ REST Routes  │  │ Background Simulator   ││
│  │ Manager      │  │ (api/routes) │  │ Loop (1 step/sec)      ││
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────────┘│
│         │                 │                    │                │
│  ┌──────┴─────────────────┴────────────────────┴───────────────┐│
│  │                    Singleton AppState                        ││
│  │                ┌─────────────────────┐                      ││
│  │                │  NetworkSimulator   │                      ││
│  │                │  (NetworkX Graph)   │                      ││
│  │                └─────────────────────┘                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ Routing      │  │ ML Models    │  │ Database Layer         ││
│  │ Algorithms   │  │ LSTM / PPO   │  │ (SQLAlchemy Async)     ││
│  └──────────────┘  └──────────────┘  └────────┬───────────────┘│
└────────────────────────────────────────────────┼────────────────┘
                                                 │
┌────────────────────────────────────────────────┴────────────────┐
│                     DATABASE (PostgreSQL 16)                     │
│  Docker Container :5433                                         │
│                                                                 │
│  Tables: routing_events, network_snapshots,                     │
│          algorithm_metrics, packet_logs                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

### 2.1 Real-Time Simulation Loop

```
NetworkSimulator.step()      →  Every 1 second (background asyncio task)
       │
       ├── Updates all link attributes (utilization, queue, packet_loss)
       ├── Manages congestion burst events
       ├── Returns NetworkState dataclass
       │
       ▼
handle_simulator_step()
       │
       ├── Broadcasts state_update to all WebSocket clients
       └── Saves NetworkSnapshot to PostgreSQL
```

### 2.2 Route Request Flow

```
Frontend: User selects source/destination → POST /network/route/compare
       │
       ▼
Backend: compare_routes()
       │
       ├── Gets current NetworkState from simulator
       ├── Runs all 4 algorithms (Dijkstra, Bellman-Ford, ACO, RL)
       ├── Saves each RoutingEvent to database
       ├── Broadcasts routing_event over WebSocket
       └── Returns comparison results as JSON
       │
       ▼
Frontend: Updates RouteComparison table + bar chart
          Highlights best path on TopologyGraph
```

### 2.3 Congestion Forecast Flow

```
GET /network/congestion-forecast?steps=3
       │
       ▼
CongestionPredictor.predict_next()
       │
       ├── Maintains rolling window of utilization snapshots
       ├── If LSTM model loaded and window >= seq_len:
       │       Uses LSTM to predict next utilization vector
       └── Otherwise:
               Returns last known snapshot (safe fallback)
       │
       ▼
Returns per-link predicted utilization for N steps ahead
```

---

## 3. Component Responsibilities

### 3.1 Backend Components

| Module                    | Responsibility                                           |
|---------------------------|----------------------------------------------------------|
| `main.py`                 | FastAPI app creation, lifespan management, CORS config    |
| `api/routes.py`           | All REST endpoint handlers and algorithm dispatch         |
| `api/state.py`            | Singleton `AppState` holding the `NetworkSimulator`       |
| `api/websocket.py`        | WebSocket connection manager and `/ws/stream` endpoint    |
| `simulator/network_sim.py`| Core simulation engine (NetworkX graph with traffic)      |
| `simulator/data_models.py`| Data classes: `LinkState`, `NetworkState`, `RoutingDecision` |
| `router/dijkstra.py`      | Dijkstra's shortest path with congestion-aware weights    |
| `router/bellman_ford.py`  | Bellman-Ford with negative cycle detection                |
| `router/aco.py`           | Ant Colony Optimization with pheromone management         |
| `router/rl_agent.py`      | PPO-based RL router with heuristic fallback               |
| `ml/congestion_lstm.py`   | LSTM model for utilization forecasting                    |
| `ml/rl_environment.py`    | Gymnasium environment for PPO training                    |
| `ml/train_rl.py`          | PPO training pipeline with callbacks                      |
| `db/database.py`          | Async SQLAlchemy engine and session factory                |
| `db/models.py`            | ORM table definitions (4 tables)                          |
| `db/init_db.py`           | Standalone database initialization script                 |

### 3.2 Frontend Components

| Component               | Responsibility                                        |
|--------------------------|------------------------------------------------------|
| `App.jsx`                | Root layout, theme management, state coordination     |
| `TopologyGraph.jsx`      | D3 force-directed graph with live color updates       |
| `CongestionHeatmap.jsx`  | Recharts horizontal bar chart of link utilization     |
| `RouteComparison.jsx`    | Algorithm comparison form, table, and latency chart   |
| `ControlPanel.jsx`       | Simulator step/reset and link failure injection       |
| `MetricsPanel.jsx`       | Key metric cards (latency, delivery, congestion)      |
| `LeftPanel.jsx`          | Top queue sizes leaderboard                           |
| `RightPanel.jsx`         | Global topology statistics                            |
| `useNetworkStream.js`    | WebSocket hook with auto-reconnect                    |
| `useRouteRequest.js`     | REST API hook for routes and simulator actions        |

---

## 4. Key Design Decisions

### 4.1 Singleton Simulator

The `NetworkSimulator` is instantiated once in `api/state.py` as a module-level singleton. All API endpoints and the background loop share this same instance, ensuring consistent state across requests.

### 4.2 Background Simulation Loop

The simulator advances every second via an `asyncio.create_task()` in the FastAPI lifespan handler. This runs independently of API requests, providing a continuously evolving network state that the dashboard visualizes in real time.

### 4.3 Congestion-Aware Edge Weights

All routing algorithms use the same weight formula:

```
weight = base_latency × (1 + 4 × utilization²)
```

This quadratic penalty makes high-utilization links exponentially more expensive, encouraging algorithms to avoid congested paths.

### 4.4 Two-Phase D3 Rendering

The `TopologyGraph` component uses a two-phase rendering strategy:
- **Phase 1**: Builds the D3 force simulation once when the node topology changes
- **Phase 2**: Updates only link colors and widths on every WebSocket tick

This prevents the graph from "dancing" on every state update while keeping visual feedback responsive.

### 4.5 CSS Variable Theming

The dashboard supports 10 color themes using CSS custom properties. Theme switching is handled entirely client-side by swapping CSS classes on `<html>`, with no backend involvement.

### 4.6 RL Fallback Strategy

The `RLRouter` gracefully falls back to a congestion-aware Dijkstra heuristic when the PPO model is not loaded. This ensures the API never fails cold and the system works out of the box before any training.

---

## 5. Communication Protocols

### 5.1 WebSocket Messages

All WebSocket messages follow this envelope format:

```json
{
  "type": "state_update" | "routing_event",
  "payload": { ... }
}
```

**`state_update`** — Sent every second with the full network state:
```json
{
  "type": "state_update",
  "payload": {
    "nodes": ["R1", "R2", ...],
    "links": [{ "source": "R1", "target": "R2", "utilization": 0.42, ... }],
    "timestamp": 1720000000.0,
    "step_count": 150
  }
}
```

**`routing_event`** — Sent whenever a route is calculated:
```json
{
  "type": "routing_event",
  "payload": {
    "source": "R1",
    "destination": "R5",
    "path": ["R1", "R3", "R5"],
    "algorithm": "dijkstra",
    "total_latency": 34.7,
    "avg_utilization": 0.35,
    "success": true
  }
}
```

### 5.2 REST API

- All REST endpoints return JSON
- POST bodies use JSON (`Content-Type: application/json`)
- Error responses use FastAPI's standard `{"detail": "..."}` format
- CORS is configured to allow `http://localhost:5173` and `http://127.0.0.1:5173`
