# API Reference

All API endpoints are served by the FastAPI backend on `http://localhost:8000`.

---

## Table of Contents

- [Health Check](#1-health-check)
- [Network State](#2-network-state)
- [Routing](#3-routing)
- [Simulator Controls](#4-simulator-controls)
- [Metrics & Analytics](#5-metrics--analytics)
- [Congestion Forecast](#6-congestion-forecast)
- [WebSocket Stream](#7-websocket-stream)

---

## 1. Health Check

### `GET /health`

Returns a minimal health response for uptime checks.

**Response:**
```json
{
  "status": "ok"
}
```

---

## 2. Network State

### `GET /network/state`

Returns the current simulator state without advancing the simulation.

**Response:**
```json
{
  "nodes": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"],
  "links": [
    {
      "source": "R1",
      "target": "R2",
      "base_latency": 12.0,
      "bandwidth": 500,
      "utilization": 0.42,
      "queue_size": 42,
      "packet_loss_rate": 0.0
    }
  ],
  "timestamp": 1720000000.0,
  "step_count": 150
}
```

### `GET /network/topology`

Returns static topology data for graph visualizations. Includes only node IDs, link endpoints, and static link attributes.

**Response:**
```json
{
  "nodes": [
    { "id": "R1", "label": "R1" }
  ],
  "links": [
    {
      "source": "R1",
      "target": "R2",
      "base_latency": 12.0,
      "bandwidth": 500
    }
  ],
  "step_count": 150
}
```

---

## 3. Routing

### `POST /network/route`

Calculate a route using a single algorithm.

**Request Body:**
```json
{
  "source": "R1",
  "destination": "R5",
  "algorithm": "dijkstra"
}
```

| Field       | Type     | Required | Default      | Description                             |
|-------------|----------|----------|--------------|-----------------------------------------|
| source      | string   | ✅       | —            | Source router ID (e.g., "R1")           |
| destination | string   | ✅       | —            | Destination router ID (e.g., "R5")      |
| algorithm   | string   | ❌       | `"dijkstra"` | One of: `dijkstra`, `bellman_ford`, `aco`, `rl` |

**Response (success):**
```json
{
  "source": "R1",
  "destination": "R5",
  "path": ["R1", "R3", "R5"],
  "algorithm": "dijkstra",
  "total_latency": 34.7,
  "avg_utilization": 0.35,
  "success": true
}
```

**Response (no path):** `400 Bad Request`
```json
{
  "detail": "No path exists between R1 and R5 using dijkstra."
}
```

---

### `POST /network/route/compare`

Compare routing decisions across multiple algorithms simultaneously.

**Request Body:**
```json
{
  "source": "R1",
  "destination": "R5",
  "algorithms": ["dijkstra", "bellman_ford", "aco", "rl"]
}
```

| Field       | Type          | Required | Default                     | Description                             |
|-------------|---------------|----------|-----------------------------|-----------------------------------------|
| source      | string        | ✅       | —                           | Source router ID                        |
| destination | string        | ✅       | —                           | Destination router ID                   |
| algorithms  | string[]      | ❌       | All 4 algorithms            | Subset of algorithms to compare         |

**Response:**
```json
{
  "source": "R1",
  "destination": "R5",
  "step_count": 150,
  "results": [
    {
      "source": "R1",
      "destination": "R5",
      "path": ["R1", "R3", "R5"],
      "algorithm": "dijkstra",
      "total_latency": 34.7,
      "avg_utilization": 0.35,
      "success": true
    },
    {
      "source": "R1",
      "destination": "R5",
      "path": ["R1", "R4", "R7", "R5"],
      "algorithm": "aco",
      "total_latency": 52.1,
      "avg_utilization": 0.28,
      "success": true
    }
  ]
}
```

---

## 4. Simulator Controls

### `POST /sim/step`

Advance the simulator by one step and return the new state.

**Request Body:** None

**Response:** Same format as `GET /network/state`

---

### `POST /sim/reset`

Reset the simulator to its initial state (step 0, seed 42).

**Request Body:** None

**Response:** Same format as `GET /network/state`

---

### `POST /sim/inject-failure`

Temporarily remove a link to simulate a network failure.

**Request Body:**
```json
{
  "source": "R1",
  "target": "R2"
}
```

**Response:** Same format as `GET /network/state` (post-failure)

**Error:** `400 Bad Request` if the link does not exist or is already failed.

---

### `POST /sim/restore-link`

Restore a previously failed link.

**Request Body:**
```json
{
  "source": "R1",
  "target": "R2"
}
```

**Response:** Same format as `GET /network/state` (post-restore)

**Error:** `400 Bad Request` if the link is not currently failed.

---

## 5. Metrics & Analytics

### `GET /metrics/summary`

Returns aggregated performance metrics from the last 100 routing decisions in the database.

**Query Parameters:**

| Param     | Type   | Required | Description                          |
|-----------|--------|----------|--------------------------------------|
| algorithm | string | ❌       | Filter by algorithm name             |

**Response:**
```json
{
  "step_count": 150,
  "avg_latency": 42.3,
  "avg_utilization": 0.38,
  "packet_delivery_rate": 0.96,
  "congestion_events": 3,
  "active_algorithm": "dijkstra",
  "rl_trained": true
}
```

---

### `GET /metrics/history`

Returns the last N routing events from the database.

**Query Parameters:**

| Param | Type | Required | Default | Description          |
|-------|------|----------|---------|----------------------|
| limit | int  | ❌       | 100     | Max events to return |

**Response:**
```json
[
  {
    "id": "uuid-string",
    "timestamp": "2026-07-06T10:00:00",
    "source": "R1",
    "destination": "R5",
    "algorithm": "dijkstra",
    "path": ["R1", "R3", "R5"],
    "total_latency": 34.7,
    "success": true,
    "step_count": 150
  }
]
```

---

### `GET /metrics/algorithm-comparison`

Compare all four algorithms on deterministic sample routes from the current live state. Does not use database data — all computations are done in real-time.

**Response:**
```json
{
  "step_count": 150,
  "results": [
    {
      "algorithm": "dijkstra",
      "avg_latency": 38.2,
      "success_rate": 1.0,
      "num_decisions": 5
    },
    {
      "algorithm": "rl",
      "avg_latency": 41.5,
      "success_rate": 1.0,
      "num_decisions": 5
    }
  ]
}
```

---

## 6. Congestion Forecast

### `GET /network/congestion-forecast`

Returns short-horizon link utilization predictions using the LSTM model.

**Query Parameters:**

| Param | Type | Required | Default | Description                     |
|-------|------|----------|---------|---------------------------------|
| steps | int  | ❌       | 3       | Number of steps to forecast (1–10) |

**Response:**
```json
{
  "step_count": 150,
  "model_trained": true,
  "predictions": [
    {
      "step_ahead": 1,
      "links": [
        {
          "source": "R1",
          "target": "R2",
          "predicted_utilization": 0.47
        }
      ]
    }
  ]
}
```

> **Note**: If the LSTM model is not loaded or insufficient history exists, the predictor returns the last known snapshot as a safe fallback.

---

## 7. WebSocket Stream

### `ws://localhost:8000/ws/stream`

Persistent WebSocket connection for real-time network state updates.

**Connection Flow:**
1. Client connects
2. Server immediately pushes the current `state_update`
3. Server broadcasts `state_update` every ~1 second (on simulator tick)
4. Server broadcasts `routing_event` whenever a route is calculated via the API

**Message Envelope:**
```json
{
  "type": "state_update" | "routing_event",
  "payload": { ... }
}
```

**Auto-reconnect**: The frontend `useNetworkStream` hook implements exponential backoff reconnection (1s → 30s max).

---

## 8. Error Handling

All error responses follow FastAPI's standard format:

```json
{
  "detail": "Error description string"
}
```

Or for structured errors:
```json
{
  "detail": {
    "message": "Unknown router node.",
    "missing_nodes": ["R99"],
    "available_nodes": ["R1", "R2", "R3", ...]
  }
}
```

| HTTP Code | Meaning                                     |
|-----------|---------------------------------------------|
| 200       | Success                                     |
| 400       | Bad request (invalid algorithm, no path, etc.)|
| 404       | Unknown router node                         |
| 422       | Validation error (missing/invalid fields)   |
| 500       | Internal server error                       |

---

## 9. External APIs Used

This project does **not** integrate with any external third-party APIs (such as Gemini, Ollama, or any cloud AI services). All AI/ML models are:

- **LSTM Congestion Predictor**: Trained locally using PyTorch on simulator-generated data
- **PPO Reinforcement Learning Agent**: Trained locally using Stable-Baselines3

All routing computations, ML inference, and data storage happen entirely on the local machine.
