# 🌐 AI-Based Network Packet Routing Optimization

An intelligent network routing simulation platform that compares classical graph algorithms and AI/ML techniques for optimizing packet delivery under dynamic traffic conditions. Features a real-time React dashboard, four routing algorithms, LSTM congestion forecasting, and a PPO reinforcement learning agent.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-FF6F00?logo=openai&logoColor=white)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Routing Algorithms](#-routing-algorithms)
- [ML / AI Components](#-ml--ai-components)
- [API Endpoints](#-api-endpoints)
- [Dashboard](#-dashboard)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Overview

This project simulates a dynamic network of 10 routers with 20 links, where traffic conditions (utilization, queue sizes, packet loss) evolve in real-time. Four routing algorithms compete to find optimal paths:

- **Dijkstra** — Classic shortest-path algorithm
- **Bellman-Ford** — Distance-vector routing with cycle detection
- **Ant Colony Optimization (ACO)** — Bio-inspired metaheuristic
- **Reinforcement Learning (PPO)** — Trained neural network policy

A real-time React dashboard visualizes the network topology, compares algorithm performance, and displays congestion forecasts from an LSTM model.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔄 **Live Network Simulation** | Dynamic traffic with utilization, queue sizes, and congestion bursts |
| 🧮 **4 Routing Algorithms** | Dijkstra, Bellman-Ford, ACO, and PPO-based RL |
| 🧠 **LSTM Congestion Forecasting** | Predicts link utilization up to 10 steps ahead |
| 🤖 **PPO Reinforcement Learning** | Trained routing agent with 500k timesteps |
| 📊 **Real-Time Dashboard** | React + D3.js + Recharts with WebSocket live updates |
| 🎨 **10 Color Themes** | Dracula, Solarized, Nord, Tokyo Night, and more |
| 💾 **PostgreSQL Persistence** | All routing events and network snapshots stored in DB |
| ⚡ **WebSocket Streaming** | Sub-second state updates to the dashboard |
| 🔧 **Simulator Controls** | Step, reset, inject link failures, restore links |
| 📈 **Algorithm Comparison** | Side-by-side latency and path comparison |

---

## 🏗️ Architecture

```
Frontend (React + Vite)     ←──WebSocket──→     Backend (FastAPI)     ←──AsyncPG──→     PostgreSQL
   ├── TopologyGraph (D3)                          ├── NetworkSimulator                    ├── routing_events
   ├── CongestionHeatmap                           ├── Routing Algorithms                  ├── network_snapshots
   ├── RouteComparison                             ├── LSTM Predictor                      ├── algorithm_metrics
   └── ControlPanel                                └── PPO RL Agent                        └── packet_logs
```

> See [Documentation/03_SYSTEM_ARCHITECTURE.md](Documentation/03_SYSTEM_ARCHITECTURE.md) for detailed architecture diagrams and data flow.

---

## 🛠️ Tech Stack

| Layer          | Technologies                                          |
|----------------|-------------------------------------------------------|
| **Backend**    | Python 3.11+, FastAPI 0.111, Uvicorn, SQLAlchemy 2.0  |
| **Frontend**   | React 18, Vite 8, Tailwind CSS 3, D3.js 7, Recharts   |
| **ML / AI**    | PyTorch 2.3, Stable-Baselines3, Gymnasium              |
| **Database**   | PostgreSQL 16 (Docker), asyncpg                        |
| **Graphs**     | NetworkX 3.3                                           |
| **Real-time**  | WebSocket (FastAPI native + browser native)             |
| **DevOps**     | Docker Compose, pgAdmin 4                              |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+, Node.js 18+, Docker Desktop, Git

### 1. Clone & Configure

```bash
git clone https://github.com/<your-org>/AI-Based-Network-Packet-Routing-Optimization.git
cd AI-Based-Network-Packet-Routing-Optimization
cp .env.example .env
```

### 2. Start Database

```bash
docker compose up -d
```

### 3. Start Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Open Dashboard

Navigate to **http://localhost:5173** — you should see the live network dashboard with "Live stream connected" status.

> For detailed setup instructions, see [Documentation/02_SETUP_AND_INSTALLATION.md](Documentation/02_SETUP_AND_INSTALLATION.md).

---

## 📁 Project Structure

```
├── Documentation/           # 📘 Comprehensive project documentation (13 files)
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── api/                 # REST endpoints + WebSocket handler
│   ├── db/                  # SQLAlchemy models + async engine
│   ├── ml/                  # LSTM predictor + PPO training + Gym environment
│   ├── router/              # Dijkstra, Bellman-Ford, ACO, RL agent
│   ├── simulator/           # NetworkSimulator + data models
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── src/
│       ├── components/      # 8 React components
│       ├── hooks/           # WebSocket + REST hooks
│       └── utils/           # Color scales
├── docker-compose.yml       # PostgreSQL + pgAdmin
└── .env.example             # Environment template
```

> See [Documentation/01_PROJECT_OVERVIEW.md](Documentation/01_PROJECT_OVERVIEW.md) for the complete file tree.

---

## 🧮 Routing Algorithms

All algorithms use **congestion-aware edge weights**: `weight = base_latency × (1 + 4 × utilization²)`

| Algorithm      | Type               | Complexity        | Key Advantage                          |
|----------------|--------------------|--------------------|----------------------------------------|
| **Dijkstra**   | Greedy (optimal)   | O((V+E) log V)     | Fastest, guaranteed optimal            |
| **Bellman-Ford**| Dynamic Programming| O(VE)              | Handles distributed routing, detects cycles |
| **ACO**        | Metaheuristic      | O(iter × ants × V) | Explores multiple paths, adaptive      |
| **RL (PPO)**   | Learned Policy     | O(inference)        | Adapts to traffic patterns over time   |

> See [Documentation/06_ROUTING_ALGORITHMS.md](Documentation/06_ROUTING_ALGORITHMS.md) for implementation details and comparisons.

---

## 🧠 ML / AI Components

### LSTM Congestion Predictor

- 2-layer LSTM (hidden=64) with dropout
- Predicts next-step link utilization from a 10-step sliding window
- Trained on 2,000 self-generated simulator snapshots

### PPO Reinforcement Learning Agent

- Stable-Baselines3 PPO with MlpPolicy
- 80-dimensional observation space (4 features × 20 links)
- Trained for 500,000 timesteps (~25 min on CPU)
- Falls back to a congestion-aware heuristic when the model is not loaded

> See [Documentation/07_ML_AND_AI.md](Documentation/07_ML_AND_AI.md) for training details, reward functions, and model architecture.

**Note**: This project does **not** use any external AI APIs (Gemini, Ollama, OpenAI, etc.). All models are trained and run locally.

---

## 📡 API Endpoints

| Method | Endpoint                          | Description                              |
|--------|-----------------------------------|------------------------------------------|
| GET    | `/health`                         | Health check                             |
| GET    | `/network/state`                  | Current network state                    |
| GET    | `/network/topology`               | Static topology for graph rendering      |
| POST   | `/network/route`                  | Calculate route (single algorithm)       |
| POST   | `/network/route/compare`          | Compare all algorithms                   |
| POST   | `/sim/step`                       | Advance simulator by 1 step              |
| POST   | `/sim/reset`                      | Reset simulator                          |
| POST   | `/sim/inject-failure`             | Remove a link                            |
| POST   | `/sim/restore-link`               | Restore a failed link                    |
| GET    | `/metrics/summary`                | Aggregated performance metrics           |
| GET    | `/metrics/history`                | Recent routing events                    |
| GET    | `/metrics/algorithm-comparison`   | Live algorithm comparison                |
| GET    | `/network/congestion-forecast`    | LSTM utilization predictions             |
| WS     | `/ws/stream`                      | Real-time state updates                  |

> See [Documentation/04_API_REFERENCE.md](Documentation/04_API_REFERENCE.md) for complete request/response documentation.

---

## 🖥️ Dashboard

The React dashboard provides a real-time view of the network simulation:

- **Topology Graph**: D3 force-directed graph with live utilization colors and path highlighting
- **Congestion Heatmap**: Horizontal bar chart of top 12 most utilized links
- **Route Comparison**: Compare all 4 algorithms with latency table and bar chart
- **Simulator Controls**: Step, reset, inject failures, restore links
- **10 Themes**: Dracula, Solarized (light/dark), Nord, Tokyo Night, Monokai, and more

> See [Documentation/08_FRONTEND_DASHBOARD.md](Documentation/08_FRONTEND_DASHBOARD.md) for component details and theming system.

---

## 🧪 Testing

```bash
cd backend

# Integration test (all algorithms, 100 decisions)
python test_integration.py

# Stress tests
python test_stress_phase1.py
python -m router.test_all_routers
python -m router.test_stress_aco
python -m router.test_stress_dijkstra
```

> See [Documentation/10_TESTING.md](Documentation/10_TESTING.md) for the full testing guide and manual checklist.

---

## 📘 Documentation

Comprehensive documentation is available in the [`Documentation/`](Documentation/) folder:

| Document | Contents |
|----------|----------|
| [01_PROJECT_OVERVIEW.md](Documentation/01_PROJECT_OVERVIEW.md) | Project structure, tech stack, phase history |
| [02_SETUP_AND_INSTALLATION.md](Documentation/02_SETUP_AND_INSTALLATION.md) | Complete setup guide with troubleshooting |
| [03_SYSTEM_ARCHITECTURE.md](Documentation/03_SYSTEM_ARCHITECTURE.md) | Architecture diagrams, data flow, design decisions |
| [04_API_REFERENCE.md](Documentation/04_API_REFERENCE.md) | All REST & WebSocket endpoints with examples |
| [05_DATABASE_SCHEMA.md](Documentation/05_DATABASE_SCHEMA.md) | PostgreSQL schema, ER diagram, access patterns |
| [06_ROUTING_ALGORITHMS.md](Documentation/06_ROUTING_ALGORITHMS.md) | Algorithm implementations and comparisons |
| [07_ML_AND_AI.md](Documentation/07_ML_AND_AI.md) | LSTM predictor and PPO RL agent details |
| [08_FRONTEND_DASHBOARD.md](Documentation/08_FRONTEND_DASHBOARD.md) | React components, hooks, and theming |
| [09_CONFIGURATION.md](Documentation/09_CONFIGURATION.md) | Environment variables and config reference |
| [10_TESTING.md](Documentation/10_TESTING.md) | Test suite overview and manual checklist |
| [11_DEPLOYMENT.md](Documentation/11_DEPLOYMENT.md) | Deployment instructions (local & production) |
| [12_KNOWN_ISSUES.md](Documentation/12_KNOWN_ISSUES.md) | Known issues and limitations |
| [13_FUTURE_IMPROVEMENTS.md](Documentation/13_FUTURE_IMPROVEMENTS.md) | Phase 2 roadmap and recommendations |

---

## 👥 Team

| Name  | Role                                                         |
|-------|--------------------------------------------------------------|
| Soham | Backend, ML/AI, routing algorithms, dashboard integration    |
| Sneha | Database, frontend components, testing                       |

---

## 📄 License

This project was developed as part of an internship program. Contact the repository owner for licensing details.
