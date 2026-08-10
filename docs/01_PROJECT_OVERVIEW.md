# Project Overview & Structure

## 1. Introduction

**AI-Based Network Packet Routing Optimization** is an intelligent network routing simulation platform that explores how classical graph algorithms and AI/ML techniques can optimize packet delivery under dynamic traffic conditions. The system simulates realistic router topologies, compares multiple routing strategies, forecasts congestion using deep learning, and visualizes network behavior in a real-time interactive dashboard.

### Project Goals

- Simulate a dynamic network of routers and links with realistic traffic patterns
- Implement and compare baseline and adaptive routing algorithms (Dijkstra, Bellman-Ford, ACO, RL, GNN, MARL)
- Predict congestion using time-series ML (LSTM)
- Train Reinforcement Learning routing agents (PPO, MARL) and a Graph Neural Network (GNN) to learn optimal routing policies
- Stream live network state to a React dashboard via WebSocket
- Store routing events and network snapshots in PostgreSQL for historical analysis

---

## 2. Tech Stack Summary

| Layer        | Technology                                    |
|------------- |-----------------------------------------------|
| Backend      | Python 3.11+, FastAPI 0.111, Uvicorn          |
| Frontend     | React 18, Vite 8, Tailwind CSS 3              |
| ML / AI      | PyTorch 2.3, Stable-Baselines3, Gymnasium     |
| Database     | PostgreSQL 16 (via Docker), SQLAlchemy 2.0     |
| Graphs       | NetworkX 3.3                                  |
| Visualization| D3.js 7, Recharts 2.12                        |
| Real-time    | WebSocket (native FastAPI / browser)           |
| DevOps       | Docker Compose, pgAdmin 4                     |

---

## 3. Repository Structure

```
AI-Based Network Packet Routing Optimization/
│
├── README.md                          # Project overview & quick start
├── .env.example                       # Environment variable template
├── .gitignore                         # Git ignore rules
├── docker-compose.yml                 # PostgreSQL + pgAdmin containers
├── WORKLOG_Soham.md                   # Developer work log (Soham)
├── WORKLOG_Sneha.md                   # Developer work log (Sneha)
│
├── Documentation/                     # 📘 Comprehensive project docs
│   ├── 01_PROJECT_OVERVIEW.md         # This file — structure & overview
│   ├── 02_SETUP_AND_INSTALLATION.md   # Setup & installation guide
│   ├── 03_SYSTEM_ARCHITECTURE.md      # Architecture & data flow
│   ├── 04_API_REFERENCE.md            # REST & WebSocket API docs
│   ├── 05_DATABASE_SCHEMA.md          # PostgreSQL schema details
│   ├── 06_ROUTING_ALGORITHMS.md       # Algorithm implementations
│   ├── 07_ML_AND_AI.md                # ML models (LSTM + RL/PPO)
│   ├── 08_FRONTEND_DASHBOARD.md       # React dashboard guide
│   ├── 09_CONFIGURATION.md            # Environment vars & config
│   ├── 10_TESTING.md                  # Testing strategy & scripts
│   ├── 11_DEPLOYMENT.md               # Deployment instructions
│   ├── 12_KNOWN_ISSUES.md             # Known issues & limitations
│   └── 13_FUTURE_IMPROVEMENTS.md      # Roadmap for Phase 2+
│
├── backend/                           # Python FastAPI application
│   ├── main.py                        # FastAPI app entry point & lifespan
│   ├── requirements.txt               # Python dependencies
│   ├── .env                           # Local environment overrides
│   │
│   ├── api/                           # REST & WebSocket API layer
│   │   ├── __init__.py
│   │   ├── routes.py                  # All REST endpoint handlers
│   │   ├── state.py                   # Singleton simulator state
│   │   └── websocket.py               # WebSocket connection manager
│   │
│   ├── db/                            # Database layer (SQLAlchemy)
│   │   ├── database.py                # Async engine & session factory
│   │   ├── models.py                  # ORM models (4 tables)
│   │   └── init_db.py                 # Standalone DB init script
│   │
│   ├── ml/                            # Machine Learning modules
│   │   ├── __init__.py
│   │   ├── congestion_lstm.py         # LSTM congestion predictor
│   │   ├── rl_environment.py          # Gymnasium RL environment
│   │   ├── train_rl.py               # PPO training pipeline
│   │   ├── train_gnn.py              # GNN training pipeline
│   │   ├── train_multi_agent.py      # MARL training pipeline
│   │   ├── gnn_model.py              # PyTorch GNN architecture
│   │   ├── multi_agent_rl_environment.py # Multi-agent Gym env
│   │   └── models/                    # Saved model checkpoints
│   │       ├── best_model.zip
│   │       ├── congestion_lstm.pt
│   │       ├── rl_router_final.zip
│   │       ├── gnn_router.pt
│   │       └── multi_agent_region_*.zip
│   │
│   ├── router/                        # Routing algorithm implementations
│   │   ├── __init__.py
│   │   ├── dijkstra.py                # Dijkstra's shortest path
│   │   ├── bellman_ford.py            # Bellman-Ford algorithm
│   │   ├── aco.py                     # Ant Colony Optimization
│   │   ├── rl_agent.py                # PPO-based RL router
│   │   ├── gnn_router.py              # GNN-based router
│   │   ├── multi_agent_router.py      # MARL-based router
│   │   ├── test_all_routers.py        # Router unit tests
│   │   ├── test_stress_aco.py         # ACO stress tests
│   │   └── test_stress_dijkstra.py    # Dijkstra stress tests
│   │
│   ├── simulator/                     # Network simulation engine
│   │   ├── __init__.py
│   │   ├── data_models.py             # LinkState, NetworkState, RoutingDecision
│   │   └── network_sim.py             # Core NetworkSimulator class
│   │
│   ├── test_integration.py            # End-to-end integration test
│   ├── test_gnn.py                    # GNN integration tests
│   ├── test_stress_phase1.py          # Phase 1 stress test suite
│   └── tests/                         # Pytest test suites (new)
│       ├── test_predictive_routing.py
│       ├── test_multi_agent_routing.py
│       └── test_benchmark_*.py
│
├── frontend/                          # React + Vite application
│   ├── index.html                     # HTML entry point
│   ├── package.json                   # Node dependencies
│   ├── vite.config.js                 # Vite build configuration
│   ├── tailwind.config.js             # Tailwind CSS configuration
│   ├── postcss.config.js              # PostCSS configuration
│   │
│   └── src/
│       ├── main.jsx                   # React app mount point
│       ├── App.jsx                    # Root app component
│       ├── index.css                  # Global styles & theme engine
│       │
│       ├── components/
│       │   ├── TopologyGraph.jsx       # D3 force-directed graph
│       │   ├── CongestionHeatmap.jsx   # Recharts bar heatmap
│       │   ├── RouteComparison.jsx     # Algorithm comparison table
│       │   ├── ControlPanel.jsx        # Simulator controls
│       │   ├── MetricsPanel.jsx        # Key metric cards
│       │   ├── LeftPanel.jsx           # Queue size panel
│       │   ├── RightPanel.jsx          # Topology stats panel
│       │   ├── PacketAnimator.jsx      # (Placeholder for animation)
│       │   ├── ExperimentBuilder.jsx   # Phase 8 scenario builder
│       │   ├── BenchmarkResultView.jsx # Phase 8 benchmark reporting
│       │   ├── BenchmarkReport.jsx     # Phase 8 individual report
│       │   ├── GuardrailBadge.jsx      # Safety limit indicator
│       │   └── __tests__/              # React component tests
│       │
│       ├── hooks/
│       │   ├── useNetworkStream.js     # WebSocket live data hook
│       │   └── useRouteRequest.js      # REST API request hook
│       │
│       └── utils/
│           └── colorScales.js          # Utilization color mapping
│
└── datasets/                          # Data storage (gitignored)
    └── README.md                      # Dataset documentation
```

---

## 4. Phase History

| Phase   | Description                          | Status      |
|---------|--------------------------------------|-------------|
| Phase 0 | Project scaffold & planning          | ✅ Complete |
| Phase 1 | Network simulator                    | ✅ Complete |
| Phase 2 | Routing algorithms                   | ✅ Complete |
| Phase 3 | FastAPI backend & database           | ✅ Complete |
| Phase 4 | LSTM congestion predictor            | ✅ Complete |
| Phase 5 | RL environment & PPO training        | ✅ Complete |
| Phase 6 | React dashboard                      | ✅ Complete |
| Phase 7 | Integration, polish & theming        | ✅ Complete |
| Phase 8 | Advanced algorithms & Benchmarking   | ✅ Complete |

---

## 5. Team

| Name  | Role                                  |
|-------|---------------------------------------|
| Soham | Backend, ML/AI, routing algorithms, dashboard integration |
| Sneha | Database (SQLAlchemy models, PostgreSQL/Docker setup), backend (FastAPI, 10 endpoints), algorithms (Dijkstra, ACO, PPO RL training), frontend components (React/D3.js dashboard), and testing (stress tests, infrastructure debugging) |
