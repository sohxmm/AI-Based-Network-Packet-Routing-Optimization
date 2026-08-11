<div align="center">

# 🌐 AI-Based Network Packet Routing Optimization

**An intelligent network routing simulation platform that dynamically optimizes packet delivery in complex, real-time networking environments.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)](#)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](#)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)](#)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss&logoColor=white)](#)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](#)

A controlled benchmarking platform for congestion-aware routing. We implement five routing strategies — two classical, three learned — and measure them head-to-head under five reproducible network scenarios with statistical significance testing, fallback tracking, and automated degeneracy detection.

</div>

---

## 📖 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Project Objectives](#-project-objectives)
3. [System Architecture](#-system-architecture)
4. [Routing Algorithms](#-routing-algorithms)
5. [AI & ML Capabilities](#-ai--ml-capabilities)
6. [Dashboard & Visualization](#-dashboard--visualization)
7. [Quick Start](#-quick-start)
8. [Documentation](#-documentation)

---

## 🚨 Problem Statement

Traditional network routing protocols (e.g., RIP, OSPF) and static routing algorithms struggle in highly dynamic network environments characterized by frequent traffic congestion, fluctuating loads, packet loss, and latency shifts. Conventional systems rely on predefined routing logic, react slowly to congestion, and cannot intelligently adapt in real-time—often creating bottlenecks and inefficient packet flows. 

Modern large-scale systems—such as cloud infrastructure, data centers, IoT networks, and telecom systems—require **adaptive routing mechanisms** capable of dynamically optimizing packet delivery. 

**Project Goal:** Design an AI-powered intelligent packet routing system that continuously learns network conditions and dynamically selects the most efficient routing paths for packet transmission.

---

## 📊 Headline Result

| Scenario | Dijkstra | GNN | RL | ACO | MARL |
|---|---|---|---|---|---|
| normal_traffic | 60.05 | 80.21 | 83.70 | 84.78 | 106.97 |
| high_congestion | 145.19 | 160.79 | 211.19 | 185.35 | 219.05 |
| link_failures_5_10pct | 60.68 | 81.29 | 86.50 | 88.26 | 111.44 |
| congestion_bursts | 65.68 | 85.70 | 88.28 | 91.06 | 121.55 |
| large_topology_100_nodes | 910.72 | 910.82 | 914.30 | 979.01 | 927.48 |

**No learned policy outperforms congestion-weighted Dijkstra in any scenario.** This is the central finding of the project and it is a consequence of the environment design: link utilization in our simulator evolves as a random walk that is independent of routing decisions (see backend/simulator/network_sim.py). In a non-reactive network, per-path latency minimization is exactly optimal, and Dijkstra solves it exactly. Learned policies restricted to a k-shortest-path candidate set can at best tie. Section 'Limitations & What We Would Change' explains what a closed-loop simulator would need to look like for these methods to have room to win.

---

## 🎯 Project Objectives

Our intelligent network routing optimization platform is designed to:
- **Dynamically Optimize Paths:** Eliminate network congestion by intelligently routing packets away from bottlenecks.
- **Improve Delivery Efficiency:** Minimize packet loss and improve end-to-end delivery guarantees.
- **Adapt to Fluctuations:** React instantly to changing traffic conditions to maximize overall bandwidth utilization.
- **Offline-Trained Policies:** PPO, GNN, and LSTM models are trained offline against the simulator and served as frozen inference-time policies. Online/continual learning is out of scope for this phase.

---

## 🏗️ System Architecture

The platform uses a decoupled frontend-backend architecture for high-performance simulation and real-time visualization.

### Frontend (React + Vite)
- User Interface & Dashboard
- Topology Graph / D3.js visualization
- Live Analytics and Metrics

### Backend (FastAPI Server)
- REST & WebSockets API
- Network Simulator Engine
- AI / ML Models (PyTorch)
- Routing Algorithms

### Infrastructure
- PostgreSQL Database for storing snapshots & events

### Processing Pipeline
1. **Network State Analysis:** Continuous monitoring of load, queue size, and bandwidth usage.
2. **Congestion Detection & Forecasting:** LSTM networks predict upcoming congestion spikes based on historical data.
3. **AI-Based Route Selection:** GNNs and MARL agents negotiate optimal packet paths.
4. **Simulation Execution:** Packet transmission through the selected routes.
5. **Feedback Loop:** Performance metrics are recorded to improve future routing decisions.

---

## 🧮 Routing Algorithms

The system evaluates routes using a hybrid approach, benchmarking state-of-the-art AI against classical baselines:

| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **Dijkstra** | Classical | Baseline shortest-path calculation optimizing for minimal transmission cost. |
| **Bellman-Ford** | Classical | Dynamic distance-vector routing capable of detecting negative-weight cycles. |
| **ACO** | Bio-inspired | Ant Colony Optimization simulating intelligent pheromones for dynamic congestion avoidance. |
| **GNN** | Deep Learning | Graph Neural Networks extracting complex topological features for predictive routing. |
| **MARL** | Reinforcement | Multi-Agent Reinforcement Learning allowing decentralized agents to cooperate for load balancing. |

---

## 🧠 AI & ML Capabilities

Our intelligent routing engine is entirely bespoke, utilizing local PyTorch models to drive real-time decisions:

- **Congestion Prediction (LSTM):** Proactively predicts future network congestion before it occurs by analyzing queue sizes, network loads, and time-series utilization data.
- **Self-Learning Routing (PPO):** The reinforcement learning engine optimizes decisions based on previous packet success rates, delay metrics, and real-time traffic patterns.
- **Intelligent Load Balancing:** Distributes packets across multiple sub-optimal routes to prevent bottlenecks and maximize overall throughput.

*Note: All models are natively built and trained using PyTorch and Stable-Baselines3. No external APIs are used.*

---

## 🖥️ Dashboard & Visualization

The real-time **Network Visualization Dashboard** functions as the central control plane, built with React and Tailwind CSS:
- **Interactive Topology Graph:** Live packet movement visualization and node mapping.
- **Congestion Heatmaps:** Instantly identifies congested nodes and link overloads.
- **Performance Analytics:** Live throughput, latency tracking, and dynamic algorithm benchmarking tables.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop

### 1. Setup Environment
Clone the repository and spin up the database infrastructure:
```bash
git clone https://github.com/<your-org>/AI-Based-Network-Packet-Routing-Optimization.git
cd AI-Based-Network-Packet-Routing-Optimization
cp .env.example .env
docker compose up -d
```

### 2. Run the Backend Engine
Set up the Python virtual environment and start the FastAPI simulator:
```bash
cd backend
python -m venv .venv
# On Windows: .\.venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Run the Frontend Dashboard
In a separate terminal, launch the Vite development server:
```bash
cd frontend
npm install
npm run dev
```

Navigate to **[http://localhost:5173](http://localhost:5173)** to view the live simulation.

---

## 🐳 Docker Deployment

For a fully containerized deployment (no local Python or Node.js setup required), the entire stack can be launched with a single command. Pre-trained models are baked into the backend image — no training wait on first run.

### Prerequisites
- Docker Desktop

### Run the Full Stack

```bash
git clone https://github.com/SwDC-kjsse/ai--ml-3
cd ai--ml-3
cp .env.example .env
docker compose up --build
```

This single command starts all four services:

| Service | Container | URL |
| :--- | :--- | :--- |
| React Dashboard | `routing-frontend` | http://localhost:5173 |
| FastAPI Backend | `routing-backend` | http://localhost:8000/docs |
| PostgreSQL | `routing-db` | `localhost:5433` |
| pgAdmin | `routing-pgadmin` | http://localhost:5050 |

### Subsequent Runs

After the initial build, start the stack instantly with:

```bash
docker compose up
```

### Stop the Stack

```bash
docker compose down
```

To also remove all stored data (database volumes):

```bash
docker compose down -v
```

### Docker vs Manual Setup

| | Docker | Manual |
| :--- | :--- | :--- |
| Setup time | One command | Multiple terminals |
| Python/Node.js required | ❌ | ✅ |
| Pre-trained models | ✅ Baked in | ✅ On disk |
| Hot reload | ❌ | ✅ |
| Best for | Deployment / sharing | Development |

> **Note:** Pre-trained PPO and GNN models are included in the image. The system is ready to route immediately after `docker compose up --build` completes.
---
## 📘 Documentation

For detailed guides, refer to the [Documentation/](./Documentation/) directory:
- [Project Overview](./Documentation/01_PROJECT_OVERVIEW.md)
- [System Architecture](./Documentation/03_SYSTEM_ARCHITECTURE.md)
- [API Reference](./Documentation/04_API_REFERENCE.md)
- [Routing Algorithms](./Documentation/06_ROUTING_ALGORITHMS.md)

### Expected Real-World Applications
- **ISPs:** Live traffic optimization.
- **Cloud Data Centers:** Efficient cross-cluster packet routing.
- **Smart Cities:** IoT communication network reliability.
- **Telecom Networks:** Fault-tolerant, self-healing backbones.

---

## ⚠️ Limitations & What We Would Change

- The simulator is open-loop. Routing decisions do not alter link utilization, so load-balancing objectives cannot be demonstrated.
- Our PPO agent's evaluation reward is flat across 500k timesteps (r² = 0.001, slope not significant at p = 0.87). We report this rather than the training-reward figure.
- Bellman-Ford returns an identical path to Dijkstra in 100% of trials; it is included for pedagogical completeness, not as an independent baseline.
- The 100-node topology generated by our simulator is a ring (degree 2). It is a scale test, not a complexity test.

---

<div align="center">
<i>Developed as a part of an AI/ML Internship Team project.</i>
</div>
