# AI-Based Network Packet Routing Optimization

An intelligent network routing simulation platform that explores how classical graph algorithms and AI/ML techniques can optimize packet delivery under changing traffic, congestion, latency, and packet-loss conditions. The project will simulate router topologies, compare Dijkstra, Bellman-Ford, Ant Colony Optimization, and Reinforcement Learning routing strategies, forecast congestion with an LSTM model, and visualize network behavior in a real-time dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?logo=tailwindcss&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)

## Project Scope

- Simulate a dynamic network of routers and links.
- Implement baseline and adaptive routing algorithms.
- Predict congestion using time-series ML.
- Train an RL routing agent.
- Stream live network state into a React dashboard.
- Store routing events and snapshots in PostgreSQL.

## Getting Started

These commands are placeholders for Phase 0. Detailed setup will be expanded as each phase is implemented.

```bash
# Backend
cd backend
python -m venv .venv
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev

# Database
docker compose up -d
```

## Repository Branches

- `main`: stable branch.
- `dev`: active development branch.

## Phase Status

- Phase 0: project scaffold.
- Phase 1: network simulator.
- Phase 2: routing algorithms.
- Phase 3: FastAPI backend and database.
- Phase 4: LSTM congestion predictor.
- Phase 5: RL environment and training.
- Phase 6: React dashboard.
- Phase 7: integration and polish.
