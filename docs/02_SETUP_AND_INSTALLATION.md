# Setup & Installation Guide

This guide walks a new developer through getting the full project running locally from scratch.

---

## Prerequisites

| Tool            | Minimum Version | Purpose                           |
|-----------------|----------------|------------------------------------|
| Python          | 3.11+          | Backend runtime                    |
| Node.js         | 18+            | Frontend toolchain                 |
| npm             | 9+             | Package management                 |
| Docker Desktop  | 4.x            | PostgreSQL & pgAdmin containers    |
| Git             | 2.x            | Version control                    |

> **Optional**: A CUDA-capable GPU speeds up RL training but is not required — the system falls back to CPU automatically.

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-org>/AI-Based-Network-Packet-Routing-Optimization.git
cd AI-Based-Network-Packet-Routing-Optimization
```

---

## 2. Database Setup (Docker)

The project uses PostgreSQL 16 running in a Docker container.

### 2.1 Configure Environment

Copy the example environment file and adjust if needed:

```bash
cp .env.example .env
```

Default values in `.env.example`:

```env
DATABASE_URL=postgresql+asyncpg://routinguser:routingpass@localhost:5433/routing_db
POSTGRES_USER=routinguser
POSTGRES_PASSWORD=routingpass
POSTGRES_DB=routing_db
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=admin
```

> **Note**: The PostgreSQL container maps to port **5433** (not the default 5432) to avoid conflicts with any local PostgreSQL installation.

### 2.2 Start Containers

```bash
docker compose up -d
```

This starts two services:

| Service  | Container Name    | Port  | Description               |
|----------|-------------------|-------|---------------------------|
| db       | routing-db        | 5433  | PostgreSQL 16 database    |
| pgadmin  | routing-pgadmin   | 5050  | pgAdmin 4 web UI          |

### 2.3 Verify Database

```bash
docker compose ps
```

Both containers should show `running` status. Access pgAdmin at `http://localhost:5050` using:
- Email: `admin@example.com`
- Password: `admin`

---

## 3. Backend Setup

### 3.1 Create Virtual Environment

```bash
cd backend
python -m venv .venv
```

### 3.2 Activate Virtual Environment

**Windows (PowerShell)**:
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**:
```bash
source .venv/bin/activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following key packages:

| Package           | Version | Purpose                              |
|-------------------|---------|--------------------------------------|
| fastapi           | 0.111.0 | Web framework                        |
| uvicorn[standard] | 0.29.0  | ASGI server                          |
| websockets        | 12.0    | WebSocket support                    |
| networkx          | 3.3     | Graph data structures                |
| numpy             | 1.26.4  | Numerical computing                  |
| torch             | 2.3.0   | Deep learning framework              |
| stable-baselines3 | 2.3.2   | RL algorithms (PPO)                  |
| gymnasium         | 0.29.1  | RL environment API                   |
| scikit-learn      | 1.5.0   | ML utilities                         |
| pandas            | 2.2.2   | Data manipulation                    |
| sqlalchemy        | 2.0.30  | ORM / async database access          |
| asyncpg           | 0.29.0  | PostgreSQL async driver              |
| python-dotenv     | 1.0.1   | Environment file loading             |
| pydantic          | 2.7.1   | Data validation                      |

### 3.4 Initialize Database Tables

Tables are created automatically when the backend starts, but you can also initialize manually:

```bash
python -m db.init_db
```

Expected output:
```
[DB] Creating database tables...
[DB] Database initialized successfully
   Tables created: routing_events, network_snapshots, algorithm_metrics, packet_logs
```

### 3.5 Start the Backend Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will:
1. Initialize database tables on startup
2. Start a background task that advances the network simulator every second
3. Begin broadcasting state updates over WebSocket

Verify: `http://localhost:8000/health` should return `{"status": "ok"}`

---

## 4. Frontend Setup

### 4.1 Install Node Dependencies

```bash
cd frontend
npm install
```

### 4.2 Start Development Server

```bash
npm run dev
```

The Vite dev server starts at `http://localhost:5173`.

### 4.3 Verify Connection

Open `http://localhost:5173` in a browser. The dashboard header should show **"Live stream connected"** in green, indicating the WebSocket connection to the backend is active.

---

## 5. Train ML Models (Optional)

Pre-trained models are included in `backend/ml/models/`. To retrain:

### 5.1 Train LSTM Congestion Predictor

```bash
cd backend
python -m ml.congestion_lstm
```

- Collects 2,000 simulation snapshots
- Trains for 30 epochs
- Saves to `backend/ml/models/congestion_lstm.pt`

### 5.2 Train RL (PPO) Routing Agent

```bash
cd backend
python -m ml.train_rl
```

- Trains for 500,000 timesteps (~25 minutes on CPU)
- Saves checkpoints every 50k steps
- Final model saved to `backend/ml/models/rl_router_final.zip`
- TensorBoard logs saved to `runs/ppo_routing/`

Monitor training:
```bash
tensorboard --logdir runs/ppo_routing/
```

---

## 6. Quick Start Summary

```bash
# 1. Start database
docker compose up -d

# 2. Start backend (in a new terminal)
cd backend
.\.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate        # macOS/Linux
uvicorn main:app --reload

# 3. Start frontend (in another terminal)
cd frontend
npm run dev

# 4. Open dashboard
# http://localhost:5173
```

---

## 7. Troubleshooting

| Problem                                 | Solution                                                                 |
|-----------------------------------------|--------------------------------------------------------------------------|
| `docker compose` command not found      | Install Docker Desktop or use `docker-compose` (hyphenated, older syntax)|
| Port 5433 already in use                | Change the port in `docker-compose.yml` and `.env`                       |
| `asyncpg` connection refused            | Ensure Docker containers are running: `docker compose ps`                |
| Frontend shows "Waiting for backend"    | Ensure backend is running on port 8000                                   |
| `torch` installation fails              | Try `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| RL model not found warning              | This is normal — the RL router falls back to a heuristic                 |
| PowerShell execution policy blocks venv | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`         |
