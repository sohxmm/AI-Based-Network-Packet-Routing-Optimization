# Internship Work Log — Sneha Ramamurthy

Total Hours: 97.0

## 2026-06-05
**Hours:** 4.0

**Tasks:**
- Reviewed Python fundamentals and data structures
- Practiced dictionaries, lists, tuples and functions
- Solved programming exercises to strengthen Python proficiency
- Prepared development environment for project implementation

**Phase:** Preparation


**Status:** Python fundamentals completed

---

## 2026-06-06
**Hours:** 4.0

**Tasks:**
- Studied graph theory concepts relevant to network routing
- Learned graph representations and weighted graphs
- Explored shortest path problems and graph traversal techniques
- Built foundational understanding required for routing algorithms

**Phase:** Preparation


**Status:** Graph theory fundamentals completed

---

## 2026-06-07
**Hours:** 6.0

**Tasks:**
- Installed and configured NetworkX
- Learned graph creation and visualization using NetworkX
- Added nodes, edges and edge attributes
- Explored NetworkX APIs and graph operations
- Visualized sample network topologies using Matplotlib

**Phase:** Phase 1 (Network Simulator)


**Status:** NetworkX fundamentals completed

---

## 2026-06-08
**Hours:** 5.0

**Tasks:**
- Built a static network topology
- Added link properties including latency, bandwidth and packet loss
- Implemented shortest-path routing using NetworkX
- Calculated total latency, packet loss and hop count
- Visualized the network topology and routing paths

**Phase:** Phase 1 (Network Simulator)


**Status:** Static network simulation completed

---

## 2026-06-09
**Hours:** 3.5

**Tasks:**
- Studied Dijkstra's shortest path algorithm
- Implemented Dijkstra's algorithm using an adjacency list and priority queue
- Tested shortest path computation on weighted graphs
- Verified routing outputs and path costs

**Phase:** Preparation

---

## 2026-06-10
**Hours:** 3.0

**Tasks:**
- Implemented Bellman-Ford algorithm
- Studied edge relaxation and shortest path updates
- Compared Bellman-Ford and Dijkstra algorithms
- Evaluated routing behavior on weighted networks

**Phase:** Preparation

---

## 2026-06-11
**Hours:** 5.0

**Tasks:**
- Developed a packet routing mini project using NetworkX
- Simulated packet transmission between source and destination nodes
- Computed routing metrics including latency, packet loss and hop count
- Evaluated network behavior under different paths

**Phase:** Phase 1 / Preparation

---

## 2026-06-15
**Hours:** 4.0

**Tasks:**
- Reviewed and finalized documentation of work distribution and execution plan
- Cloned and set up the repository
- Created a virtual environment and installed necessary dependencies and packages
- Reviewed data_models.py and network_sim.py


**Phase:** Phase 1 (Network Simulator)


**Status:** Project execution plan finalized and repository set up

---

## 2026-06-16
**Hours:** 5.5

**Tasks:**
- Learnt Docker and related concepts (~2 hrs)
- Learning Git and GitHub for version control
- Set up Docker configurations for the project (PostgreSQL and pgAdmin) and copied required environment variables
- Verified working of images and containers
- Npm installation and setup done
- Implemented docstrings and to_dict() for easier help and JSON conversion; environment variables defined

**Phase:** Phase 1 (Network Simulator) 

**Status:** All dependencies, packages, and containers worked properly
---

## 2026-06-17
**Hours:** 5.5

**Tasks:**
- Learnt Docker and related concepts (~2 hrs)
- Learning Git and GitHub for version control
- Set up Docker configurations for the project (PostgreSQL and pgAdmin) and copied required environment variables
- Verified working of images and containers
- Npm installation and setup done
- Implemented docstrings and to_dict() for easier help and JSON conversion; environment variables defined

**Phase:** Phase 1 (Network Simulator) 

**Status:** All dependencies, packages, and containers worked properly
---

## 2026-06-18
**Hours:** 7.5

**Tasks:**
- Learnt Docker and related concepts (~2 hrs)
- Learning Git and GitHub for version control
- Set up Docker configurations for the project (PostgreSQL and pgAdmin) and copied required environment variables
- Verified working of images and containers
- Npm installation and setup done
- Implemented docstrings and to_dict() for easier help and JSON conversion; environment variables defined

**Phase:** Phase 1 (Network Simulator) 

**Status:** All dependencies, packages, and containers worked properly
---

## 2026-06-19
**Hours:** 2.5 - Sneha Ramamurthy

**Tasks:**
- Continued investigating the Invalid PasswordError preventing successful execution of db/init_db.py
- Verified PostgreSQL Docker container configuration:
  - Confirmed the routinguser role exists with superuser privileges
  - Confirmed the routing_db database exists and is owned by routinguser
  - Ruled out Docker and PostgreSQL configuration issues as the source of the problem
- Tested database authentication directly using psql -h localhost -W to mirror the same network-based authentication flow used by Python applications
- Successfully authenticated using the credentials stored in .env
- Confirmed database credentials are valid and consistent
- Reviewed the DATABASE_URL value in .env
- Verified the password contains no special characters requiring URL encoding
- Ruled out connection-string parsing issues caused by password formatting
- Performed root-cause analysis and narrowed the issue to two likely causes:
  - load_dotenv() may not be locating the env file when python -m db.init_db is executed from the backend/ directory
  - A previously exported DATABASE_URL environment variable may be overriding the value loaded from .env
- Planned additional diagnostics to inspect environment-variable precedence and working-directory behavior before reattempting database initialization

**Phase:** Week 3


**Status:** Database infrastructure and credentials have been independently verified through direct PostgreSQL testing. The root cause of the Invalid PasswordError has not yet been confirmed, but investigation has narrowed the issue to environment loading or environment-variable precedence. Next steps involve validating .env discovery and runtime configuration before retrying init_db.py.

---

## 2026-06-21
**Hours:** 4.5

**Tasks:**
- Debugged persistent InvalidPasswordError blocking db/init_db.py

- Discovered root cause: native PostgreSQL 18 Windows service (
  postgresql-x64-18) competing on port 5432 with Docker container, intercepting Python/asyncpg connections
- Stopped and disabled native PostgreSQL service via admin PowerShell
- Verified fix via raw asyncpg connection test and python -m db.init_db

- Tables routing_events and network_snapshots successfully created in PostgreSQL

**Phase:** Week 3 (Database + REST Endpoints)
**Status:** init_db.py working. Database tables confirmed via psql.

---

## 2026-06-22
**Hours:** 6.0

**Tasks:**
- Built main.py
- FastAPI app with asynccontextmanager lifespan, shared NetworkSimulator on app.state, CORS middleware for localhost:5173

- Built api/routes.py
- all 10 REST endpoints with DB logging, 404/422 error handling, and metrics aggregation
- Validated all endpoints via Swagger UI at http://localhost:8000/docs

- Wrote test_stress_api.py
- 15 test cases covering all endpoints, error cases, link failure/restore lifecycle, and performance under load
- Fixed Windows async event loop compatibility with asyncio.WindowsSelectorEventLoopPolicy

- Fixed TestClient lifespan issue by setting app.state.simulator directly on app object
- All stress tests passing; committed and pushed to sneha-week1


**Phase:** Week 3 (Database + REST Endpoints)
**Status:** All REST endpoints validated via Swagger UI and stress test. Tables confirmed in PostgreSQL via psql. Week 3 complete.

---

## 2026-06-23
**Hours:** 3.5


**Tasks:**
- Built `ml/rl_environment.py` — Gymnasium `NetworkRoutingEnv` wrapping `NetworkSimulator`
  - Observation space: flat vector of 20 link utilization values
  - Action space: `Discrete(5)` — pick one of k candidate paths
  - Reward function: delivery bonus (+1.0), latency penalty (-0.01 × latency), congestion penalty (-0.5 × avg utilization)
  - Random src/dst pairs per episode for generalization across all node pairs
- Verified environment with `stable_baselines3.common.env_checker.check_env()` — passed
- Built `ml/train_rl.py` — PPO training pipeline with `CheckpointCallback` and `EvalCallback`
  - Saves checkpoints every 10,000 steps
  - Evaluates on a separate env (different seed) every 5,000 steps
  - Saves best model separately from final model
- Updated `router/rl_agent.py` — PPO model inference with random path fallback when no model loaded
- Committed RL environment and training pipeline to `sneha-week1`
**Phase:** Week 4 (RL Environment + PPO Training Pipeline)
**Status:** RL environment verified, training pipeline ready. Model training pending.

---

## 2026-06-24
**Hours:** 6.0


**Tasks:**
- Built `frontend/src/hooks/useNetworkStream.js` — custom WebSocket hook with exponential backoff reconnection (1s → 2s → 4s, max 30s), handles `network_state` and `routing_event` message types
- Built `frontend/src/components/TopologyGraph.jsx` — D3 force-directed network graph
  - Node coloring: blue (normal) / red (any connected link >80% utilization)
  - Link coloring: green → yellow → orange → red by utilization; thickness scales with utilization
  - Drag behavior: nodes are repositionable via D3 drag
  - Tooltip: hover any link to see utilization, latency, queue size, packet loss
  - Path animation: 2-second cyan glow/pulse on links used in last routing decision
- Built `frontend/src/components/RouteComparison.jsx` — src/dst dropdowns, Compare All button, results table showing all 4 algorithms, best result highlighted in green with ★
- Built `frontend/src/components/ControlPanel.jsx` — Step +1, Step +10, Inject Failure modal, Restore Link modal, Algorithm selector dropdown, feedback toasts
- Updated `frontend/src/App.jsx` — wired all components together, 2-second polling of `/network/state`, connected indicator, step counter in header
- Verified full end-to-end dashboard working: topology renders live, Compare All returns all 4 algorithms, Step +1/+10 advance the simulation, inject failure removes links from graph
- Committed all frontend work to `sneha-week1`

**Phase:** Week 5 (React Dashboard)
**Status:** Dashboard fully functional. All 4 algorithms visible in route comparison. Live topology updating every 2 seconds.

---

## 2026-06-25
**Hours:** 2.0


**Tasks:**
- Demo preparation and testing
- Verified full stack running: Docker (PostgreSQL), uvicorn backend, Vite frontend
- Debugged uvicorn startup issue caused by Python 3.14 subprocess conflict with venv (3.11); resolved by running `python -m uvicorn main:app` instead of the bare `uvicorn` command
- Ran demo walkthrough: live topology, route comparison across all 4 algorithms, step controls, link failure/restore

**Phase:** Week 4–5
**Status:** Full stack demo running successfully. Pending: parts of frontend.

---

## 2026-06-28
**Hours:** 4.0


**Tasks:**
- Fixed `.env` credentials in new project folder (`swdc_final/ai--ml-3`) — `DATABASE_URL` had placeholder `user:password` values; updated to real `routinguser:routingpass` credentials
- Debugged and resolved `.env` formatting issue — all variables had been merged onto one line, causing PostgreSQL to reject the database name
- Ran `python -m db.init_db` to create tables in fresh Docker container
- Wired trained PPO model into live API — added auto-load in `api/routes.py` on server startup so RL algorithm uses real trained decisions instead of random path fallback
- Updated `api/routes.py` to load model from `ml/models/ppo_routing_agent.zip` at import time

**Phase:** Week 5 (UI/UX improvements) + Week 4 (RL model integration)
**Status:** Dashboard significantly improved. RL model loading on startup. Metrics updating live.

---

## 2026-06-29
**Hours:** 3.5


**Tasks:**
- Trained PPO routing agent — ran `python -m ml.train_rl` for 50,000 timesteps
  - Used `MlpPolicy`, learning rate 3e-4, `n_steps=2048`, `batch_size=64`, `n_epochs=10`
  - Checkpoints saved every 10,000 steps to `ml/models/checkpoints/`
  - Best model saved to `ml/models/best/`
  - Final model saved to `ml/models/ppo_routing_agent.zip` (169KB)
- Investigated why RL agent replicates Dijkstra behavior on 10-node network — confirmed this is expected: small state space with unambiguous optimal actions means RL converges to the same greedy solution. RL's advantage would emerge on larger, more dynamic topologies.
- Investigated and documented UI/UX improvement plan — fixed moving graph issue (D3 simulation rebuilding on every poll), identified side panel additions (network health, algorithm comparison, node detail, simulation log)
- Pushed all changes to `sneha-week1`

**Phase:** Week 4 (PPO training) + Week 5
**Status:** Trained model committed and loading in production. UI improvement plan documented.
---

## 2026-06-30
**Hours:** 3

**Tasks:**
- Designed and documented improved dashboard layout with left sidebar (network health, algorithm comparison, topology stats, recent decisions) and right sidebar (stat cards, node detail panel with link utilization bars, route comparison table, simulation log)[yet to finalize and implement]
- Investigated Wireshark/Nmap/Mininet integration feasibility — documented post-project extension plan with correct ordering (Wireshark first once WebSocket is live, then Nmap, then Mininet)
- Updated worklog for June 22–30
---

## 2026-07-01
**Hours:** 3.5

**Tasks:**
- Updated frontend/src/utils/colorScales.js — added ALGO_COLORS export and utilizationToColor helper function
- Updated frontend/src/index.css — added CSS custom properties for full color system and warmer dark navy background
- Rewrote frontend/src/components/RouteComparison.jsx — algorithm-specific colored tags and bar chart, fixed page-blank crash caused by undefined results (Cannot read properties of undefined (reading 'success')), added null guards in both map loops
- Rewrote frontend/src/components/CongestionHeatmap.jsx — fetches /network/state directly, renders top 8 links sorted by utilization with green→yellow→orange→red color coding
- Debugged RouteComparison crash via browser console (F12 → Console)
- Investigated RL agent convergence behavior on 10-node network

**Phase:** Week 5 (UI/UX improvements)
**Status:** Color system applied. RouteComparison crash fixed. CongestionHeatmap colors working.

---

## 2026-07-02
**Hours:** 2.0

**Tasks:**
- Designed improved dashboard layout with left and right side panels
- Reviewed and proposed new color palette (warmer dark navy, algorithm-specific colors, utilization gradient)
- Committed trained PPO model (ppo_routing_agent.zip, 169KB) to repository using git add -f
- Confirmed all Member 1 deliverables complete across Weeks 1–5
- Updated worklog and resume snippet for project

**Phase:** Week 5 (polish + documentation)
**Status:** All Member 1 work complete. 

**Phase:** Week 5 
**Status:**  Pending: merge sneha-week1 PR into dev, changes to be made in UI/UX then dev into main for final submission.
---

## 2026-07-03
**Hours:** 3.5

**Tasks:**

-Tried better colour cominations for the simulation algorithms and the congestion heatmaps on older theme and layout.
-Pulled the main branch to check the updated changes and finalize on the theme and UI/UX layout and themes .

**Phase:** Week 5 (UI/UX improvements)
**Status:** Color system foundation laid using the team's chosen palette.

---


