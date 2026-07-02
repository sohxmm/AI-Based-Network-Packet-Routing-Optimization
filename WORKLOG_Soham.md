# Internship Work Log

Total hours: 79.0

## 09/06/2026

Hours: 5.5

Tasks:
- Set up Python environment and project dependencies
- Reviewed NetworkX and graph-based routing concepts
- Implemented core network data models
- Built topology generation system
- Started dynamic traffic simulation

Status:
- Initial network simulator completed

## 11/06/2026

Hours: 6.0

Tasks:
- Completed network simulator functionality
- Added congestion modeling and link failure handling
- Implemented Dijkstra, Bellman-Ford, ACO and RL routing
- Added validation and comparison tests
- Studied routing algorithm behavior and evaluation metrics

Status:
- Phase 1 and routing validation completed

## 14/06/2026

Hours: 4.0

Tasks:
- Planned project architecture and workflow
- Defined milestones and implementation strategy
- Prepared work distribution document
- Reviewed backend and AI/ML integration requirements

Status:
- Project execution plan finalized

## 15/06/2026

Hours: 4.0

Tasks:
- Developed stress testing suite
- Tested simulator stability and fault recovery
- Evaluated routing performance under load
- Verified reproducible simulation outputs
- Analyzed test results and routing performance

Status:
- All stress tests passed

## 16/06/2026

Hours: 5.5

Tasks:
- Reviewed Docker fundamentals and containerized workflows
- Installed and configured Docker Desktop
- Set up PostgreSQL and pgAdmin containers
- Fixed import and environment configuration issues
- Verified FastAPI/Uvicorn startup
- Investigated database authentication issues

Status:
- Backend infrastructure configured
- Database initialization issue under investigation

## 22/06/2026

Hours: 4.0

Tasks:
- Reviewed project progress against plan
- Validated Phase 1 simulator and Phase 2 routing modules
- Tested core FastAPI endpoints
- Reviewed Phase 3 integration status
- Diagnosed database credential configuration issues

Status:
- Phase 1 and Phase 2 validation passed, REST API scaffold smoke-tested successfully
- Phase 3 partially complete; WebSocket streaming and DB-backed metrics/history pending

Next:
- Implement WebSocket streaming
- Add simulator background execution
- Complete database integration and validation after team coordination

## 23/06/2026

Hours: 3.0

Tasks:
- Implemented WebSocket endpoint `/ws/stream` for live network-state updates
- Added connected-client handling for dashboard WebSocket sessions
- Added FastAPI lifespan startup/shutdown logic for background simulator stepping
- Verified WebSocket messages use `state_update` type with serialized simulator payload
- Smoke-tested health endpoint and WebSocket stream step progression

Status:
- WebSocket streaming and automatic simulator background execution completed
- Frontend can now receive live network-state updates without manually polling REST endpoints

Next:
- Connect frontend `useNetworkStream.js` hook to `/ws/stream`
- Coordinate with Sneha on database endpoint and credential setup before DB-backed metrics/history work

## 24/06/2026

Hours: 4.0

Tasks:
- Added simulator control APIs for manual step, link failure and link restore
- Added live metrics and algorithm comparison endpoints
- Enabled CORS for frontend-backend communication

Status:
- Backend dashboard support endpoints completed

## 25/06/2026

Hours: 5.5

Tasks:
- Connected frontend to WebSocket live network stream
- Built route comparison flow using all routing algorithms
- Added topology path highlighting and congestion heatmap visualization
- Implemented LSTM congestion predictor module and forecast API scaffold
- Added integration test for 100 routing decisions across all algorithms

Status:
- Dashboard displaying live network state and fastest route results
- LSTM forecasting scaffold in place

## 26/06/2026

Hours: 5.5

Tasks:
- Designed and implemented `NetworkRoutingEnv` using Gymnasium API
- Built 80-dim normalised observation space (utilization, queue, loss, latency per link)
- Defined shaped reward function penalising latency, utilization, and packet loss
- Validated environment with `check_env()` — zero warnings
- Wrote `train_rl.py` PPO training pipeline with CheckpointCallback and EvalCallback
- Verified backend compile, integration test and frontend production build

Status:
- Gymnasium environment complete and validated
- Training pipeline ready to run

## 27/06/2026

Hours: 5.0

Tasks:
- Ran PPO training for 500,000 timesteps on CPU (24.9 min)
- Monitored TensorBoard logs; reward improved from -77 to -61 (+21%)
- Best eval checkpoint achieved mean reward -45.81 at 25k steps
- Saved final model and 10 periodic checkpoints to `backend/ml/models/`
- Updated `router/rl_agent.py` to load PPO model for deterministic inference
- Added heuristic fallback for cold-start when model not yet trained

Status:
- RL router fully operational with trained PPO policy

## 28/06/2026

Hours: 4.0

Tasks:
- Fixed topology graph re-layout bug caused by full D3 teardown on every WebSocket tick
- Refactored `TopologyGraph.jsx` into two-effect strategy: simulation built once, colors updated per tick
- Added CSS transitions for smooth link color animation
- Added drag-to-pin node behaviour so user-positioned nodes stay fixed

Status:
- Topology graph stable; only link colors update on each simulator tick

## 29/06/2026

Hours: 3.5

Tasks:
- Cleaned up and organised worklog entries
- Committed Phase 4 changes to `dev` branch and pushed to GitHub
- Prepared demo walkthrough and presentation guide for project review meeting
- Reviewed all API endpoints and verified end-to-end routing flow

Status:
- Phase 4 complete; RL routing, LSTM forecasting, and stable dashboard all production-ready
- Project ready for final presentation

## 01/07/2026

Hours: 3.5

Tasks:
- Investigated UI improvements for dashboard 
- Planned migration from 2-column to 3-column CSS Grid layout for better space utilization
- Explored Tailwind dark mode configuration

Status:
- UI improvement plan finalized

## 02/07/2026

Hours: 4.5

Tasks:
- Migrated dashboard to full-width 3-column layout
- Implemented light/dark mode theme toggle across all React components
- Updated Recharts and D3 TopologyGraph to dynamically respond to theme changes
- Added click-to-expand feature for long routing paths in the Route Comparison table

Status:
- UI overhaul successfully implemented and verified
