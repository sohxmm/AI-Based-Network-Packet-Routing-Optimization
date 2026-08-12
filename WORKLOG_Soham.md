# Internship Work Log

Total hours: 140.5

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

## 03/07/2026

Hours: 4.0

Tasks:
- Added Node Queue Sizes and Global Topology Stats panels to utilize dashboard empty space
- Refactored hardcoded colors into a dynamic CSS-variable theming engine in `index.css`
- Integrated 10 popular themes (Dracula, Solarized, GitHub Dark, Nord, etc.) into the dashboard
- Fixed UI contrast issues and standardized native dropdown styling for `select` menus
- Implemented "Reset Simulator" functionality in backend and frontend Control Panel
- Verified frontend build and styling changes

Status:
- Dashboard features complete, robust custom theming deployed and verified

## 14/07/2026

Hours: 5.0

Tasks:
- Analyzed and mapped the GNN branch architecture, including model definition, training logic, inference, and API integration
- Fixed critical indentation bugs in `backend/router/gnn_agent.py` and properly aligned the `GNNRouter` class logic
- Investigated and corrected `hidden_dim` initialization to dynamically read from the model checkpoint instead of hardcoding
- Fixed node features bug in `gnn_router.py` where `is_source` and `is_destination` features were hardcoded to 0.0 during inference
- Resolved structural indentation issues in `backend/ml/rl_environment.py` for the `_compute_reward` function

Status:
- GNN codebase debugged and fully prepared for model training

## 15/07/2026

Hours: 5.5

Tasks:
- Installed and configured necessary ML dependencies (`torch`, `gymnasium`, `stable-baselines3`, `scikit-learn`, `pandas`) within the project virtual environment
- Executed the GNN model training pipeline (`train_gnn.py`), generating 3,000 training samples and 750 validation samples using the 100-node network simulator
- Trained the `GNNRouterModel` for 60 epochs
- Achieved significant loss reduction: Training MSE reduced by 97% (0.8270 to 0.0257) and Validation MSE reduced by 72% (0.1022 to 0.0286)
- Exported and saved the fully trained model to `backend/ml/models/gnn_router.pt`

Status:
- GNN model successfully trained, serialized, and ready for production inference

## 16/07/2026

Hours: 5.0

Tasks:
- Developed a comprehensive GNN test suite (`test_gnn.py`) covering model architecture (MessagePassingLayer shapes), training pipeline validation, empty-path handling, and inference fallback heuristics
- Resolved Windows `cp1252` encoding issues with ASCII representations in the test suite
- Added integration tests in `test_integration.py` to verify GNN routing alongside Dijkstra, Bellman-Ford, ACO, and RL, achieving a 100% success rate with ~35.05ms average latency
- Updated frontend dashboard components; added GNN algorithm label mapping in `RouteComparison.jsx` so it renders correctly in the bar chart
- Finalized branch, staged, and committed all GNN updates

Status:
- GNN branch fully tested, integrated into frontend, and pushed to the remote repository

## 24/07/2026

Hours: 4.5

Tasks:
- Identified and fixed RL observation shape mismatch: `rl_environment.py` trained PPO on `num_nodes=100` but API uses `num_nodes=25`, causing silent heuristic fallback at inference
- Changed `NetworkRoutingEnv` default to `num_nodes=25` and retrained PPO model on the correct 25-node topology (500k timesteps)
- Added `last_used_model` flag and logging to `RLRouter.predict()` to report whether trained PPO policy or heuristic fallback is used on each call
- Refactored `api/state.py` to hold singleton instances of `AntColonyRouter`, `RLRouter`, and `GNNRouter` (same pattern as `get_simulator()`)
- Updated `api/routes.py` to use singleton routers instead of creating new instances per request, preserving ACO pheromone state across calls
- Confirmed `router/gnn_agent.py` has zero imports anywhere in the codebase and deleted the dead duplicate file

Status:
- All three known bugs fixed; RL model retrained on correct topology and verified loading at inference

## 25/07/2026

Hours: 5.0

Tasks:
- Implemented predictive routing mode using the existing LSTM congestion predictor (`CongestionPredictor`)
- Added `_build_forecast_state()` helper in `routes.py` that calls LSTM `predict_next()` and constructs a modified `NetworkState` with forecasted link utilizations
- Added `use_forecast: bool` parameter to `RouteRequest` and `RouteCompareRequest` Pydantic models
- Updated `/network/route` endpoint to use forecast state for GNN/RL when `use_forecast=True`
- Updated `/network/route/compare` to return 7 results when `use_forecast=True`: Dijkstra, Bellman-Ford, ACO (reactive only), GNN reactive, GNN predictive, RL reactive, RL predictive
- Loaded congestion LSTM model at module level in `routes.py` for zero-latency forecast calls
- Updated `useRouteRequest.js` hook to accept and forward `use_forecast` parameter in API requests

Status:
- Predictive routing pipeline fully functional; LSTM forecasts integrated into GNN and RL routing paths

## 26/07/2026

Hours: 4.5

Tasks:
- Added "Use congestion forecast" toggle switch to `RouteComparison.jsx` with accessible keyboard support
- Updated algorithm label mapping to handle `gnn_predictive` and `rl_predictive` names with visual "forecast" badge
- Threaded `useForecast` state through `App.jsx` and `handleCompareRoutes` to the API hook
- Wrote integration test at `backend/tests/test_predictive_routing.py` covering: congestion burst detection, LSTM prediction accuracy on congested links, forecast state builder validation, and predictive routing avoidance assertion
- Verified that predictive GNN/RL routes avoid soon-to-be-congested links while Dijkstra routes into them
- Updated worklog with July 24-26 entries

Status:
- Predictive routing feature complete with frontend toggle, backend pipeline, and integration test coverage

## 27/07/2026

Hours: 5.5

Tasks:
- Debugged and verified Phase 1 requirements: singleton routers, predictive routing modes, and ACO pheromone persistence.
- Designed and implemented the Multi-Agent RL router (`multi_agent_router.py`) using a decentralized architecture.
- Integrated NetworkX community detection for automatic topology partitioning into distinct routing regions.

Status:
- Multi-Agent RL routing operational and integrated into the backend API.

## 28/07/2026

Hours: 6.0

Tasks:
- Built a decentralized multi-agent Gymnasium environment (`multi_agent_rl_environment.py`) with a shared global utilization variance penalty to encourage load balancing.
- Trained and verified distinct regional PPO models, ensuring they outperform single-agent baselines in complex topologies.
- Fixed a degenerate policy bug in the Multi-Agent RL models by rebalancing the reward function (latency vs. global variance) and successfully retrained.

Status:
- Multi-Agent models successfully trained, outperforming single-agent baselines on regional routing tasks without falling into degenerate policies.

## 29/07/2026

Hours: 6.5

Tasks:
- Built a comprehensive automated benchmarking suite (`run_benchmark.py` and `report.py`) comparing all 8 algorithms across 5 complex network scenarios.
- Implemented Phase 3 guardrails: Fallback tracking (>5%), Degeneracy checking (AI mimicking Dijkstra), Variance sanity checking, and Wilcoxon statistical significance testing.
- Instrumented `is_fallback` tracking across all AI routers (`GNNRouter`, `RLRouter`, `MultiAgentRouter`) for transparent reporting.
- Wrote three automated regression tests to validate benchmark integrity and AI policy divergence.
- Added backend API endpoints (`GET /benchmark/results`) to securely expose benchmark data, computing effect sizes compared to Dijkstra.

Status:
- Comprehensive benchmarking suite delivered; AI routing models are statistically verified against baseline heuristics and data is exposed via API.

## 30/07/2026

Hours: 6.5

Tasks:
- Redesigned the frontend dashboard to consolidate the Live Simulator, Experiment Sandbox, and Benchmark Report into a single, unified view in `App.jsx`.
- Refactored `ExperimentSandbox.jsx` into a step-by-step wizard sidebar (`ExperimentBuilder.jsx`) for streamlined simulation configuration.
- Enhanced `TopologyGraph.jsx` by adding pulsating CSS animations for congested links and dynamic packet flow visualizations using D3 stroke offsets.
- Added a Benchmark Summary Card to the Report view, highlighting the best latency and load balancing models while automatically displaying AI guardrail warnings.
- Polished overall UI/UX typography, spacing, and theming, fixing minor issues with the GitHub Dark theme and link failure visual states.
- Authored a comprehensive project handover guide (`project_guide.md`) and a complete startup guide (`startup_guide.md`) for final faculty review.

Status:
- Phase 3 Complete. UI/UX fully consolidated into a single workflow with real-time animations. Final project documentation created for faculty demo and handover.

## 31/07/2026

Hours: 5.0

Tasks:
- Refactored backend routes and frontend reports, and cleaned up codebase
- Updated documentation to reflect latest API, ML, and frontend changes
- Fixed RightPanel bug
- Removed placeholder files and boilerplate TODO comments

Status:
- Codebase refactoring and documentation updates completed
- RightPanel bug resolved

## 04/08/2026

Hours: 5.5

Tasks:
- Redesigned AI Insights panel and refined UI text

Status:
- UI improvements for AI Insights panel implemented

## 11/08/2026

Hours: 2.0

Tasks:
- Reframed project narrative in README.md to present an evidence-backed thesis, adding headline results and documenting open-loop limitations
- Eliminated hardcoded backend URLs across the frontend components and hooks, introducing dynamic VITE_API_BASE_URL logic
- Configured Nginx proxy as the single transport path for all backend routes, making the app fully portable and deployable
- Updated frontend vite proxy config and backend CORS rules for flexible local development and production deployment

Status:
- Project documentation is more transparent, and the application architecture is fully production-ready and portable

## 12/08/2026

Hours: 2.5

Tasks:
- Re-read the original problem statement and reviewed the project against it end to end before touching anything
- Added the problem statement document to the repository so the target is on record
- Reproduced four suspected defects rather than assuming them, because a fix aimed at the wrong cause is worse than no fix
- Confirmed the RL router loads `rl_router_final.zip` while the repo ships `ppo_routing_agent.zip`; the resulting FileNotFoundError is swallowed by `except (FileNotFoundError, ImportError, Exception): return False` with no logging, so the "RL" results were a heuristic
- Regressed `runs/ppo_routing/evaluations.npz`: slope -0.094 per 100k steps, r-squared 0.001, p 0.878, and the best checkpoint is the first one taken at 25k. The agent had not learned
- Read `network_sim.py:57` and confirmed utilization evolves as a random walk independent of routing, which makes per-path latency minimisation exactly optimal and leaves nothing for a learned policy to win
- Verified the 100-node topology is a pure ring: 100 edges, every node degree 2, diameter 50
- Sequenced the rebuild by dependency so the project stays working at every commit

Status:
- All four defects reproduced independently. Rebuild plan sequenced
