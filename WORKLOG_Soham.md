# Internship Work Log

Total hours: 192.5

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

## 13/08/2026

Hours: 6.5

Tasks:
- Restructured the repository into core / routing / ml / experiments / service / web, using `git mv` throughout so history is preserved
- Removed all seven `sys.path.insert(...)` hacks; `pytest.ini` sets pythonpath and the Docker image sets PYTHONPATH
- Built the domain core: `core/cost.py`, `core/models.py`, `core/paths.py`, `core/qos.py`, `core/simulator.py`, `core/sources.py`
- The congestion cost formula existed at 14 sites across 11 files with three different exponents among them, which meant the RL agent trained against one cost and served against another. It now exists at exactly one site
- Corrected two behaviours while consolidating rather than porting them faithfully: `path_cost` returns infinity for an invalid path (the old copies filtered out missing edges and under-reported the cost of a broken path), and `candidate_paths` is congestion-weighted by default, which fixes a train/serve skew hidden behind a boolean default
- Closed the simulation loop: `register_flow()` means routing decisions now add load to the links they traverse
- Hit a calibration bug worth recording. Adding the flow term as a shock outside the AR(1) update amplified it by 1/(1-a) at steady state, pinning mean utilization at 0.68 with p95 at 1.0 — everything saturated and the cost function degenerate again. Fixed by folding flow load into the offered-load baseline the process mean-reverts toward. Mean now sits at 0.39
- Replaced the ring with a small-world topology: 100 nodes is now 200 edges, average degree 4, diameter 8
- Replaced the random-walk utilization process with AR(1) around a per-link diurnal cycle, because the Bayes-optimal one-step predictor for a random walk is the identity function and the LSTM was being trained to learn f(x) = x
- Added `core/qos.py` with five traffic classes carrying hard constraints, and `core/sources.py` so the same stack can be driven by the simulator, a recorded trace or live measurement
- Removed roughly 350 duplicated lines: 6 copies of `_average_path_utilization`, 6 of `_failed_decision`, 4 of `_path_cost`, 3 byte-identical `_find_candidate_paths`, 2 dead `_build_adjacency`

Status:
- The definition of "good routing" now lives in one place, and the project's central question is answerable for the first time

## 14/08/2026

Hours: 6.5

Tasks:
- Rebuilt the routing layer on the shared primitives: one `Router` protocol, one `build_router_set()`, eight algorithms across `classical/`, `heuristic/` and `learned/`
- There had been three different method names across six routers, so every call site needed its own dispatch table and adding an algorithm meant editing five files
- Added `routing/random_baseline.py` as the floor. A reward number or an accuracy percentage conveys nothing without knowing what random guessing scores
- Added `routing/classical/constrained.py` as the ceiling: k-shortest paths filtered by QoS feasibility, plus `qos_oracle()` and `qos_floor()`
- Added `routing/failover.py`: reroute detection on watched flows, and `measure_convergence()` which cuts a link and counts ticks to a QoS-satisfying route, reporting latency before and after so a fast-but-worse recovery is distinguishable from a slow-but-better one
- Created `ml/model_registry.py` as the single source of truth for artifact paths, read by both training and serving so they cannot diverge again
- Created `ml/features.py` and `ml/local_features.py` so the environment and the router build byte-identical observations
- Rewrote every `try_load_model` so a missing artifact logs at WARNING with the exact training command instead of returning False in silence
- Found a second bug while fixing the first: `GNNRouter.load_model` assigned `self._model` before `load_state_dict`, so a failed load left a randomly-initialised model installed with `is_trained` reporting True. The router would have served pure noise while claiming to be trained
- Made learned routers import torch lazily, so `import routing` costs milliseconds and works on a machine with no ML stack installed

Status:
- Silent model-loading failures are now structurally impossible, and every algorithm answers the same interface

## 17/08/2026

Hours: 3.5

Tasks:
- Rebuilt the RL environments so the task is actually learnable
- The old observation encoded per-link features only while the environment resampled (src, dst) every step. The agent was asked to pick "path index 2" without being told which pair it was routing, and the meaning of index 2 changed between steps. That is not a partially observable MDP, it is an unobservable one, and it fully explains the flat evaluation curve
- New observation is 286 dimensions in four blocks: link state, the task as one-hot source and destination, per-candidate features, and the QoS class
- Fixed the reward and observation ordering so both describe the same decision
- Made each episode start from a freshly seeded simulator; the old one never reset, so utilization random-walked to the boundaries and the agent trained on states it would never serve
- Registered the chosen path before the tick so the global load term has real gradient. Previously it was computed over all links independent of the action, which in policy-gradient terms is a pure state-dependent baseline contributing exactly zero
- Rebuilt the regional environment for genuine decentralised execution: a local observation of constant width, and a single next-hop action instead of a complete end-to-end path

Status:
- There is a real signal to learn, measured at random -45.2 against oracle -31.9 on a 200-step episode

## 18/08/2026

Hours: 6.0

Tasks:
- Rewrote all four training pipelines and the benchmark harness
- GNN: changed message aggregation from unnormalised sum to mean, because embedding magnitudes were scaling with node degree; replaced node-mean path pooling, which was permutation invariant, length invariant and edge blind, so the model was asked to predict a path's cost while being shown neither its length nor its links; switched from MSE to a pairwise margin ranking loss, since the output is only ever consumed by argmin
- Gave the GNN a genuinely independent validation set from a different seed. The old one continued the same simulator instance, so the reported "72% val MSE reduction" was not evidence of generalisation
- Trained it: held-out top-1 accuracy 0.978 against 0.227 for random, mean regret 0.0006 against 0.676
- LSTM: wrote a real pipeline with a chronological 70/15/15 split and a persistence baseline. First run scored a skill score of -1.77 and the script refused to save the checkpoint, which is the behaviour I wanted
- Diagnosed it: predicting the utilization *level* means competing with persistence, which on a strongly autocorrelated series is right to within the one-step noise almost every time. The network spent its capacity relearning the identity. Rewrote it to predict the residual, differencing the input window and re-attaching the level on output. Second run scored +0.1497 and saved
- Rewrote the scenarios as declarative dataclasses. `high_congestion` no longer adds +0.4 cumulatively every step, which saturated everything within 10 steps and collapsed the ranking to base-latency order, and link failures now persist instead of re-randomising every tick
- Added `cascading_failure` and `qos_mixed_traffic`
- Added `experiments/statistics.py` and `ml/evaluation/baselines.py`: Cliff's delta, Wilcoxon, bootstrap CIs, path entropy, and random / greedy / oracle policies for normalising a raw return

Status:
- Every model beats a stated baseline on its own task, and every scenario stresses what its name claims

## 19/08/2026

Hours: 6.0

Tasks:
- Rebuilt the service layer around the new packages
- Split the monolithic route module into `network`, `simulator`, `metrics`, `benchmark`, `experiments`, `websocket` and `dispatch`
- The simulator loop now survives exceptions, logs them with a stack trace, backs off after ten consecutive failures, and `/health` reports the age of the last successful tick. It previously caught only `CancelledError`, so any other exception killed the task while the app stayed up and `/health` kept returning `{"status": "ok"}` with the dashboard frozen
- Added `GET /health/models`, which answers "is the AI actually running?" in one request and reports file presence separately from load success
- Made the network source swappable at runtime, and made the benchmark harness build its own isolated router set. Running a sandbox experiment used to permanently shift the live dashboard's ACO pheromone table
- Capped `/metrics/history`, which took an unbounded limit
- Added hard caps to the experiment sandbox that reject rather than clamp, because silently clamping returns results that do not match what was asked for
- Made snapshot writes fire-and-forget and every tenth tick; they were awaited inside the 1 Hz loop, so a slow database stalled the simulation itself, at roughly 860 MB/day unbounded

Status:
- A dead background loop is now externally detectable instead of hiding behind a green check

## 20/08/2026

Hours: 6.5

Tasks:
- Reworked the dashboard
- Added the missing ESLint config. `npm run lint` had never been able to run: four eslint plugins were installed and no configuration file existed anywhere in the repo. It now exits 0 at `--max-warnings 0`
- Added a shared `extractApiError`; structured backend errors were rendering as "[object Object]" and a non-JSON 502 from nginx threw a parse error over the real problem
- Added an `isStale` watchdog to the WebSocket hook, because a backend whose simulator loop has died looks identical to an idle one from the client's side
- Removed the hardcoded `http://localhost:8000`, which meant any non-localhost deployment required a source edit
- Memoized the tick-driven panels and split `BenchmarkResultView` from 543 lines to 90
- Built the path divergence view: every algorithm's route overlaid on one topology, perpendicular-offset so shared edges stay visible, fallback routes dashed, with a per-hop cost breakdown so the total is auditable rather than asserted
- Added a model status banner and a warnings callout, so a user reading results always knows whether a model was behind them
- Replaced the chart palette after checking it: the original 8 series colours failed colour-vision separation, with a worst adjacent pair at delta-E 3.0, and one read as grey
- Cut the theme engine from ten themes to four. Ten cost about what fixing the model-loading bug cost, and the project needed the second one more

Status:
- The dashboard now shows the argument, not just the network

## 21/08/2026

Hours: 6.5

Tasks:
- Made the test suite installable. There was no pytest in any requirements file, no `pytest.ini`, no `conftest.py`, and two of the eleven "test files" contained zero test functions
- Wrote `test_train_serve_parity.py`, asserting the environment and the router build byte-identical observations
- Wrote `test_marl_locality.py`, which perturbs only the global observation block and asserts the actor's action distribution does not move while the critic's value estimate does. That is the decentralised-execution claim, verified rather than described
- Wrote `test_closed_loop.py`, including that round-robin over three paths keeps the worst link cooler than saturating one path. That load-balancing property is what the whole project is about and was previously unmeasurable
- Added honesty gates for degeneracy, silent fallbacks, structurally constant metrics, p-values of exactly zero, missing effect sizes and ring topologies
- Wrote `scripts/verify_claims.py`, which cross-checks documented numbers against committed artifacts. This is the check that would have caught the model filename mismatch on day one
- Added CI with four jobs, including a Docker smoke test that asserts the models really loaded
- Two-stage Dockerfile with no compiler in the runtime image and a non-root user; made `env_file` optional so a fresh clone actually starts. Three services declared `env_file: .env`, `.env` is gitignored, and nothing created it, so the very first command anyone ran failed
- Moved pgAdmin behind a `dev` profile, because a tool with a default password should not appear on a port by accident, and added a Makefile so first run is one command
- A legacy test caught a real bug during the port: routing a node to itself returned infinity instead of 0, because a one-node path has no links and "no links" was being conflated with "invalid"

Status:
- `git clone && make up` works from nothing, and the guardrails now enforce rather than merely report

## 22/08/2026

Hours: 4.5

Tasks:
- Fixed the multi-agent checkpoint reload. Stable-Baselines3 rebuilds both the policy and value feature extractors from identical kwargs, so `PPO.load` reconstructed the actor with the critic's 129-dimensional input and failed with a size mismatch
- Solved it by encoding the asymmetry in a policy class rather than applying it after construction, so it is part of what gets serialized. Verified the save/load roundtrip keeps the actor local-only and the critic global
- Fixed the partition, which was derived from a freshly built 25-node simulator regardless of the topology being served. On the 100-node scenario every node above R25 mapped to region -1 and forced a fallback, which is where the measured `fallback_rate = 0.75` came from
- Fixed the degeneracy metric. Under per-algorithm closed-loop trajectories the networks legitimately diverge by step two, so comparing chosen paths measures trajectory divergence rather than algorithmic similarity — measured that way even Bellman-Ford scored 0.00 against Dijkstra, which is impossible for two exact solvers. It now runs as a separate shared-state open-loop probe
- Wrote four model cards reporting use-matched metrics and known failure modes, plus an architecture diagram

Status:
- The decentralised-execution claim is supported by the code rather than asserted in a docstring

## 24/08/2026

Hours: 6.0

Tasks:
- Rebuilt the benchmark on independent replications
- The old harness ran a paired Wilcoxon over 20,000 observations from a *single* trajectory. Successive steps are heavily autocorrelated, so that is roughly one independent observation repeated, and with that much pseudo-replication any difference becomes significant. It is why every result file reported p = 0.0, which is numerical underflow rather than a p-value
- The unit of replication is now one seeded run, with each algorithm on its own closed-loop trajectory. That is mandatory once routing changes the network, or whichever ran first would pollute the state the others observe
- Added Cliff's delta and a bootstrap 95% CI, and removed `effect_size_pct`, which was a raw percent difference in means and is not an effect size
- Fixed `diversity_index`, which read a `path` key that was never stored and was therefore 0.000 in every committed file. The metric that would have shown whether the AI explores alternatives had never once worked
- Fixed `max_path_utilization`, which took a max over 20,000 samples and was therefore 1.000 for every algorithm in every scenario
- Seeded every source of randomness; the old harness used the unseeded global `random` module in seven places and had no `--seed` flag
- Introduced Alembic. Tables were created with `create_all`, which can only CREATE and never ALTER, so adding a column to an existing deployment silently did nothing
- Dropped `PacketLog`, which had zero references anywhere, activated `AlgorithmMetric`, whose rows were built and thrown away behind a commented-out commit, and added the `avg_utilization` column that `congestion_events` had been reading without it existing
- Started the learning guide: background, the system, and all eight algorithms written up in full

Status:
- The benchmark produces evidence rather than numbers
