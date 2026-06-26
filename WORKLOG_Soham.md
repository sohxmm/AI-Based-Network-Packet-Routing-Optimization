# Internship Work Log

Total hours: 45.5

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
- Latest completed task: Phase 1 and Phase 2 validation passed, and current REST API scaffold was smoke-tested successfully
- Phase 3 is partially complete; WebSocket streaming and DB-backed metrics/history remain pending

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
- Latest completed task: WebSocket streaming and automatic simulator background execution completed
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

Hours: 5.0

Tasks:
- Connected frontend to WebSocket live network stream
- Built route comparison flow using all routing algorithms
- Added topology path highlighting and congestion heatmap visualization

Status:
- Dashboard can display live network state and fastest route results

## 26/06/2026

Hours: 4.5

Tasks:
- Implemented LSTM congestion predictor module and forecast API scaffold
- Added integration test for 100 routing decisions across all algorithms
- Verified backend compile, integration test and frontend production build

Status:
- Demo-ready simulator, routing and dashboard integration prepared
