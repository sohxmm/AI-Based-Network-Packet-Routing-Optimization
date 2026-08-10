# Future Improvements (Phase 2 Roadmap)

This document outlines recommended improvements and feature additions for the Phase 2 development team.

---

## 1. High Priority

### 1.1 Authentication & Authorization

- Add JWT-based authentication for API endpoints
- Protect WebSocket connections with token validation
- Add role-based access control (admin vs. viewer)

### 1.2 Database Migrations with Alembic

- Initialize Alembic for versioned schema migrations
- Create migration scripts for existing tables
- Enable zero-downtime schema evolution

### 1.3 Environment Variable Externalization

- Move hardcoded URLs (frontend API base, WebSocket port) to environment variables
- Use Vite's `import.meta.env` for frontend configuration
- Create separate `.env.production` templates

### 1.4 Data Retention & Cleanup

- Add configurable retention policies for `network_snapshots` and `routing_events`
- Implement periodic cleanup jobs (e.g., keep last 24 hours of snapshots)
- Add database size monitoring

---

## 2. Medium Priority

### 2.1 Improved RL Agent

- **Larger observation space**: Include source/destination as part of the observation
- **Curriculum learning**: Start with simple topologies and gradually increase complexity
- **Transformer-based models**: Investigate attention mechanisms for capturing long-range dependencies in large topologies
- **Reward shaping**: Experiment with reward functions that better capture routing quality

### 2.2 Packet-Level Simulation

- Implement the `PacketAnimator` component with actual packet flow visualization
- Populate the `packet_logs` database table with simulated packet transmissions
- Add packet delivery success/failure tracking per route

### 2.3 Advanced Dashboard Features

- **Historical trends**: Line charts showing latency/utilization over time
- **Algorithm leaderboard**: Persistent ranking based on database metrics
- **Network health score**: Composite metric combining utilization, loss, and congestion
- **Alert system**: Visual notifications when congestion exceeds thresholds
- **Topology editor**: Allow users to add/remove routers and links

### 2.4 CI/CD Pipeline

- Add GitHub Actions workflow for:
  - Python linting (ruff/flake8)
  - Frontend linting (eslint)
  - Unit tests
  - Integration tests
  - Production build verification
- Add pre-commit hooks

---

## 3. Nice to Have

### 3.1 Docker Full-Stack Deployment

- Create Dockerfiles for backend and frontend
- Extend `docker-compose.yml` to include all services
- Add Nginx reverse proxy container
- Health check endpoints for all services

### 3.2 Multi-Topology Support

- Allow users to load different network topologies (mesh, star, tree, etc.)
- Import/export topologies as JSON
- Configurable number of nodes and link density

### 3.3 Real-Time Algorithm Racing

- Run all algorithms simultaneously on every simulator tick
- Show live performance comparison over time (not just per-request)
- Build an algorithm performance dashboard

### 3.4 Traffic Pattern Simulation

- Support different traffic patterns (uniform, hotspot, bursty)
- Configurable traffic intensity and distribution
- Simulate time-of-day traffic variation

### 3.5 API Rate Limiting

- Add rate limiting middleware to prevent abuse
- Configurable limits per endpoint
- Return proper `429 Too Many Requests` responses

### 3.6 Logging & Monitoring

- Structured logging (JSON format) with configurable levels
- Application performance monitoring (APM)
- Database query performance tracking
- WebSocket connection metrics

### 3.7 Frontend Testing

- Add React Testing Library unit tests
- Add Playwright/Cypress end-to-end tests
- Visual regression testing for theme changes

### 3.8 API Documentation

- Enable FastAPI's built-in Swagger UI at `/docs`
- Add OpenAPI descriptions to all endpoints
- Generate SDK clients from the OpenAPI spec

---

## 4. Technical Debt

| Item                                   | Effort | Impact |
|----------------------------------------|--------|--------|
| Fix `datetime.utcnow()` deprecation   | Low    | Low    |
| Cache `RLRouter`/`AntColonyRouter` instances | Low | Medium |
| Fix `RightPanel.jsx` capacity field    | Low    | Low    |
| Add pytest configuration               | Low    | Medium |
| Replace hardcoded API URLs with env vars| Medium | High  |
| Add Alembic migrations                 | Medium | High   |
| Implement `PacketAnimator`             | Medium | Low    |
| Add authentication                      | High   | High   |
