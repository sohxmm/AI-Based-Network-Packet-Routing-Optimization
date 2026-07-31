# Testing

---

## 1. Test Overview

| Test File                         | Type           | Scope                                   |
|-----------------------------------|----------------|------------------------------------------|
| `backend/test_integration.py`     | Integration    | End-to-end routing across algorithms     |
| `backend/test_gnn.py`             | Integration    | Validates GNN functionality and tests    |
| `backend/test_stress_phase1.py`   | Stress         | Simulator stability under load           |
| `backend/router/test_all_routers.py` | Unit        | All routing algorithms                   |
| `backend/router/test_stress_aco.py`  | Stress      | ACO under heavy congestion               |
| `backend/router/test_stress_dijkstra.py` | Stress  | Dijkstra under heavy congestion          |
| `backend/tests/test_predictive_routing.py`| Integration| Verifies predictive routing pipeline end-to-end |
| `backend/tests/test_multi_agent_routing.py`| Unit  | Compares single-agent vs MARL success rates |
| `backend/tests/test_benchmark_*.py` | Benchmark  | Algorithm differentiation and fallback logic |
| `frontend/src/components/__tests__/`| Unit (Vitest/Jest) | Frontend UI component tests (e.g., BenchmarkReport) |

---

## 2. Running Tests

### Integration Test

```bash
cd backend
python test_integration.py
```

**What it does:**
1. Creates a `NetworkSimulator` with seed 42
2. Runs 25 simulation steps
3. On each step, selects a random (source, destination) pair
4. Runs all 4 algorithms (Dijkstra, Bellman-Ford, ACO, RL) on the same pair
5. Reports success rate and average latency per algorithm
6. **Asserts overall success rate ≥ 95%**

**Expected output:**
```
Algorithm      Success Rate  Avg Latency
--------------------------------------------
dijkstra          100.0%       38.42 ms
bellman_ford      100.0%       38.42 ms
aco               100.0%       44.18 ms
rl                100.0%       38.42 ms
--------------------------------------------
Overall success: 100.0%
```

### Stress Tests

```bash
cd backend

# Phase 1 simulator stress test
python test_stress_phase1.py

# Algorithm-specific stress tests
python -m router.test_all_routers
python -m router.test_stress_aco
python -m router.test_stress_dijkstra
```

### Advanced Routing & Pytest Suites

```bash
cd backend
python -m pytest tests/test_predictive_routing.py -v
python -m pytest tests/test_multi_agent_routing.py -v
python -m pytest tests/test_benchmark_algorithm_differentiation.py -v
python test_gnn.py
```

### Frontend Testing
```bash
cd frontend
npm test
```

---

## 3. What Each Test Validates

### `test_integration.py`

- All algorithms return valid `RoutingDecision` objects
- Success rate stays above 95% threshold
- No crashes or exceptions during routing
- Algorithm outputs contain expected fields

### `test_stress_phase1.py`

- Simulator runs for many steps without crashing
- Link utilization stays within [0, 1] bounds
- Queue sizes remain non-negative
- Packet loss rate stays within expected range
- Congestion bursts trigger and resolve correctly
- Link failure injection and restoration work correctly
- State remains consistent after reset

### `test_stress_aco.py`

- ACO finds valid paths under normal conditions
- ACO handles high-congestion scenarios
- Pheromone evaporation works correctly
- ACO respects visited-node constraints (no cycles)
- Performance remains stable over many iterations

### `test_stress_dijkstra.py`

- Dijkstra finds optimal paths in known topologies
- Dijkstra handles disconnected graphs gracefully (returns failure)
- Edge weights correctly reflect congestion
- Path reconstruction produces valid sequences

---

## 4. Manual Testing Checklist

For a full system verification:

### Backend

- [ ] `docker compose ps` — Both containers running
- [ ] `curl http://localhost:8000/health` → `{"status": "ok"}`
- [ ] `curl http://localhost:8000/network/state` → JSON with nodes/links
- [ ] `curl -X POST http://localhost:8000/sim/step` → State advances
- [ ] `curl -X POST http://localhost:8000/network/route/compare -H "Content-Type: application/json" -d '{"source":"R1","destination":"R5"}'` → Comparison results
- [ ] WebSocket: Connect to `ws://localhost:8000/ws/stream` and verify `state_update` messages arrive

### Frontend

- [ ] Dashboard loads at `http://localhost:5173`
- [ ] Header shows "Live stream connected" (green badge)
- [ ] TopologyGraph displays nodes and links with color
- [ ] CongestionHeatmap shows bar chart with utilization data
- [ ] Route Comparison: Select R1→R5, click "Compare All", results appear
- [ ] Control Panel: "Step +1" advances simulation (step count increases)
- [ ] Control Panel: "Reset" returns to step 0
- [ ] Theme selector changes colors across all components
- [ ] Link failure: Select a link, click "Inject Failure", topology updates
- [ ] Link restore: Click "Restore Link", link reappears

---

## 5. Adding New Tests

When adding new routing algorithms or features, follow this pattern:

```python
# In backend/test_integration.py or a new test file:

from simulator.network_sim import NetworkSimulator

def test_new_feature():
    simulator = NetworkSimulator(seed=42)
    
    for _ in range(10):
        state = simulator.step()
        # Test your feature against the current state
        result = your_function(state, "R1", "R5")
        assert result.success, f"Failed at step {state.step_count}"
        assert result.total_latency > 0
```

---

## 6. Known Test Limitations

- **No CI/CD pipeline**: Tests must be run manually
- **Database-dependent tests**: Some API endpoints require a running PostgreSQL instance; tests that don't need DB skip it gracefully
- **Migration to Pytest**: Some older tests still use plain `assert` statements and `if __name__ == "__main__"`, though all new test suites in `backend/tests/` fully utilize `pytest`.
