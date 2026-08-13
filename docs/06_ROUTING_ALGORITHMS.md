# Routing Algorithms

The system implements multiple routing algorithms (including advanced ML-based and predictive approaches), each operating on the same `NetworkState` and producing a `RoutingDecision`. All algorithms share a common **congestion-aware edge weight formula**:

```
weight = base_latency × (1 + 4 × utilization²)
```

This makes highly-utilized links quadratically more expensive, encouraging all algorithms to prefer less congested paths.

---

## 1. Algorithm Summary

| Algorithm      | Time Complexity  | Type              | Strengths                             | File                          |
|----------------|-------------------|-------------------|---------------------------------------|-------------------------------|
| Dijkstra       | O((V+E) log V)    | Greedy / Optimal  | Fastest, guaranteed optimal for non-negative weights | `router/dijkstra.py`   |
| Bellman-Ford   | O(VE)              | Dynamic Programming| Handles negative weights, detects cycles | `router/bellman_ford.py` |
| ACO            | O(iterations × ants × V) | Metaheuristic | Multi-path exploration, adaptive | `router/aco.py`              |
| RL (PPO)       | O(inference)       | Learned Policy    | Adapts to traffic patterns over time  | `router/rl_agent.py`         |
| GNN            | O(inference)       | Learned Policy    | Understands topology, balances load   | `router/gnn_router.py`       |
| MARL           | O(inference)       | Multi-Agent       | Distributed routing across regions    | `router/multi_agent_router.py`|
| Predictive     | O(LSTM + inference)| Forecasting       | Avoids congestion before it happens   | `api/routers/network.py`     |

RL and GNN might provide better results for larger networks with more routers.
---

## 2. Dijkstra's Algorithm

**File**: `backend/router/dijkstra.py`

### How It Works

1. Build an adjacency list from the current `NetworkState` with congestion-aware weights
2. Initialize a min-heap with the source node at cost 0
3. Repeatedly expand the cheapest unvisited node
4. When a neighbor has a lower cost via the current node, update it and push to the heap
5. Stop when the destination is reached
6. Reconstruct the path by following parent pointers backward

### Key Properties

- **Optimal**: Always finds the minimum-cost path for non-negative weights
- **Efficient**: Uses a binary heap for O((V+E) log V) time complexity
- **Deterministic**: Same input always produces the same output
- **Limitation**: Does not explore alternative paths — only returns the single best

### Code Flow

```python
find_route(state, src, dst) → RoutingDecision
├── _build_adjacency(state)              # Build weighted adjacency list
├── Dijkstra loop (heap-based)           # Find shortest path
├── _reconstruct_path(previous, src, dst) # Backtrack through parents
└── Return RoutingDecision with path and cost
```

---

## 3. Bellman-Ford Algorithm

**File**: `backend/router/bellman_ford.py`

### How It Works

1. Build a directed edge list (each undirected link → two directed edges)
2. Initialize all distances to infinity, source to 0
3. Relax all edges V-1 times
4. Early termination if no distances change in a pass
5. Extra pass to detect negative-weight cycles (safety check)
6. Reconstruct path via predecessor pointers

### Key Properties

- **Correct for negative weights**: Although this project uses non-negative congestion costs, the algorithm is included for educational completeness
- **Cycle detection**: Built-in negative cycle check (returns failure if detected)
- **Slower**: O(VE) vs Dijkstra's O((V+E) log V)
- **Use case**: Models distributed routing where nodes only know neighbors (distance-vector routing)

### Why Include It?

Bellman-Ford models how real-world distributed routing protocols (like RIP) work — routers repeatedly share distance vectors with neighbors. Including it alongside Dijkstra demonstrates the performance trade-off and validates that both produce identical paths in this non-negative-weight network.

---

## 4. Ant Colony Optimization (ACO)

**File**: `backend/router/aco.py`

### How It Works

1. Initialize pheromone levels on all links to 1.0
2. For each iteration:
   a. Each ant constructs a path from source to destination
   b. Next-hop selection is probabilistic, weighted by pheromone and inverse cost
   c. After all ants complete, evaporate pheromone on all edges
   d. Deposit new pheromone on paths proportional to path quality (Q/cost)
3. Track the best path found across all iterations and ants

### Configuration Parameters

| Parameter         | Default | Description                                       |
|-------------------|---------|---------------------------------------------------|
| `alpha`           | 1.0     | Pheromone influence exponent                      |
| `beta`            | 2.0     | Heuristic (cost) influence exponent               |
| `evaporation_rate`| 0.2     | Fraction of pheromone that evaporates per iteration|
| `Q`               | 100     | Pheromone deposit constant                        |
| `n_ants`          | 20      | Number of ants per iteration                      |
| `n_iterations`    | 30      | Number of ACO iterations                          |

### Next-Hop Selection Formula

For ant at node `i`, probability of moving to neighbor `j`:

```
          τ(i,j)^α  ×  η(i,j)^β
P(j) = ─────────────────────────────
        Σ_k  τ(i,k)^α  ×  η(i,k)^β
```

Where:
- `τ(i,j)` = pheromone level on edge (i,j)
- `η(i,j)` = 1 / cost(i,j) — heuristic desirability
- `α` controls pheromone influence
- `β` controls cost influence

### Key Properties

- **Exploratory**: Discovers multiple alternative paths
- **Adaptive**: Pheromone evaporation allows the algorithm to adapt when traffic changes
- **Stochastic**: May return different paths on different runs (seeded with `Random(42)` for reproducibility in testing)
- **Good for multi-path routing**: In a real system, could split traffic across multiple paths

---

## 5. Reinforcement Learning (PPO)

**File**: `backend/router/rl_agent.py`

### How It Works

The RL router uses a trained PPO (Proximal Policy Optimization) agent to select the best path from a set of candidate paths.

1. Find up to K=5 candidate paths between source and destination (BFS-based)
2. Build a flat observation vector from the current network state
3. Feed the observation to the PPO model for deterministic inference
4. Map the model's discrete action to a candidate path index
5. Return the selected path as a `RoutingDecision`

### Model Architecture

- **Policy**: MlpPolicy (2 hidden layers of 64 units each)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3 2.3.2

### Observation Space

Flat vector of shape `(n_links × 4,)`, normalized to [0, 1]:

| Feature Index | Feature                     | Normalization          |
|---------------|-----------------------------|------------------------|
| i*4 + 0       | Link utilization            | Already 0–1            |
| i*4 + 1       | Queue size                  | Divided by 100         |
| i*4 + 2       | Packet loss rate            | Divided by 0.06        |
| i*4 + 3       | Base latency                | Divided by 25          |

### Fallback Behavior

If the PPO model file (`backend/ml/models/rl_router_final.zip`) is not found:
- The router falls back to a **congestion-aware heuristic** that selects the candidate path with the lowest total congestion-adjusted cost
- This ensures the API never fails, even before training
- A warning is printed to console but no error is raised

### Model Loading

```python
router = RLRouter()
router.try_load_model()           # Returns False if model missing
decision = router.predict(state, "R1", "R5")
```

---

## 6. Graph Neural Network (GNN)

**File**: `backend/router/gnn_router.py`

### How It Works
The GNN router uses a PyTorch-based Graph Neural Network that inherently understands the network topology. It processes the current network state to predict path qualities, focusing on minimizing congestion and maximizing load balancing.

---

## 7. Multi-Agent Reinforcement Learning (MARL)

**File**: `backend/router/multi_agent_router.py`

### How It Works
Instead of a single global agent, the network is partitioned into distinct regions, each governed by its own PPO agent. This decentralized approach (using naive self-play for training) models distributed routing, where regions cooperate implicitly via a shared global reward signal.

---

## 8. Predictive Routing Mode

**File**: Implemented via API (`backend/api/routers/network.py`) and Benchmark (`backend/benchmark/run_benchmark.py`)

### How It Works
Predictive routing is not a standalone pathfinding algorithm; rather, it is a mode that enhances the RL or GNN routers.
1. The **LSTM Congestion Predictor** takes historical utilization data and forecasts the *future* network state.
2. This predicted state is fed to the RL (`rl_predictive`) or GNN (`gnn_predictive`) router.
3. The router makes a decision based on the anticipated congestion, avoiding links before they actually become congested.

---

## 9. Algorithm Comparison

All four algorithms are compared via `POST /network/route/compare`:

```
Typical results (step 150, R1 → R5):

Algorithm      Path              Latency    Utilization
─────────────────────────────────────────────────────────
Dijkstra       R1→R3→R5          34.7 ms    0.35
Bellman-Ford   R1→R3→R5          34.7 ms    0.35
ACO            R1→R4→R7→R5       52.1 ms    0.28
RL (PPO)       R1→R3→R5          34.7 ms    0.35
GNN            R1→R2→R4→R5       42.0 ms    0.22
MARL           R1→R3→R5          34.7 ms    0.35
Predictive     R1→R4→R7→R5       50.1 ms    0.20
```

### Observations

- **Dijkstra and Bellman-Ford** typically produce identical paths (both find the true shortest path in non-negative networks)
- **ACO** sometimes finds alternative paths that avoid high-utilization links, trading latency for resilience
- **RL** often matches Dijkstra when well-trained, but may choose different paths under heavy congestion due to its learned policy

---

## 10. Common Data Models

All algorithms return a `RoutingDecision` dataclass:

```python
@dataclass
class RoutingDecision:
    source: str            # e.g., "R1"
    destination: str       # e.g., "R5"
    path: List[str]        # e.g., ["R1", "R3", "R5"]
    algorithm: str         # e.g., "dijkstra"
    total_latency: float   # Total path cost in ms
    avg_utilization: float # Mean utilization of path links
    success: bool          # True if a path was found
```

Failed routes return `path=[]`, `total_latency=inf`, `success=False`.
