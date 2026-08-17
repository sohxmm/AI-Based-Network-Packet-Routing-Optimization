# System architecture

Thirteen documents contained zero pictures. This is the picture.

## Layered view

```mermaid
flowchart TB
  subgraph Browser["Browser"]
    UI["React dashboard<br/>topology · divergence · benchmark · lab"]
  end

  subgraph Edge["nginx"]
    PROXY["Static files<br/>+ /network /sim /metrics /benchmark /experiments /ws"]
  end

  subgraph Service["service/ — FastAPI"]
    API["REST routers"]
    WS["WebSocket manager"]
    LOOP["1 Hz tick loop<br/>crash-proof, liveness-reported"]
    STATE["AppState singletons"]
    FAIL["Failover monitor"]
  end

  subgraph Sources["core/sources.py — pluggable"]
    SIM["NetworkSimulator<br/><b>closed loop</b>"]
    TRACE["Trace replay<br/>JSONL / CSV"]
    LIVE["Live ICMP probe<br/>opt-in, read-only"]
  end

  subgraph Core["core/ — domain, no web, no ML"]
    COST["link_cost<br/><b>defined once</b>"]
    PATHS["candidate_paths · path_cost"]
    QOS["QoS classes<br/>multi-constrained scoring"]
  end

  subgraph Routing["routing/"]
    CL["classical/<br/>Dijkstra · Bellman-Ford · constrained"]
    HE["heuristic/<br/>ACO"]
    LE["learned/<br/>GNN · PPO · multi-agent · forecaster"]
  end

  subgraph ML["ml/ — training only"]
    ARCH["architectures/"]
    ENVS["environments/"]
    TRAIN["training/"]
    CK[("checkpoints/<br/>committed, ~1 MB")]
  end

  subgraph Exp["experiments/ — not part of the service"]
    SCEN["scenarios.py"]
    RUN["runner.py<br/>N seeded replications"]
    STATS["statistics.py<br/>Cliff's δ · bootstrap CI"]
    RES[("results/")]
  end

  DB[("PostgreSQL<br/>events · snapshots · metrics")]

  UI --> PROXY --> API & WS
  API --> STATE --> Routing
  LOOP --> Sources
  Sources --> Core
  Routing --> Core
  LE --> CK
  TRAIN --> CK
  TRAIN --> ENVS --> Core
  RUN --> Routing & SCEN & STATS --> RES
  API --> RES
  LOOP -. "every 10th tick" .-> DB
  API --> DB
  WS -. "1 Hz broadcast" .-> UI
  LOOP --> FAIL
```

**The one thing to notice:** `core/` has no arrow pointing into the web service,
the ML stack or the benchmark. It is the scientific core, and it is deliberately
usable on its own. The previous layout buried it inside `backend/`, where it
looked like an implementation detail of a web application.

## The closed loop

This is the change that makes the project's central question answerable at all.

```mermaid
sequenceDiagram
    participant U as Dashboard
    participant A as API
    participant R as Router
    participant S as Simulator

    U->>A: POST /network/route (R1 → R14, class=emergency)
    A->>S: get_state()
    S-->>A: NetworkState
    A->>R: find_route(state, R1, R14, profile)
    R-->>A: RoutingDecision (+ QoS evaluation)
    A->>S: register_flow(path)
    Note over S: The chosen links get busier.<br/>The next decision sees it.
    A-->>U: decision + per-hop cost breakdown
    S->>U: state_update (1 Hz, over WebSocket)
```

Before this, `register_flow` did not exist. Utilization evolved as a random walk
independent of routing, so no decision anyone made changed anything — which made
per-path latency minimisation exactly optimal and Dijkstra unbeatable by
construction.

## Where a routing decision comes from

```mermaid
flowchart LR
    D["Demand<br/>(src, dst, class)"] --> C["candidate_paths()<br/>k=5, congestion-weighted"]
    C --> DJ["Dijkstra<br/>argmin additive cost"]
    C --> CN["Constrained<br/>feasible-first, then cost"]
    C --> AC["ACO<br/>pheromone sampling"]
    C --> GN["GNN<br/>edge-pooled ranker"]
    C --> RL["PPO<br/>action = candidate index"]
    D --> MA["Multi-agent<br/>hop-by-hop, local obs"]
    DJ & CN & AC & GN & RL & MA --> E["evaluate_path()<br/>score + feasibility"]
    E --> OUT["RoutingDecision<br/>path · cost · is_fallback · QoS"]
```

Every candidate-based router draws from the **same** generator, in the **same**
order. Three different generators used to exist — training ordered candidates by
congestion-adjusted cost, inference by hop count on an unweighted graph, and the
GNN trainer by raw BFS order — so action index *k* meant a different path in
each. That is invisible when it happens and it invalidates every learned result.
