# ML & AI Components

The project uses four machine learning approaches: an **LSTM congestion predictor** for time-series forecasting, a **PPO reinforcement learning agent**, a **Graph Neural Network (GNN)**, and a **Multi-Agent RL (MARL)** system for adaptive routing.

---

## 1. LSTM Congestion Predictor

**File**: `backend/ml/congestion_lstm.py`

### Purpose

Predicts the next-step link utilization values across the network, enabling proactive congestion avoidance. The model takes a sliding window of recent utilization snapshots and outputs a predicted utilization vector for the next simulator tick.

### Architecture

```
Input: (batch_size, seq_len=10, n_links=20)
    │
    ▼
┌─────────────────────────────────┐
│  LSTM (2 layers, hidden=64)     │
│  dropout=0.2 between layers     │
│  batch_first=True               │
└──────────┬──────────────────────┘
           │ (batch_size, hidden=64)
           ▼
┌─────────────────────────────────┐
│  Dropout (0.2)                  │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Linear (64 → n_links=20)      │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Sigmoid (output clamped 0–1)   │
└─────────────────────────────────┘

Output: (batch_size, n_links=20)
```

### Training Details

| Parameter        | Value          |
|------------------|----------------|
| Sequence length  | 10 timesteps   |
| Training data    | 2,000 simulator snapshots |
| Epochs           | 30             |
| Learning rate    | 0.001          |
| Optimizer        | Adam           |
| Loss function    | MSE Loss       |
| Batch size       | 32             |
| Device           | CUDA (if available), else CPU |

### Training Pipeline

```python
# From backend/ directory:
python -m ml.congestion_lstm

# Steps:
# 1. Creates a NetworkSimulator (seed=42)
# 2. Runs 2,000 simulation steps, collecting utilization snapshots
# 3. Builds sliding-window dataset (window=10 → predict next)
# 4. Trains the LSTM for 30 epochs
# 5. Saves model to backend/ml/models/congestion_lstm.pt
```

### Saved Model Format

The model checkpoint (`congestion_lstm.pt`) contains:

```python
{
    "state_dict": model.state_dict(),  # PyTorch weights
    "n_links": 20,                     # Number of network links
    "seq_len": 10,                     # Input sequence length
    "link_keys": ["R1-R2", "R2-R3", ...],  # Link labels
}
```

### API Integration

The predictor is used by `GET /network/congestion-forecast`:

1. On each call, the current link utilization snapshot is appended to a rolling history
2. If the model is loaded and enough history exists (≥ `seq_len`), the LSTM predicts the next state
3. For multi-step forecasts, each prediction is fed back as input for the next step (autoregressive)
4. **Predictive Routing Integration**: The forecasted states can be routed to `gnn_predictive` or `rl_predictive` agents. This allows the agents to evaluate path qualities based on anticipated future congestion rather than just current metrics.
5. **Fallback**: If the model is not loaded or insufficient history, the last known snapshot is returned

---

## 2. PPO Reinforcement Learning Agent

### 2.1 Environment

**File**: `backend/ml/rl_environment.py`

The `NetworkRoutingEnv` is a custom Gymnasium environment that frames packet routing as a sequential decision-making problem.

#### Observation Space

**Box** space, shape `(n_links × 4,)`, normalized to [0, 1]:

```
For each of the 20 links:
  [utilization, queue_size/100, packet_loss_rate/0.06, base_latency/25]

Total: 80-dimensional float vector
```

#### Action Space

**Discrete(5)** — The agent selects one of K=5 pre-computed candidate paths.

#### Episode Structure

1. **Reset**: Simulator resets; a random (src, dst) pair is sampled
2. **Each step**:
   - Agent selects a candidate path (action)
   - Reward is computed based on the selected path's quality
   - Simulator advances one tick
   - New (src, dst) pair is sampled (keeps exploration diverse)
3. **Truncation**: Episode ends after 200 steps (no terminal state)

#### Reward Function

```python
reward = -(W_LATENCY × norm_latency + W_UTIL × mean_util + W_LOSS × norm_loss)
         - congested_link_count × CONGESTION_PENALTY × 0.1
```

| Weight / Constant         | Value | Description                              |
|---------------------------|-------|------------------------------------------|
| `W_LATENCY`               | 0.5   | Weight for normalized path latency       |
| `W_UTIL`                  | 0.3   | Weight for mean path utilization         |
| `W_LOSS`                  | 0.2   | Weight for normalized packet loss        |
| `CONGESTION_THRESHOLD`    | 0.85  | Utilization threshold for penalty        |
| `CONGESTION_PENALTY`      | 2.0   | Base penalty for congested links         |
| `_MAX_LATENCY_MS`         | 200.0 | Upper bound for latency normalization    |

The reward is always negative (minimization problem). Higher values (closer to 0) are better.

### 2.2 Training Pipeline

**File**: `backend/ml/train_rl.py`

```bash
cd backend
python -m ml.train_rl
```

#### Training Configuration

| Parameter          | Value            |
|--------------------|------------------|
| Algorithm          | PPO              |
| Policy             | MlpPolicy        |
| Total timesteps    | 500,000          |
| Learning rate      | 3×10⁻⁴           |
| n_steps            | 2,048            |
| Batch size         | 64               |
| n_epochs           | 10               |
| Gamma (discount)   | 0.99             |
| GAE Lambda         | 0.95             |
| Clip range         | 0.2              |
| Entropy coeff      | 0.01             |
| Value function coeff| 0.5             |
| Max grad norm      | 0.5              |
| Checkpoint freq    | Every 50k steps  |
| Eval freq          | Every 25k steps  |
| Eval episodes      | 5                |
| Device             | CUDA if available, else CPU |

#### Training Outputs

```
backend/ml/models/
├── rl_router_final.zip          # Final trained model
├── best_model.zip               # Best evaluation checkpoint
├── ppo_checkpoint_50000_steps.zip
├── ppo_checkpoint_100000_steps.zip
├── ... (every 50k steps up to 500k)
└── congestion_lstm.pt           # LSTM model (separate training)
```

#### Training Results

From the training run (CPU, ~25 minutes):

| Metric                   | Value          |
|--------------------------|----------------|
| Training time            | 24.9 minutes   |
| Initial mean reward      | -77            |
| Final mean reward        | -61 (+21%)     |
| Best eval reward         | -45.81         |
| Best checkpoint step     | 25,000         |

#### TensorBoard Monitoring

```bash
tensorboard --logdir runs/ppo_routing/
```

Logs are saved to `runs/ppo_routing/` and include:
- Episode reward
- Episode length
- Policy loss
- Value loss
- Entropy

### 2.3 Inference

**File**: `backend/router/rl_agent.py`

```python
router = RLRouter()
router.try_load_model()  # Loads from backend/ml/models/rl_router_final.zip

# During API calls:
decision = router.predict(state, "R1", "R5")
```

#### Inference Flow

```
predict(state, src, dst)
    │
    ├── Find K=5 candidate paths via BFS
    │
    ├── If PPO model is loaded:
    │       ├── Build observation vector (80-dim)
    │       ├── model.predict(obs, deterministic=True)
    │       └── Map action index to candidate path
    │
    └── If PPO model NOT loaded (fallback):
            └── Pick path with lowest congestion-adjusted cost
    │
    └── Return RoutingDecision
```

---

## 3. Graph Neural Network (GNN) Router

**File**: ackend/ml/gnn_model.py and ackend/ml/train_gnn.py

### Architecture
The GNN model uses PyTorch to process the network topology dynamically. It takes the current network state and aims to minimize a custom congestion-adjusted loss function, optimizing for load balancing.

### Training Pipeline
`ash
python -m ml.train_gnn
`
The model is saved to ackend/ml/models/gnn_router.pt.

---

## 4. Multi-Agent Reinforcement Learning (MARL)

**File**: ackend/ml/train_multi_agent.py and ackend/ml/multi_agent_rl_environment.py

### How It Works
The MARL approach partitions the network into distinct regions and trains one PPO agent per region. To avoid full simultaneous MARL instability, it employs a naive self-play rotation: training one region's policy while others are frozen, then rotating.

### Training Pipeline
`ash
python -m ml.train_multi_agent
`
Models are saved as ackend/ml/models/multi_agent_region_{i}.zip.

---

## 5. External APIs

> **Important**: This project does **not** use any external AI APIs (Gemini, Ollama, OpenAI, etc.). All ML models are trained and run locally:

| Model                    | Framework         | Training Data Source         |
|--------------------------|-------------------|------------------------------|
| LSTM Congestion Predictor| PyTorch 2.3       | Self-generated simulator data|
| PPO Routing Agent        | Stable-Baselines3 | Custom Gymnasium environment |

---

## 6. Model Files

| File                              | Size    | Description                          |
|-----------------------------------|---------|--------------------------------------|
| `congestion_lstm.pt`              | ~225 KB | LSTM model weights + metadata        |
| `rl_router_final.zip`            | ~257 KB | Final PPO policy weights             |
| `best_model.zip`                 | ~257 KB | Best evaluation checkpoint           |
| `ppo_checkpoint_*_steps.zip`     | ~257 KB each | Periodic training checkpoints  |

> **Note**: Model files are listed in `.gitignore` under `backend/ml/models/` but may be committed for portability. If missing, retrain using the commands above.

---

## 7. Key Design Decisions

### Why LSTM for Congestion Prediction?

- Network utilization is a time-series signal with temporal dependencies
- LSTM handles variable-length sequences and captures both short and long-term patterns
- Lightweight enough to run inference in real-time on every API call

### Why PPO for Routing?

- PPO is state-of-the-art for continuous/discrete control with stable training
- The routing problem naturally maps to discrete action selection (choose a path)
- Stable-Baselines3 provides a battle-tested implementation with monitoring tools
- PPO's clipped objective prevents catastrophic policy updates

### Why Heuristic Fallback?

- The system must work out-of-the-box before any training occurs
- The fallback (congestion-aware Dijkstra) is itself a strong baseline
- This ensures zero-downtime cold-start behavior
