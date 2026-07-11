from __future__ import annotations

import sys
import time
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure backend root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from ml.gnn_model import GNNRouterModel
from simulator.network_sim import NetworkSimulator


_ROOT = Path(__file__).parent
_MODEL_DIR = _ROOT / "models"
_FINAL_MODEL_PATH = _MODEL_DIR / "gnn_router.pt"


def find_candidate_paths(simulator: NetworkSimulator, src: str, dst: str, limit: int = 5) -> list[list[str]]:
    """BFS to find candidate paths."""
    graph = simulator.graph
    queue: list[list[str]] = [[src]]
    paths: list[list[str]] = []

    while queue and len(paths) < limit:
        path = queue.pop(0)
        current = path[-1]

        if current == dst:
            paths.append(path)
            continue

        for neighbor in graph.neighbors(current):
            if neighbor not in path:
                queue.append([*path, neighbor])

    return paths


def get_path_cost(simulator: NetworkSimulator, path: list[str]) -> float:
    """Calculate the actual congestion-adjusted cost of a path."""
    cost = 0.0
    for i in range(len(path) - 1):
        cost += simulator.get_edge_weight(path[i], path[i + 1])
    return cost


def generate_dataset(simulator: NetworkSimulator, num_samples: int = 1500) -> list[dict]:
    """Advance simulator and sample random routing tasks to build dataset."""
    dataset = []
    print(f"Generating {num_samples} training samples from network simulator...")
    
    # Run the simulator to populate initial traffic
    for _ in range(50):
        simulator.step()

    while len(dataset) < num_samples:
        state = simulator.step()
        nodes = list(state.nodes)
        
        # Select 5 random src-dst pairs at this step
        for _ in range(5):
            src, dst = random.sample(nodes, 2)
            candidates = find_candidate_paths(simulator, src, dst, limit=5)
            if not candidates:
                continue

            node_to_idx = {node: i for i, node in enumerate(nodes)}
            num_nodes = len(nodes)

            # Build node features
            degrees = [0] * num_nodes
            for link in state.links:
                degrees[node_to_idx[link.source]] += 1
                degrees[node_to_idx[link.target]] += 1
            max_deg = max(1, max(degrees))

            x_list = []
            for i, node in enumerate(nodes):
                is_src = 1.0 if node == src else 0.0
                is_dst = 1.0 if node == dst else 0.0
                deg = float(degrees[i]) / max_deg
                x_list.append([is_src, is_dst, deg])

            # Build edge index & edge attributes
            edges_src = []
            edges_dst = []
            edge_attr_list = []

            for link in state.links:
                u_idx = node_to_idx[link.source]
                v_idx = node_to_idx[link.target]
                attr = [
                    float(link.utilization),
                    float(link.queue_size) / 100.0,
                    float(link.packet_loss_rate) / 0.06,
                    float(link.base_latency) / 25.0,
                ]
                # u -> v
                edges_src.append(u_idx)
                edges_dst.append(v_idx)
                edge_attr_list.append(attr)
                # v -> u
                edges_src.append(v_idx)
                edges_dst.append(u_idx)
                edge_attr_list.append(attr)

            paths_idx = [[node_to_idx[n] for n in p] for p in candidates]
            costs = [get_path_cost(simulator, p) for p in candidates]

            dataset.append({
                "x": torch.tensor(x_list, dtype=torch.float32),
                "edge_index": torch.tensor([edges_src, edges_dst], dtype=torch.long),
                "edge_attr": torch.tensor(edge_attr_list, dtype=torch.float32),
                "paths": paths_idx,
                "targets": torch.tensor(costs, dtype=torch.float32)
            })

            if len(dataset) >= num_samples:
                break
                
    return dataset


def main() -> None:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GNN Routing Model Training")
    print("=" * 60)

    # 1. Initialize simulator (25 nodes, which is the scaled default)
    sim = NetworkSimulator(num_nodes=25, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 2. Generate training and validation data
    train_data = generate_dataset(sim, num_samples=1600)
    val_data = generate_dataset(sim, num_samples=400)

    # 3. Initialize GNN Model
    model = GNNRouterModel(node_dim=3, edge_dim=4, hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    epochs = 40
    print(f"\nTraining GNN for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        random.shuffle(train_data)

        for sample in train_data:
            x = sample["x"].to(device)
            edge_index = sample["edge_index"].to(device)
            edge_attr = sample["edge_attr"].to(device)
            paths = sample["paths"]
            targets = sample["targets"].to(device)

            optimizer.zero_grad()
            predictions = model(x, edge_index, edge_attr, paths)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sample in val_data:
                x = sample["x"].to(device)
                edge_index = sample["edge_index"].to(device)
                edge_attr = sample["edge_attr"].to(device)
                paths = sample["paths"]
                targets = sample["targets"].to(device)

                predictions = model(x, edge_index, edge_attr, paths)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        train_mse = total_loss / len(train_data)
        val_mse = val_loss / len(val_data)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # 4. Save trained weights
    torch.save(
        {
            "state_dict": model.state_dict(),
            "node_dim": 3,
            "edge_dim": 4,
            "hidden_dim": 32
        },
        _FINAL_MODEL_PATH
    )
    print(f"Model saved to: {_FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
