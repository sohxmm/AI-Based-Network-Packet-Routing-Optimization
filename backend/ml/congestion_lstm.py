from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from simulator.network_sim import NetworkSimulator


class CongestionLSTM(nn.Module):
    """Predict next-step link utilization from recent utilization windows."""

    def __init__(
        self,
        n_links: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_links,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, n_links)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predicted link utilization for the next simulator step."""
        sequence_output, _ = self.lstm(inputs)
        last_step = sequence_output[:, -1, :]
        return torch.sigmoid(self.output(self.dropout(last_step)))


class CongestionPredictor:
    """Collect simulator data, train the LSTM, and produce utilization forecasts."""

    def __init__(self, seq_len: int = 10, device: str | None = None) -> None:
        self.seq_len = seq_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: CongestionLSTM | None = None
        self.history: list[list[float]] = []
        self.link_keys: list[str] = []

    def collect_data(self, simulator: NetworkSimulator, steps: int = 2000) -> list[list[float]]:
        """Run the simulator and collect link utilization snapshots."""
        snapshots: list[list[float]] = []
        for _ in range(steps):
            state = simulator.step()
            if not self.link_keys:
                self.link_keys = [f"{link.source}-{link.target}" for link in state.links]
            snapshots.append([link.utilization for link in state.links])

        self.history = snapshots
        return snapshots

    def prepare_dataset(
        self,
        snapshots: list[list[float]] | None = None,
        batch_size: int = 32,
    ) -> DataLoader:
        """Build sliding-window samples where 10 snapshots predict the next one."""
        values = snapshots or self.history
        if len(values) <= self.seq_len:
            raise ValueError("Not enough snapshots to build an LSTM dataset.")

        inputs = []
        targets = []
        for index in range(len(values) - self.seq_len):
            inputs.append(values[index:index + self.seq_len])
            targets.append(values[index + self.seq_len])

        dataset = TensorDataset(
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        snapshots: list[list[float]] | None = None,
        epochs: int = 30,
        learning_rate: float = 0.001,
    ) -> list[float]:
        """Train the LSTM and return epoch losses."""
        values = snapshots or self.history
        if not values:
            raise ValueError("No training snapshots available.")

        self.model = CongestionLSTM(n_links=len(values[0])).to(self.device)
        loader = self.prepare_dataset(values)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        losses: list[float] = []

        self.model.train()
        for _ in range(epochs):
            total_loss = 0.0
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                predictions = self.model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / max(1, len(loader)))

        return losses

    def predict_next(self, recent_snapshots: list[list[float]]) -> list[float]:
        """Predict the next utilization vector, or persist the latest vector safely."""
        if not recent_snapshots:
            return []

        if self.model is None or len(recent_snapshots) < self.seq_len:
            return recent_snapshots[-1]

        window = recent_snapshots[-self.seq_len:]
        features = torch.tensor([window], dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features).cpu().squeeze(0).tolist()

        return [float(max(0.0, min(1.0, value))) for value in prediction]

    def save(self, path: str | Path) -> None:
        """Save the trained model and predictor metadata."""
        if self.model is None:
            raise ValueError("Cannot save before training a model.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_links": self.model.output.out_features,
                "seq_len": self.seq_len,
                "link_keys": self.link_keys,
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        """Load a previously trained congestion model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.seq_len = checkpoint["seq_len"]
        self.link_keys = checkpoint.get("link_keys", [])
        self.model = CongestionLSTM(n_links=checkpoint["n_links"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    simulator = NetworkSimulator(seed=42)
    predictor = CongestionPredictor(seq_len=10)
    data = predictor.collect_data(simulator, steps=2000)
    losses = predictor.train(data, epochs=30)
    predictor.save(Path(__file__).parent / "models" / "congestion_lstm.pt")

    print(f"Final validation-style loss: {losses[-1]:.6f}")
    for index in range(5):
        start = index
        window = data[start:start + predictor.seq_len]
        actual = data[start + predictor.seq_len]
        predicted = predictor.predict_next(window)
        print(f"Prediction {index + 1}:")
        print(f"  predicted={predicted[:5]}")
        print(f"  actual={actual[:5]}")
