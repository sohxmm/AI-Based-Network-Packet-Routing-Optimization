"""Sequence model that forecasts next-step link utilization.

This file holds the ``nn.Module`` and nothing else. It previously also contained
the data collector, the trainer, the inference wrapper and a ``__main__``
training block — four responsibilities in one module, and the reason the model
had no train/validation/test split and no baseline comparison. Training now
lives in ``ml/training/train_lstm.py`` and inference in
``routing/learned/forecaster.py``.
"""

from __future__ import annotations

import torch
from torch import nn


class CongestionLSTM(nn.Module):
    """Predict the next utilization vector from a window of recent snapshots.

    Input  ``[batch, seq_len, n_links]``
    Output ``[batch, n_links]`` in [0, 1] via a sigmoid, matching the domain of
    utilization exactly so the model cannot emit impossible values.
    """

    def __init__(
        self,
        n_links: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_links = n_links
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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

    def parameter_count(self) -> int:
        """Total trainable parameters, reported in the model card."""
        return sum(p.numel() for p in self.parameters())


__all__ = ["CongestionLSTM"]
