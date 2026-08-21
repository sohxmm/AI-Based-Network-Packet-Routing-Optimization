"""Sequence model that forecasts next-step link utilization.

This file holds the ``nn.Module`` and nothing else. It previously also contained
the data collector, the trainer, the inference wrapper and a ``__main__``
training block — four responsibilities in one module, and the reason the model
had no train/validation/test split and no baseline comparison. Training now
lives in ``ml/training/train_lstm.py`` and inference in
``routing/learned/forecaster.py``.

**The model predicts a change, not a level.** Link utilization is strongly
autocorrelated, so "copy the last value forward" is a very hard baseline: it is
right to within the one-step noise almost every time. A network asked to output
the level spends all of its capacity re-learning the identity function and still
loses to persistence — measured at a skill score of -1.77 on the first attempt.
Predicting the *residual* ``u_{t+1} - u_t`` removes the identity from the
problem entirely, so every unit of capacity goes to the part that is actually
predictable: the diurnal cycle and the mean-reversion drift. Persistence becomes
the specific hypothesis "the residual is zero", which is a fair fight.
"""

from __future__ import annotations

import torch
from torch import nn


class CongestionLSTM(nn.Module):
    """Predict the next utilization vector from a window of recent snapshots.

    Input  ``[batch, seq_len, n_links]``
    Output ``[batch, n_links]``

    With ``predict_delta=True`` (the default) the network emits a bounded change
    which is added to the most recent observation and clamped to [0, 1], so the
    output is always a valid utilization regardless of what the network says.
    """

    #: Maximum single-step change the model may predict, in utilization units.
    #: The simulator's per-step noise has sigma 0.03, so 0.5 is far beyond any
    #: real movement and never binds in practice; it exists to keep a badly
    #: initialised network from emitting nonsense.
    MAX_DELTA = 0.5

    def __init__(
        self,
        n_links: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        predict_delta: bool = True,
    ) -> None:
        super().__init__()
        self.n_links = n_links
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predict_delta = predict_delta

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
        if self.predict_delta:
            return torch.clamp(inputs[:, -1, :] + self.delta(inputs), 0.0, 1.0)
        sequence_output, _ = self.lstm(inputs)
        return torch.sigmoid(self.output(self.dropout(sequence_output[:, -1, :])))

    def delta(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the predicted change from the last observation.

        Training uses this directly, so the loss is computed on the residual
        rather than on a level that is dominated by the input.
        """
        # Differencing the window makes the input stationary, which is what the
        # recurrent layers should be modelling. The final level is re-attached
        # in forward().
        differenced = inputs[:, 1:, :] - inputs[:, :-1, :]
        sequence_output, _ = self.lstm(differenced)
        raw = self.output(self.dropout(sequence_output[:, -1, :]))
        return self.MAX_DELTA * torch.tanh(raw)

    def parameter_count(self) -> int:
        """Total trainable parameters, reported in the model card."""
        return sum(p.numel() for p in self.parameters())


__all__ = ["CongestionLSTM"]
