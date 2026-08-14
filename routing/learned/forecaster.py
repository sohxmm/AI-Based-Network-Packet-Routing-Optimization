"""CongestionPredictor — inference wrapper around the trained LSTM forecaster.

What "predictive routing" actually is: forecast each link's utilization one step
ahead, build a hypothetical :class:`NetworkState` from those forecasts, and route
on *that*. A router in predictive mode steers around congestion that has not
arrived yet.

Why it never worked before. The LSTM artifact was absent from the repository, so
:func:`build_forecast_state` returned ``None`` on every call and the caller fell
through to ``or state``. Predictive routing was a no-op — which is exactly why
``gnn_predictive`` and ``rl_predictive`` were byte-identical to ``gnn`` and
``rl`` in all five committed benchmark files. Two of the eight benchmarked
"algorithms" were duplicate columns.

Two further defects are fixed here:

* ``predict_next`` crashed when a link failed, because the model's input width
  is fixed at training time while ``len(state.links)`` shrinks on failure. It
  now detects the width change and degrades to persistence.
* ``torch.load`` ran without ``weights_only=True``, which executes arbitrary
  pickle payloads from the checkpoint file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.models import LinkState, NetworkState
from ml.model_registry import path_for

logger = logging.getLogger(__name__)

_torch: Any = None
_lstm_class: Any = None


def _try_import_torch() -> bool:
    global _torch, _lstm_class
    if _torch is not None:
        return True
    try:
        import torch

        from ml.architectures.lstm import CongestionLSTM

        _torch = torch
        _lstm_class = CongestionLSTM
        return True
    except ImportError:
        return False


class CongestionPredictor:
    """Hold the trained LSTM and produce next-step utilization forecasts."""

    def __init__(self, seq_len: int = 20, device: str | None = None) -> None:
        self.seq_len = seq_len
        self.model: Any = None
        self.link_keys: list[str] = []
        self.skill_score: float | None = None
        self._device_name = device
        self._device: Any = None
        #: Rolling window of recent snapshots, owned by this object rather than
        #: living as a module-level global mutated from request handlers.
        self.history: list[list[float]] = []

    # -- model management -------------------------------------------------

    def load(self, path: str | Path | None = None) -> bool:
        """Load the checkpoint if present; return whether a model is available."""
        candidate = Path(path) if path else path_for("lstm")
        if not candidate.exists():
            logger.info(
                "CongestionPredictor: no trained LSTM at %s. Predictive routing "
                "is disabled. Train with: python -m ml.training.train_lstm",
                candidate.name,
            )
            return False
        if not _try_import_torch():
            logger.warning("CongestionPredictor: torch unavailable.")
            return False

        try:
            checkpoint = _torch.load(
                str(candidate), map_location="cpu", weights_only=True
            )
            self.seq_len = int(checkpoint.get("seq_len", self.seq_len))
            self.link_keys = list(checkpoint.get("link_keys", []))
            self.skill_score = checkpoint.get("skill_score")
            self._device = _torch.device(
                self._device_name
                or ("cuda" if _torch.cuda.is_available() else "cpu")
            )
            self.model = _lstm_class(
                n_links=int(checkpoint["n_links"]),
                hidden_size=int(checkpoint.get("hidden_size", 64)),
                num_layers=int(checkpoint.get("num_layers", 2)),
            ).to(self._device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
        except Exception:
            logger.exception("CongestionPredictor: failed to load %s", candidate)
            self.model = None
            return False

        logger.info(
            "CongestionPredictor loaded %s (seq_len=%d, skill_score=%s)",
            candidate.name,
            self.seq_len,
            self.skill_score,
        )
        return True

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    @property
    def expected_links(self) -> int | None:
        """Input width the model was trained with, or None if not loaded."""
        return self.model.output.out_features if self.model is not None else None

    # -- inference --------------------------------------------------------

    def observe(self, state: NetworkState) -> None:
        """Append the current utilization vector to the rolling window."""
        self.history.append([link.utilization for link in state.links])
        del self.history[: -self.seq_len]

    def predict_next(self, recent_snapshots: list[list[float]]) -> list[float]:
        """Forecast the next utilization vector, degrading to persistence.

        Persistence (``u_hat = u_t``) is returned whenever the model is absent,
        the window is too short, or the topology changed so the snapshots no
        longer match the model's input width. That last case used to raise,
        because a failed link shortens every subsequent row.
        """
        if not recent_snapshots:
            return []

        if self.model is None or len(recent_snapshots) < self.seq_len:
            return list(recent_snapshots[-1])

        expected = self.expected_links
        window = recent_snapshots[-self.seq_len :]
        if expected is not None and any(len(row) != expected for row in window):
            logger.debug(
                "Forecast window width changed (expected %s); using persistence.",
                expected,
            )
            return list(recent_snapshots[-1])

        features = _torch.tensor([window], dtype=_torch.float32, device=self._device)
        with _torch.no_grad():
            prediction = self.model(features).cpu().squeeze(0).tolist()
        return [float(max(0.0, min(1.0, value))) for value in prediction]


def build_forecast_state(
    state: NetworkState,
    predictor: CongestionPredictor,
    history: list[list[float]] | None = None,
) -> NetworkState | None:
    """Return a hypothetical next-step state, or None if forecasting is off.

    Returning ``None`` rather than the current state is deliberate: the caller
    must decide what to do, instead of silently routing on present-tense data
    while reporting it as a forecast.
    """
    if predictor.model is None:
        return None

    window = history if history is not None else predictor.history
    if len(window) < predictor.seq_len:
        return None

    predicted = predictor.predict_next(window)
    if len(predicted) != len(state.links):
        return None

    links = []
    for index, link in enumerate(state.links):
        utilization = max(0.0, min(1.0, predicted[index]))
        links.append(
            LinkState(
                source=link.source,
                target=link.target,
                base_latency=link.base_latency,
                bandwidth=link.bandwidth,
                utilization=utilization,
                queue_size=int(utilization * 100),
                packet_loss_rate=max(0.0, utilization - 0.7) * 0.2,
            )
        )

    return NetworkState(
        nodes=list(state.nodes),
        links=links,
        timestamp=state.timestamp,
        step_count=state.step_count,
    )


__all__ = ["CongestionPredictor", "build_forecast_state"]
