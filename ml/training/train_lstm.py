"""Train the congestion forecaster, and prove it is worth having.

Run from the repository root::

    python -m ml.training.train_lstm

The audit's verdict on the previous version was that it was unfalsifiable
busywork, and it was right on every count:

* **The target was a random walk.** Utilization evolved as
  ``u_{t+1} = clip(u_t + N(0, 0.05))``. The Bayes-optimal one-step predictor for
  a random walk is the *identity function*. A 2-layer, 64-unit LSTM was being
  trained to learn persistence, and any low MSE it reported was that, not
  intelligence. The simulator now generates an AR(1) process around a per-link
  diurnal cycle, so there is genuine temporal structure to learn — while
  persistence remains a strong baseline.
* **There was no baseline.** The model was never compared against persistence,
  so its MSE meant nothing. This script evaluates three predictors and reports a
  **skill score**, ``1 - mse_lstm / mse_persistence``: positive means the LSTM
  beat persistence, negative means it did not.
* **There was no train/val/test split.** ``train()`` used 100% of the data and
  then printed the training loss under the label "final validation-style loss".
  The split is now chronological — never shuffled across the boundary, because
  that leaks the future into the past for a time series.
* **There was no training pipeline at all.** Training lived in an
  ``if __name__ == "__main__"`` block inside the model file, the artifact was
  absent from the repository, and so predictive routing was a silent no-op.

The checkpoint is written **only if the skill score is positive**. A forecaster
that cannot beat copying the last value should not be deployed, and shipping it
anyway is how a headline feature ends up never having run.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

import numpy as np
import torch
from torch import nn

from core.simulator import NetworkSimulator
from ml.architectures.lstm import CongestionLSTM
from ml.model_registry import RESULTS_DIR, path_for

logger = logging.getLogger("train_lstm")


def collect_snapshots(steps: int, num_nodes: int = 25, seed: int = 42) -> list[list[float]]:
    """Run the simulator and record the utilization vector at every step."""
    simulator = NetworkSimulator(num_nodes=num_nodes, seed=seed)
    return [[link.utilization for link in simulator.step().links] for _ in range(steps)]


def make_windows(
    snapshots: list[list[float]], seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding windows: ``seq_len`` snapshots predict the next one."""
    inputs, targets = [], []
    for index in range(len(snapshots) - seq_len):
        inputs.append(snapshots[index : index + seq_len])
        targets.append(snapshots[index + seq_len])
    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


def persistence_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """MSE of ``u_hat = u_t`` — copy the most recent observation forward."""
    return float(torch.mean((x[:, -1, :] - y) ** 2))


def window_mean_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """MSE of ``u_hat = mean(window)`` — the other trivial predictor."""
    return float(torch.mean((x.mean(dim=1) - y) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the congestion forecaster.")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-anyway",
        action="store_true",
        help="Save even if the skill score is negative (for debugging only).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    # Each episode builds a fresh simulator, which logs its topology at INFO.
    # That is useful once and noise 10,000 times.
    logging.getLogger("core.simulator").setLevel(logging.WARNING)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Collecting %d simulator steps...", args.steps)
    snapshots = collect_snapshots(args.steps, seed=args.seed)
    n_links = len(snapshots[0])

    # Chronological split. Shuffling a time series across the split boundary
    # leaks future information into training and inflates every metric.
    train_end = int(len(snapshots) * 0.70)
    val_end = int(len(snapshots) * 0.85)
    train_x, train_y = make_windows(snapshots[:train_end], args.seq_len)
    val_x, val_y = make_windows(snapshots[train_end:val_end], args.seq_len)
    test_x, test_y = make_windows(snapshots[val_end:], args.seq_len)
    logger.info(
        "Chronological split: train=%d val=%d test=%d windows over %d links",
        len(train_x),
        len(val_x),
        len(test_x),
        n_links,
    )

    model = CongestionLSTM(
        n_links=n_links, hidden_size=args.hidden_size, num_layers=args.num_layers
    )
    logger.info("Model has %d trainable parameters", model.parameter_count())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x))
        total = 0.0
        for start in range(0, len(train_x), args.batch_size):
            batch = permutation[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y))

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "epoch %02d/%d  train_mse=%.6f  val_mse=%.6f",
                epoch,
                args.epochs,
                total / max(1, len(train_x)),
                val_loss,
            )

        if epochs_without_improvement >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - started

    # --- the only comparison that means anything ---------------------------
    model.eval()
    with torch.no_grad():
        mse_lstm = float(criterion(model(test_x), test_y))
    mse_persistence = persistence_mse(test_x, test_y)
    mse_mean = window_mean_mse(test_x, test_y)
    skill = 1.0 - (mse_lstm / mse_persistence) if mse_persistence > 0 else 0.0

    logger.info("Training finished in %.1fs", elapsed)
    logger.info("HELD-OUT TEST MSE")
    logger.info("  persistence (copy last) : %.6f", mse_persistence)
    logger.info("  window mean             : %.6f", mse_mean)
    logger.info("  LSTM                    : %.6f", mse_lstm)
    logger.info("  SKILL SCORE vs persistence: %+.4f", skill)

    report = {
        "model": "congestion_lstm",
        "seed": args.seed,
        "seq_len": args.seq_len,
        "n_links": n_links,
        "split": "chronological 70/15/15",
        "train_windows": len(train_x),
        "val_windows": len(val_x),
        "test_windows": len(test_x),
        "test_mse": {
            "persistence": mse_persistence,
            "window_mean": mse_mean,
            "lstm": mse_lstm,
        },
        "skill_score_vs_persistence": skill,
        "beats_persistence": bool(skill > 0),
        "train_seconds": round(elapsed, 1),
        "parameters": model.parameter_count(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "lstm_evaluation.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", RESULTS_DIR / "lstm_evaluation.json")

    if skill <= 0 and not args.save_anyway:
        logger.warning(
            "Skill score is %+.4f, so the LSTM does not beat persistence. The "
            "checkpoint was NOT saved: predictive routing stays disabled and "
            "this negative result is reported in the README.",
            skill,
        )
        return

    destination = path_for("lstm")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_links": n_links,
            "seq_len": args.seq_len,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "skill_score": skill,
            "link_keys": [],
        },
        destination,
    )
    logger.info("Saved checkpoint to %s (skill score %+.4f)", destination, skill)


if __name__ == "__main__":
    main()
