# Model card: `congestion_lstm`

## Overview

**Task.** Forecast every link's utilization one step ahead, so a router can steer
around congestion that has not arrived yet.

**Input.** A 20-step window of utilization vectors, `[batch, 20, 50]`.
**Output.** The next utilization vector, `[batch, 50]`, in [0, 1].

**Where it runs.** `routing/learned/forecaster.py`. When `use_forecast=true`, the
forecast builds a hypothetical `NetworkState` and the router routes on *that*.

> Predictive routing had **never once executed** before this. The artifact was
> absent from the repository, so `build_forecast_state` returned `None` on every
> call and the caller fell through to `or state`. That is why `gnn_predictive`
> and `rl_predictive` were byte-identical to `gnn` and `rl` in all five committed
> benchmark files: two of the eight benchmarked "algorithms" were duplicate
> columns.

## Training data

6,000 consecutive steps from `NetworkSimulator(num_nodes=25, seed=42)`, split
**chronologically** 70/15/15 into 4,180 / 880 / 880 windows. Never shuffled
across the split boundary — doing so leaks the future into the past and inflates
every metric on a time series. The previous version used 100% of the data for
training and printed the *training* loss under the label "final validation-style
loss".

## Architecture

2-layer LSTM, hidden 64, dropout 0.1, linear head to 50 outputs.
**66,226 parameters.**

**It predicts a change, not a level.** This is the single most important design
choice in the model and it was made in response to a measured failure:

- First attempt, predicting the level: **skill score −1.77**. The training script
  refused to save the checkpoint, correctly.
- Utilization is strongly autocorrelated, so "copy the last value forward" is
  right to within the one-step noise almost every time. A network asked to emit
  the level spends all its capacity re-learning the identity function and still
  loses.
- Predicting the residual `u_{t+1} − u_t` removes the identity from the problem.
  Persistence becomes the specific hypothesis "the residual is zero", which is a
  fair fight. The window is differenced on input and the level re-attached on
  output, clamped to [0, 1].

Second attempt, predicting the residual: **skill score +0.15**. Saved.

## Hyperparameters

| | |
|---|---|
| Optimizer | Adam, lr 1e-3, gradient clipping at 1.0 |
| Schedule | `ReduceLROnPlateau` on validation MSE |
| Loss | MSE **on the residual**; validation and test on the reconstructed level |
| Batch | 64 |
| Epochs | 120 max, early stopping patience 15 — **stopped at 69** |
| Seed | 42 |
| Wall clock | 28 s on 4 CPU cores |

## Metrics

The only comparison that means anything is against the trivial predictors, on
the held-out chronological test set.

| Predictor | Test MSE |
|---|---|
| Window mean | 0.016811 |
| Persistence (copy last value) | 0.001337 |
| **LSTM** | **0.001137** |

**Skill score vs persistence: +0.1497** — the LSTM explains about 15% more of the
one-step variance than copying the last value forward.

That is a modest but genuine result, and the modesty is the honest part: on an
AR(1) process with σ = 0.03 noise, persistence is close to optimal and there is
not much left to win. **The checkpoint is only written when the skill score is
positive**; a forecaster that cannot beat copying the last value should not be
deployed, and shipping one anyway is how a headline feature ends up never
having run.

## Known failure modes

- **Degrades to persistence, deliberately**, when the model is absent, the window
  is too short, or the topology changed. That last case used to *crash*: the
  model's input width is fixed at training time while `len(state.links)` shrinks
  when a link fails, so the rolling window held ragged rows and tensor
  construction raised. It was latent only because the model never loaded.
- **One step ahead only.** Multi-step forecasting would compound its own error.
- **Trained on this simulator's dynamics.** The learnable structure is a diurnal
  cycle with a 40-step period plus AR(1) mean reversion. On traffic without that
  structure the skill score would collapse toward zero, which is the correct
  outcome and would be visible in the metric.
- **A +0.15 skill score is not a large edge.** Predictive routing changes the
  chosen path only occasionally.

## Reproduction

```bash
python -m ml.training.train_lstm       # ~30 s, CPU only
```
