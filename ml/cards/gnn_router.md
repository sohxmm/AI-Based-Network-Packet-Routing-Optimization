# Model card: `gnn_router`

## Overview

**Task.** Rank *k* candidate paths for a routing demand, conditioned on the QoS
traffic class.

**Input.** Node features (is-source, is-destination, normalised degree), edge
features (utilization, queue, loss, base latency), the node and edge indices of
each candidate path, and a 6-dimensional encoding of the traffic class.

**Output.** One score per candidate. **Only `argmin` is ever used.** The absolute
value carries no meaning and must never be reported as a latency estimate.

**Where it runs.** `routing/learned/gnn.py`, at inference time, on every
`algorithm="gnn"` request.

## Training data

- Generator: `core/simulator.py`, three independently seeded instances.
- Topology: 25 nodes, 50 links, average degree 4.0, diameter 5–6 — **the same
  topology it is served on**.
- Samples: 2,500 train / 600 validation / 600 test.
- **The validation set is genuinely independent**: a different seed and therefore
  a different topology and traffic realisation. It is *not* a temporal
  continuation of the training trajectory. The previous version generated
  validation data by continuing the same simulator instance, so its reported
  "72% val MSE reduction" was not evidence of generalisation at all.

## Architecture

- 2 × `MessagePassingLayer`, hidden dim 64, edge-conditioned messages,
  **mean aggregation** (degree-normalised).
- Path scorer: mean-pooled node embeddings ‖ mean-pooled edge embeddings ‖ 9
  explicit path/QoS features → MLP(137 → 64 → 1).
- **47,361 parameters.**

Two representational fixes over the previous version:

- Aggregation was an unnormalised `index_add_`, so embedding magnitudes scaled
  with node degree. Training on a degree-2 ring and serving on a degree-4 mesh
  roughly doubled activations at inference — a self-inflicted covariate shift.
- The path scorer was a plain mean over node embeddings: permutation invariant,
  length invariant, and edge blind. It was asked to predict a path's cost while
  being shown neither the path's length nor its links.

## Hyperparameters

| | |
|---|---|
| Optimizer | Adam, lr 1e-3 |
| Schedule | `ReduceLROnPlateau` on validation top-1, patience 4, factor 0.5 |
| Loss | Pairwise margin ranking loss, margin 0.05 |
| Batch | 16 (gradient accumulation; graphs have different shapes) |
| Epochs | 40 max, early stopping patience 8 — **stopped at 25** |
| Checkpoint | Best validation top-1, not the last epoch |
| Seeds | `torch`, `random`, `numpy` all set to 42 |
| Wall clock | 594 s on 4 CPU cores |

## Metrics

Reported against how the model is *used* (argmin over candidates), not against
the training loss.

| Metric | Model | Random baseline |
|---|---|---|
| **Top-1 accuracy** (held-out test) | **0.978** | 0.227 |
| **Mean regret vs oracle** | **0.0006** | 0.676 |
| Best validation top-1 | 0.970 | — |

MSE is deliberately not reported. The previous version optimised MSE on a
regression target whose value is never consumed.

## Known failure modes

- **It converges to Dijkstra on best-effort traffic.** The measured
  `dijkstra_match_rate` is ~1.00 on single-objective scenarios. This is *correct*
  — Dijkstra is provably optimal for an additive cost, so a good ranker must
  agree with it — but it means the GNN adds no information there. It only
  differentiates under QoS constraints. The benchmark declares this in its
  `warnings` block.
- **Restricted to the candidate set.** It ranks 5 pre-computed paths; it cannot
  discover a route outside them.
- **Trained at 25 nodes.** It runs on larger topologies (the architecture is
  size-agnostic) but has not been validated there.
- **Class conditioning is an input, not a guarantee.** Nothing forces it to
  respect a hard constraint; constraint satisfaction is measured, not enforced.
  Filtering its output through the QoS oracle would make its satisfaction rate
  identical to the constrained baseline by construction, which would be a
  meaningless win.

## Reproduction

```bash
python -m ml.training.train_gnn        # ~10 min, CPU only
```

Writes `ml/checkpoints/gnn_router.pt` and `ml/results/gnn_evaluation.json`.
