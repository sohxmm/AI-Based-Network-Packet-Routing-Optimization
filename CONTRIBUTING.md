# Contributing

## Getting set up

```bash
git clone <repo> && cd AI-Based-Network-Packet-Routing-Optimization
python -m venv .venv && source .venv/bin/activate
pip install -r service/requirements-dev.txt
cd web && npm ci && cd ..
pre-commit install
```

No GPU is needed for anything in this repository, including training.

## Branches

`feature/<short-description>`, `fix/<short-description>` or
`docs/<short-description>`, branched from `main`.

## Before you push

```bash
make lint     # ruff + eslint, both must be clean
make test     # backend + frontend suites
make verify   # honesty gates
```

## The one rule that is not negotiable

**Any change that affects a reported metric must be accompanied by a re-run of
the pipeline that produced it, and by `python scripts/verify_claims.py` passing.**

This project once documented a training result — "mean reward improved from -77
to -61, best -45.81" — that the committed evaluation file flatly contradicted.
Nobody did that deliberately; the number came from a run that was never
committed, and there was no mechanism that could notice. `verify_claims.py` is
that mechanism. If you change the reward function, the topology generator, the
cost model or a training script, regenerate the artifacts and re-run it:

```bash
make train && make bench && make verify
```

## Adding a routing algorithm

1. Implement `routing.base.Router` in the right subpackage
   (`classical/`, `heuristic/` or `learned/`).
2. Register it in `routing/registry.py` — that is the only list. The dispatcher,
   the benchmark harness, the API schema and the dashboard all read from it, so
   there is nothing else to update and nothing that can drift.
3. If it is a learned router, add its artifact to `ml/model_registry.py` so a
   missing file is reported loudly instead of silently degrading to a heuristic.
4. Add it to the benchmark and check the guardrails: a `dijkstra_match_rate`
   above 0.95 means it is degenerate, and that has to be reported rather than
   presented as a distinct algorithm.

## Style

Ruff enforces the Python style; ESLint the JavaScript. Comments should explain
*why*, especially where a decision looks odd — most of the unusual choices in
this codebase are deliberate responses to a specific defect, and the comment is
what stops someone helpfully reverting the fix.
