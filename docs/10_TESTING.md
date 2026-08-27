# Testing

```bash
make test        # Python + frontend
make lint        # ruff + eslint, both at zero tolerance
make verify      # honesty gates + documented claims vs. artifacts
```

Current state: **120 Python tests** and **18 frontend tests** passing, both
linters clean, the claim verifier reporting no violations.

Before this revision the suite was **not installable**. Tests imported `from
simulator.network_sim import ...`, which only resolved if you happened to be
`cd`'d into `backend/`, and several were plain scripts with `if __name__ ==
"__main__"` rather than pytest tests. There was no `conftest.py`, no `pytest.ini`
and no CI. "We have over 50 automated tests" was true in the sense that 50 test
functions existed.

---

## 1. Layout

```
tests/
├── conftest.py                 # shared fixtures; puts the repo root on sys.path
├── unit/
│   ├── core/                   # 11 tests — simulator, closed loop, cost, paths
│   ├── routing/                # 16 tests — every algorithm, stress cases
│   └── ml/                     # 33 tests — features, parity, locality, GNN, forecasting
├── integration/
│   └── api/                    # 48 tests — endpoints against a live TestClient
└── honesty/                    # 12 tests — the gates (see §3)
```

Run a slice:

```bash
python -m pytest tests/unit/ml -v
python -m pytest tests/honesty -v
python -m pytest -m "not slow"
```

---

## 2. Unit tests

### `tests/unit/core`

- **`test_closed_loop.py`** — the most important test in the directory. It
  asserts that `register_flow()` actually changes subsequent utilisation. Without
  this property the entire benchmark is measuring nothing: if routing decisions
  do not affect the network, "congestion-aware routing" has nothing to be aware
  of and every algorithm is scored against a movie it cannot influence.
- **`test_stress_phase1.py`** — simulator invariants over long runs: utilisation
  stays in [0, 1], queues stay non-negative, failure injection and restoration
  work, reset produces an identical state for an identical seed.

### `tests/unit/routing`

Every algorithm on known topologies, plus:

- Disconnected graphs return failure rather than raising.
- Routing a node **to itself** returns a valid zero-hop path. This one caught a
  real bug: the shared `path_links` helper returned `inf` for a single-node path,
  so a self-route cost infinity. Legacy tests earn their keep.
- ACO respects visited-node constraints (no cycles) and its pheromone table
  evaporates.
- `test_multi_agent_routing.py` compares single-agent against regional routing
  and reports which regions have trained policies versus which are falling back.

### `tests/unit/ml`

- **`test_train_serve_parity.py`** — that the observation built at serving time
  is identical to the one built during training. This is the test that would have
  caught the `candidate_paths(weighted=...)` skew, where training ranked one
  candidate set and inference ranked a different one.
- **`test_marl_locality.py`** — that a regional agent's observation genuinely
  contains no information from outside its region. Decentralised execution is a
  claim about what the agent *cannot see*, and a claim like that needs a test, not
  a comment.
- **`test_gnn.py`** — forward pass shape, degree-invariance of the aggregation,
  and that a failed checkpoint load leaves `is_trained` false.
- **`test_predictive_routing.py`** — that predictive mode actually routes on a
  forecast, and that `build_forecast_state()` returns `None` rather than falling
  back to the present state.

---

## 3. Honesty gates

`tests/honesty/test_honesty_gates.py`. These are the tests that make the rest of
the project trustworthy, and they are worth reading in full.

The project already had good instincts before the audit: `is_fallback` was
threaded through the stack, `dijkstra_match_rate` was invented specifically to
catch its own models being degenerate, and there was a UI badge whose only job
was to display "matches Dijkstra, no differentiation". What it did not have was
**enforcement**. The guardrails reported problems and nothing acted on them,
which is how a documented training result ended up contradicting the committed
evaluation file for the life of the project.

| Gate | What fails the build |
|---|---|
| `test_results_exist_for_the_dashboard_to_display` | `experiments/results/` is empty |
| `test_no_algorithm_is_silently_degenerate` | An algorithm matches Dijkstra >95% and **no warning declares it** |
| `test_reported_results_are_not_secretly_the_fallback` | Fallback rate >20% with **no warning declaring it** |
| `test_no_metric_is_structurally_constant` | A metric is identical across every algorithm |
| `test_max_utilization_is_not_pinned_at_one` | p95 bottleneck at 1.000 (an extreme statistic over a whole run) |
| `test_p_values_are_not_exactly_zero` | `p = 0.0` exactly — underflow from pseudo-replication |
| `test_statistics_use_independent_runs` | `n_runs < 2`, or a comparison's *n* disagrees with the scenario's |
| `test_effect_sizes_are_real_effect_sizes` | A comparison has no Cliff's delta and magnitude |
| `test_topology_is_recorded_and_not_a_ring` | Average degree below 3, or topology not recorded |
| `test_warnings_block_is_present` | A results file has no `warnings` key |
| `test_model_load_status_is_recorded` | A results file does not say which models were loaded |
| `test_predictive_variants_differ_from_their_base` | `gnn_predictive` equals `gnn` — the forecaster never ran |

### Two of these are subtler than they look

**Degeneracy requires a declaration, not an absence.** The gate does *not* fail
when an algorithm reproduces Dijkstra's path. That would be an unsatisfiable
test, because the GNN converging to Dijkstra under additive costs is the
*correct* outcome — Dijkstra is provably optimal there. What fails the build is
degeneracy that no warning declares. The project is allowed to have a negative
result; it is not allowed to have an undisclosed one.

**Fallback is the same shape.** The PPO agent's fixed-width observation genuinely
cannot fit a 100-node topology, so a 100% fallback rate on that scenario is a
documented architectural limitation, correctly handled. Failing CI for it would
be failing CI for the router behaving properly. What must never happen — and what
the gate catches — is the heuristic's answer served under the model's name with
nothing recording it.

Both gates are therefore about *disclosure*, which is the property that actually
matters.

---

## 4. Integration tests

`tests/integration/api/` runs against a real `TestClient`, so the app, the
routers, the models and the background wiring are all exercised.

Covered: every endpoint's contract, error codes (403 for live probing disabled,
404 for unknown scenarios, 409 for premature result requests, 422 for cap
violations), the experiment lifecycle, and the fact that a sandbox run reports
the same statistics block as the committed benchmark.

One test is worth calling out for how it was fixed rather than what it checks.
`test_results_before_completion_returns_409` used to submit a deliberately slow
experiment and then assert it had not finished. `TestClient` runs FastAPI
background tasks **synchronously after the response is returned**, so the job was
always already done — the test was a test of scheduling luck. It now injects a
job in a known state and asserts the contract directly.

---

## 5. Frontend tests

```bash
cd web && npm run test        # 18 tests
```

Vitest + Testing Library, covering `BenchmarkReport`, `ExperimentBuilder` and
`PathCostBreakdown`.

`web/src/test/setup.js` provides jest-dom matchers and a `ResizeObserver` stub —
jsdom has none and Recharts requires one. `vite.config.js` sets
`esbuild: { jsx: "automatic" }`; without it every test failed with "React is not
defined".

---

## 6. The claim verifier

```bash
python scripts/verify_claims.py
```

Not a unit test — a cross-check between the documentation and the artifacts. It
fails with a numbered list if:

- a model declared as shipping in `ml/model_registry.py` is missing from disk (or
  vice versa);
- a reward figure in the docs falls outside the measured range in
  `ml/results/rl_evaluation.json`;
- a scenario named in `experiments/scenarios.py` has no results file;
- a results file does not match the current schema;
- the documentation references a module path that no longer exists.

It exists because of one specific incident. `12_KNOWN_ISSUES.md` stated the PPO
agent's reward "improved from -77 to -61 (+21%)" with a best evaluation at
"-45.81". The committed `evaluations.npz` contained values from **-86.57 to
-99.67**, with the best at the *first* checkpoint. Nobody wrote those numbers
dishonestly — they came from a run that was never committed — but there was no
mechanism that could ever notice the divergence.

A document may still quote a refuted figure, which is how you write down what was
wrong, but only under an explicit `<!-- verify-claims: refuted -->` marker. It is
a marker rather than a turn of phrase deliberately: a phrase-based escape hatch
is one an author trips over by accident.

---

## 7. CI

`.github/workflows/ci.yml` runs four jobs on every push and pull request:

| Job | Steps |
|---|---|
| `backend-lint` | `ruff check .` |
| `backend-test` | pytest with coverage against a real PostgreSQL service, then the honesty gates, then `verify_claims.py` |
| `frontend` | `npm ci`, lint, test, build |
| `docker` | `docker compose build && up`, wait for `/health` to report **`ok`**, smoke-test the API, assert `/health/models` reports every model loaded, `down -v` |

The `docker` job is the important one: it is what catches a reviewer's very first
command failing, which is precisely what used to happen. It waits for
`"status": "ok"` rather than merely for the endpoint to respond — a container in
trouble reports `"degraded"`, and grepping for the key alone would pass on it.

Its final step is the one that would have caught the original model-path bug:
it asserts `/health/models` reports `status: ok`, meaning every declared model
was found **and** loaded. A silently absent checkpoint fails the build.

`.pre-commit-config.yaml` runs ruff and the frontend linter locally before a
commit.

---

## 8. What is not tested

Stated plainly, because an untested area that nobody names reads as a tested one.

- **The Docker path, locally.** CI has a `docker` job that builds and starts the
  full stack, so the path *is* covered there. But the environment this revision
  was developed in had no Docker daemon, so `docker compose up` was never run by
  hand during the work. If it fails on your machine, treat that as a bug worth
  reporting rather than a known limitation.
- **The database under load.** Persistence is exercised, concurrency is not.
- **The WebSocket under many clients.** Broadcast is exercised with one client.
- **Live probing.** `LiveProbeSource` is unit-tested against a fake `ping`, but
  nothing in CI probes a real host, and nothing should.
- **Model quality regressions.** The evaluation scripts measure quality and write
  it to `ml/results/`, but no test asserts a floor — retraining with worse
  hyperparameters would produce a worse model and a green build. `train_lstm.py`
  is the exception: it refuses to save a checkpoint that loses to persistence.
