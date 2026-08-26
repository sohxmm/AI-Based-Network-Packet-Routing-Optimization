# 15. Gotchas

Things that will cost you an hour if nobody tells you. Every entry here is
something that actually cost someone an hour.

---

## Setup

**`docker compose up` needs a `.env`, but you do not have to make one.**
Run `make up` — it copies `.env.example` for you. This used to be a hard failure:
three services declared `env_file: .env`, `.env` is gitignored, and nothing
created it, so the very first command a reviewer ran did not work.

**`pytest` is in `service/requirements-dev.txt`, not `requirements.txt`.**
`pip install -r service/requirements.txt` gives you a runnable service and an
unrunnable test suite.

**Run everything from the repository root.** `pytest.ini` sets `pythonpath = .`
and the Docker image sets `PYTHONPATH=/app`. There are no `sys.path.insert`
hacks left — there were seven — so a wrong working directory shows up as an
`ImportError` rather than mysteriously working.

**`tensorboard` is optional.** `train_rl` checks whether it is importable and
says so if not. It will not silently point you at an empty log directory, which
the previous version did.

---

## Running it

**The backend must run with exactly one uvicorn worker.**
`service/state.py` holds a module-level singleton network source. A second worker
owns a *different* network, and the dashboard flickers between two realities. The
Dockerfile pins `--workers 1`; if you launch it by hand, do the same.

**PostgreSQL is optional.** Without it the app runs, logs a warning, and degrades
history and metrics to live-state estimates. Nothing else breaks. If you want the
database, `make up` starts it.

**The database is published on host port 5433, not 5432.** `.env.example` is
correct for both modes and says which is which. The previous version's
`DATABASE_URL` was wrong for Docker *and* wrong for local development.

**A green `/health` used to be a lie.** It now reports
`simulator_last_tick_age_s`. If that is `null` or large, the background loop is
not running even though the process is up.

---

## Models

**Check `/health/models` before believing any AI result.**

```bash
curl localhost:8000/health/models | python -m json.tool
```

This is the single most useful diagnostic in the project. Its existence is the
direct result of three of four AI features silently serving heuristics for months
because their artifacts were missing or misnamed.

**Trained checkpoints are committed on purpose.** About 4 MB. `.gitignore` says
so explicitly. Do not "clean up" `ml/checkpoints/` — a fresh clone would then
produce a demo where every AI row is a fallback, which is exactly the failure
this project was rebuilt to eliminate.

**Retraining changes the published numbers.** `make bench` with the committed
checkpoints reproduces the results exactly. `make train` will not. If you retrain,
re-run `make bench` and `make verify`.

**Model paths live only in `ml/model_registry.py`.** If you add a model, register
it there. Nothing else has a hardcoded path, and that is deliberate.

**A checkpoint from before an observation-space change will not load**, and that
is the correct behaviour. The router detects the width mismatch, logs a warning
naming the retrain command, and falls back — flagged as a fallback. It does not
reshape into nonsense.

---

## The simulator

**It is closed-loop.** `register_flow()` means routing decisions raise
utilization on the links they use. Two consequences:

- The benchmark cannot share one trajectory across algorithms. Whichever ran
  first would pollute the state the others observe, so each gets its own.
- `dijkstra_match_rate` therefore cannot be computed from the main run — the
  networks legitimately diverge. It is measured by a separate shared-state
  open-loop probe. Measured the wrong way, even Bellman-Ford scores 0.00 against
  Dijkstra, which is impossible for two exact solvers.

**The 100-node topology used to be a ring.** Degree 2, diameter 50, exactly two
paths between any pair. Results generated before that fix are not comparable to
anything. It is now degree 4, diameter 8.

**Utilization is AR(1) around a diurnal cycle, not a random walk.** If you change
it back, the LSTM's task becomes learning the identity function and its skill
score collapses to zero — correctly.

**If you add a term to the utilization update, work out its steady state.** The
recursion is `u ← a·u + (1−a)·offered + noise` with `a = 0.85`. A term added
*outside* the `(1−a)` factor is amplified tenfold at steady state. That mistake
saturated the entire network on the first attempt.

---

## The benchmark

**The unit of replication is one seeded run, not one routing decision.** Testing
across the decisions inside a run is pseudo-replication; it produced
`p = 0.0` in every previously published file, which is underflow rather than
evidence.

**Degeneracy is expected and must be declared, not avoided.** A good learned
ranker on a single additive objective *converges to Dijkstra*, because Dijkstra
is optimal there. The honesty gate requires a warning in the results, not an
absence of degeneracy.

**Bellman-Ford and `constrained` are exempt from the degeneracy guardrail.**
Matching Dijkstra is mathematically required for the first and intended for the
second.

**ACO is roughly 10× slower than everything else** — 600 ant walks per decision.
That cost is not reflected in the latency metric, which measures the path rather
than the time taken to find it.

**`experiments/README.md` is load-bearing.** `service/api/benchmark.py` reads it
for the dashboard's Known Limitations panel and regex-matches on the literal
string `**Limitation**:`. Delete or reword that and the panel goes empty, which is
how it was for the whole life of the previous version.

---

## Live probing

**It is off unless `LIVE_PROBE_ENABLED=1`.** It measures real hosts.

**Its star topology cannot benchmark routing.** One path per destination means no
algorithm can differentiate. Live mode demonstrates telemetry and congestion
detection. Do not publish a routing comparison from it.

**Utilization is inferred from RTT inflation.** It is a standard proxy and a
coarse one. ICMP is de-prioritised or rate-limited by many networks, so a high
reading may be router policy rather than congestion.

---

## Frontend

**`npm run lint` runs at `--max-warnings 0`.** It will fail on a warning. It also
used to fail *immediately* with "no configuration found", because four ESLint
plugins were installed and no config file existed anywhere.

**Chart colours are validated, not chosen.** The palette in
`web/src/utils/colorScales.js` passes colour-vision-deficiency separation,
lightness-band and contrast checks in both light and dark mode. The previous one
did not — its worst adjacent pair had a CVD delta-E of 3.0 and one series read as
grey. If you change a colour, re-validate rather than eyeball it.

**Identity colour follows the algorithm, never its rank.** Filtering the
comparison must not repaint the survivors.
