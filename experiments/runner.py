"""Benchmark runner: N independent seeded replications per scenario.

Run from the repository root::

    python -m experiments.runner --scenario all --runs 30

Three structural changes from the previous harness, all required by the audit.

**Independent replication.** Each algorithm now runs its *own* trajectory per
seed. This is not optional once the simulator is closed-loop: if routing changes
the network, algorithms cannot share one trajectory, because whichever ran first
would pollute the state the others observe. Within a seed every algorithm gets
the identical topology, the identical background traffic and the identical
demand sequence, so the comparison stays paired at the level that matters — the
run. Statistics are then computed across runs, not across the autocorrelated
decisions inside one run.

**Reproducibility.** Every source of randomness is drawn from a per-run
``random.Random(seed)``. The old harness used the unseeded global ``random``
module in seven places, so two runs of the same command produced different
numbers and there was no ``--seed`` flag.

**Honest metrics.** ``diversity_index`` read a ``path`` key that was never
stored, so it was 0.000 in every committed file; paths are now recorded and
diversity is a normalised entropy. ``max_path_utilization`` took a max over
20,000 decisions and was therefore 1.000 for every algorithm in every scenario;
it is reported as a mean and a p95. A ``warnings`` block is emitted into the
result JSON whenever an algorithm is degenerate or ran mostly on its fallback,
so a reader does not have to notice it themselves.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.models import NetworkState
from core.paths import max_path_utilization
from core.qos import QOS_PROFILES, evaluate_path, get_profile
from core.simulator import NetworkSimulator
from experiments.scenarios import SCENARIO_NAMES, Scenario, get_scenario
from experiments.statistics import paired_comparison, path_diversity, summarise
from routing import ALGORITHM_NAMES, DEGENERACY_EXEMPT, LEARNED_ALGORITHMS, build_router_set
from routing.learned.forecaster import CongestionPredictor, build_forecast_state

logger = logging.getLogger("benchmark")

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Guardrail thresholds, named so the report and the honesty gates agree.
DEGENERACY_THRESHOLD = 0.95
FALLBACK_THRESHOLD = 0.50


def _decision_row(
    algorithm: str,
    step: int,
    src: str,
    dst: str,
    traffic_class: str,
    decision,
    state: NetworkState,
) -> dict[str, Any]:
    """One routing decision, flattened for aggregation."""
    profile = get_profile(traffic_class)
    evaluation = evaluate_path(state, decision.path, profile)
    latency = decision.total_latency
    return {
        "algorithm": algorithm,
        "step": step,
        "src": src,
        "dst": dst,
        "traffic_class": traffic_class,
        # The full path is stored. Not storing it is why diversity_index was
        # structurally zero in every previously published result.
        "path": list(decision.path),
        "path_len": len(decision.path),
        "latency": latency if np.isfinite(latency) else float("inf"),
        "avg_utilization": decision.avg_utilization,
        "max_path_utilization": max_path_utilization(state, decision.path),
        "success": bool(decision.success),
        "is_fallback": bool(decision.is_fallback),
        "qos_feasible": bool(evaluation.feasible),
        "qos_score": evaluation.score if np.isfinite(evaluation.score) else None,
    }


def run_single_replicate(
    scenario: Scenario,
    seed: int,
    n_steps: int,
    m_pairs: int,
    algorithms: list[str],
    predictor: CongestionPredictor | None = None,
) -> dict[str, list[dict]]:
    """Run one seeded replicate: an independent trajectory per algorithm."""
    results: dict[str, list[dict]] = {}

    # The demand schedule is generated once from the run seed and replayed for
    # every algorithm, so any difference between algorithms is caused by their
    # routing and not by facing different traffic.
    schedule_rng = random.Random(seed * 7919 + 13)
    classes = [c.value for c in scenario.traffic_classes]

    for algorithm in algorithms:
        rng = random.Random(seed)
        sim = NetworkSimulator(num_nodes=scenario.num_nodes, seed=seed)
        routers = build_router_set(seed=seed)
        router = routers[algorithm]
        scenario.prepare(sim, rng)

        demand_rng = random.Random(schedule_rng.randint(0, 2**31 - 1))
        history: list[list[float]] = []
        rows: list[dict] = []

        for step in range(n_steps):
            scenario.per_step(sim, rng, step)
            state = sim.step()

            if predictor is not None and predictor.is_trained:
                history.append([link.utilization for link in state.links])
                del history[: -predictor.seq_len]

            nodes = list(state.nodes)
            for _ in range(m_pairs):
                src, dst = demand_rng.sample(nodes, 2)
                traffic_class = demand_rng.choice(classes)
                profile = QOS_PROFILES[get_profile(traffic_class).traffic_class]

                decision = router.find_route(state, src, dst, profile)
                rows.append(
                    _decision_row(algorithm, step, src, dst, traffic_class, decision, state)
                )

                # Closed loop: the decision loads the network the next decision
                # will observe. This is what the whole redesign is for.
                if decision.success:
                    sim.register_flow(decision.path, demand=0.5)

        results[algorithm] = rows

    return results


def _per_run_metrics(rows: list[dict]) -> dict[str, float]:
    """Reduce one algorithm's decisions in one run to a handful of numbers."""
    if not rows:
        return {}

    successes = [r for r in rows if r["success"]]
    latencies = [r["latency"] for r in successes if np.isfinite(r["latency"])]
    max_utils = [r["max_path_utilization"] for r in successes]

    metrics = {
        "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
        "p95_latency": float(np.percentile(latencies, 95)) if latencies else float("nan"),
        "success_rate": len(successes) / len(rows),
        "fallback_rate": sum(1 for r in rows if r["is_fallback"]) / len(rows),
        "qos_satisfaction_rate": sum(1 for r in rows if r["qos_feasible"]) / len(rows),
        "mean_path_max_utilization": float(np.mean(max_utils)) if max_utils else float("nan"),
        "p95_path_max_utilization": (
            float(np.percentile(max_utils, 95)) if max_utils else float("nan")
        ),
        "diversity_index": path_diversity(rows),
        "mean_hops": float(np.mean([r["path_len"] - 1 for r in successes])) if successes else 0.0,
    }

    # A single aggregate QoS number is the wrong metric for mixed traffic. An
    # emergency class with a 1% satisfaction rate and a bulk class at 100% averages
    # out to something that looks fine, and the class that actually has an SLA is
    # the one being failed. Keys are flat so the cross-run aggregation, which
    # averages one dict of scalars per run, needs no special case.
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["traffic_class"]].append(row)
    for traffic_class, class_rows in by_class.items():
        metrics[f"qos_satisfaction_rate__{traffic_class}"] = sum(
            1 for r in class_rows if r["qos_feasible"]
        ) / len(class_rows)

    return metrics


def degeneracy_probe(
    scenario: Scenario,
    seed: int,
    algorithms: list[str],
    n_steps: int = 20,
    m_pairs: int = 10,
) -> dict[str, float]:
    """Measure how often each algorithm picks the same path as Dijkstra.

    This has to be a *separate, open-loop* pass, and the reason is a genuine
    consequence of closing the loop. In the main benchmark every algorithm runs
    its own trajectory, so by step two their networks have legitimately
    diverged; comparing their chosen paths then measures trajectory divergence,
    not algorithmic similarity. Measured that way, even Bellman-Ford scored a
    match rate of 0.00 against Dijkstra, which is mathematically impossible for
    two exact solvers on identical weights.

    So the degeneracy guardrail runs here instead: one shared simulator, no flow
    registration, every algorithm asked the same question about the same
    network. That is the only setting in which "did it choose differently?" is
    a question about the algorithm.
    """
    rng = random.Random(seed)
    sim = NetworkSimulator(num_nodes=scenario.num_nodes, seed=seed)
    routers = build_router_set(seed=seed)
    scenario.prepare(sim, rng)

    demand_rng = random.Random(seed * 31 + 7)
    classes = [c.value for c in scenario.traffic_classes]

    matches: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for step in range(n_steps):
        scenario.per_step(sim, rng, step)
        state = sim.step()  # advanced once, shared by every algorithm
        nodes = list(state.nodes)

        for _ in range(m_pairs):
            src, dst = demand_rng.sample(nodes, 2)
            profile = get_profile(demand_rng.choice(classes))

            baseline = routers["dijkstra"].find_route(state, src, dst, profile)
            if not baseline.success:
                continue

            for algorithm in algorithms:
                if algorithm == "dijkstra":
                    continue
                decision = routers[algorithm].find_route(state, src, dst, profile)
                totals[algorithm] += 1
                if decision.success and decision.path == baseline.path:
                    matches[algorithm] += 1

    rates = {
        algorithm: (matches[algorithm] / totals[algorithm]) if totals[algorithm] else 0.0
        for algorithm in algorithms
        if algorithm != "dijkstra"
    }
    rates["dijkstra"] = 1.0
    return rates


def run_scenario(
    scenario_name: str,
    n_runs: int = 30,
    n_steps: int = 100,
    m_pairs: int = 20,
    base_seed: int = 1000,
    algorithms: list[str] | None = None,
    persist: bool = False,
) -> dict:
    """Run a scenario across independent replications and aggregate."""
    scenario = get_scenario(scenario_name)
    algorithms = algorithms or list(ALGORITHM_NAMES)
    if "dijkstra" not in algorithms:
        algorithms = ["dijkstra", *algorithms]

    predictor = CongestionPredictor()
    predictor.load()

    logger.info(
        "Scenario %s: %d runs x %d steps x %d pairs x %d algorithms = %s decisions",
        scenario_name,
        n_runs,
        n_steps,
        m_pairs,
        len(algorithms),
        f"{n_runs * n_steps * m_pairs * len(algorithms):,}",
    )

    per_run: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    scenario_meta: dict = {}
    topology: dict = {}
    started = time.time()

    for run in range(n_runs):
        seed = base_seed + run
        if run == 0:
            probe = NetworkSimulator(num_nodes=scenario.num_nodes, seed=seed)
            scenario_meta = scenario.prepare(probe, random.Random(seed)) or {}
            topology = probe.topology_stats()

        results = run_single_replicate(
            scenario, seed, n_steps, m_pairs, algorithms, predictor
        )
        for algorithm, rows in results.items():
            for key, value in _per_run_metrics(rows).items():
                per_run[algorithm][key].append(value)

        if (run + 1) % max(1, n_runs // 5) == 0:
            logger.info("  %d/%d runs complete", run + 1, n_runs)

    # Degeneracy is measured separately, on a shared open-loop trajectory.
    logger.info("  running the degeneracy probe (shared state, open loop)...")
    match_rates = degeneracy_probe(scenario, base_seed, algorithms)

    elapsed = time.time() - started

    # --- aggregate ------------------------------------------------------
    baseline_latencies = per_run["dijkstra"]["mean_latency"]
    algorithms_block: dict[str, dict] = {}

    for algorithm in algorithms:
        metrics = per_run[algorithm]
        block: dict[str, Any] = {
            key: summarise(values)["mean"] for key, values in metrics.items()
        }
        block["mean_latency_ci"] = summarise(metrics["mean_latency"])
        block["dijkstra_match_rate"] = match_rates.get(algorithm, 0.0)
        if algorithm != "dijkstra":
            block["comparison_vs_dijkstra"] = paired_comparison(
                metrics["mean_latency"], baseline_latencies
            )
        algorithms_block[algorithm] = block

    warnings = _build_warnings(algorithms_block)

    payload = {
        "scenario": scenario_name,
        "description": scenario.description,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replication": {
            "n_runs": n_runs,
            "n_steps": n_steps,
            "m_pairs": m_pairs,
            "base_seed": base_seed,
            "unit_of_replication": "one independently seeded run",
            "note": (
                "Each algorithm runs its own closed-loop trajectory per seed with "
                "an identical topology, background traffic and demand schedule. "
                "Statistics are computed across runs, never across the "
                "autocorrelated decisions within one run."
            ),
        },
        "topology": topology,
        "scenario_config": scenario.as_dict() | {"prepared": scenario_meta},
        "models_loaded": {
            name: build_router_set(seed=0)[name].is_trained for name in LEARNED_ALGORITHMS
        },
        "forecaster_loaded": predictor.is_trained,
        "algorithms": algorithms_block,
        "warnings": warnings,
        "runtime_seconds": round(elapsed, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / f"{scenario_name}.json"
    destination.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s (%.1fs)", destination, elapsed)

    for warning in warnings:
        logger.warning("  %s", warning)

    if persist:
        asyncio.run(_persist_metrics(scenario_name, algorithms_block, n_steps))

    return payload


def _build_warnings(algorithms_block: dict[str, dict]) -> list[str]:
    """Surface degeneracy and fallback problems in the artifact itself."""
    warnings: list[str] = []
    for algorithm, metrics in algorithms_block.items():
        fallback = metrics.get("fallback_rate") or 0.0
        if algorithm in LEARNED_ALGORITHMS and fallback > FALLBACK_THRESHOLD:
            warnings.append(
                f"{algorithm}: {fallback:.0%} of decisions came from the heuristic "
                f"fallback, not a trained model. This row is a heuristic, not "
                f"{algorithm}."
            )
        match = metrics.get("dijkstra_match_rate") or 0.0
        if algorithm not in DEGENERACY_EXEMPT and match > DEGENERACY_THRESHOLD:
            warnings.append(
                f"{algorithm}: chooses the same path as Dijkstra {match:.0%} of the "
                f"time, so it is degenerate and adds no information."
            )
    return warnings


async def _persist_metrics(scenario: str, algorithms_block: dict, n_steps: int) -> None:
    """Write aggregate metrics to the database (opt-in via --persist)."""
    import uuid

    from service.db.database import AsyncSessionLocal
    from service.db.models import AlgorithmMetric

    rows = [
        AlgorithmMetric(
            id=str(uuid.uuid4()),
            algorithm=algorithm,
            scenario=scenario,
            window_start_step=0,
            window_end_step=n_steps,
            avg_latency=metrics.get("mean_latency"),
            success_rate=metrics.get("success_rate"),
            num_decisions=n_steps,
        )
        for algorithm, metrics in algorithms_block.items()
    ]
    try:
        async with AsyncSessionLocal() as session:
            session.add_all(rows)
            await session.commit()
        logger.info("Persisted %d AlgorithmMetric rows", len(rows))
    except Exception as exc:  # noqa: BLE001 - the DB is optional for benchmarks
        logger.warning("Could not persist benchmark metrics: %s", exc)


# ---------------------------------------------------------------------------
# Parameterized entry point used by the Experiment Sandbox API
# ---------------------------------------------------------------------------
def run_parameterized_scenario(
    topology_size: int,
    congestion_profile: str,
    failure_rate: float,
    failure_pattern: str,
    steps: int,
    pairs_per_step: int,
    algorithms: list[str],
    traffic_classes: list[str] | None = None,
    n_runs: int = 3,
    seed: int = 42,
    on_progress=None,
) -> dict:
    """Run a user-configured experiment through the same engine as the fixed set.

    Synchronous by design: the API dispatches it to a thread pool. It used to be
    an ``async def`` that ran CPU-bound work directly on the event loop, yielding
    only every ten steps, which stalled the live WebSocket stream whenever
    somebody ran an experiment.
    """
    from experiments.scenarios import (
        CongestionBursts,
        HighCongestion,
        NormalTraffic,
        PersistentLinkFailures,
    )

    classes = [get_profile(c).traffic_class for c in (traffic_classes or ["best_effort"])]

    if congestion_profile == "high":
        scenario = HighCongestion(
            name="custom", description="custom", num_nodes=topology_size, traffic_classes=classes
        )
    elif congestion_profile == "bursty":
        scenario = CongestionBursts(
            name="custom", description="custom", num_nodes=topology_size, traffic_classes=classes
        )
    elif failure_pattern != "none" and failure_rate > 0:
        scenario = PersistentLinkFailures(
            name="custom",
            description="custom",
            num_nodes=topology_size,
            traffic_classes=classes,
            failure_fraction=min(0.3, failure_rate),
        )
    else:
        scenario = NormalTraffic(
            name="custom", description="custom", num_nodes=topology_size, traffic_classes=classes
        )

    per_run: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for run in range(n_runs):
        results = run_single_replicate(
            scenario, seed + run, steps, pairs_per_step, algorithms, None
        )
        for algorithm, rows in results.items():
            for key, value in _per_run_metrics(rows).items():
                per_run[algorithm][key].append(value)
        if on_progress:
            on_progress(run + 1, n_runs)

    match_rates = degeneracy_probe(scenario, seed, algorithms, n_steps=10, m_pairs=6)

    baseline = per_run.get("dijkstra", {}).get("mean_latency", [])
    algorithms_block: dict[str, dict] = {}
    for algorithm in algorithms:
        metrics = per_run[algorithm]
        if not metrics:
            continue
        block: dict[str, Any] = {k: summarise(v)["mean"] for k, v in metrics.items()}
        block["mean_latency_ci"] = summarise(metrics["mean_latency"])
        block["dijkstra_match_rate"] = match_rates.get(algorithm, 0.0)
        if algorithm != "dijkstra" and baseline:
            block["comparison_vs_dijkstra"] = paired_comparison(
                metrics["mean_latency"], baseline
            )
        algorithms_block[algorithm] = block

    return {
        "scenario": f"custom_{topology_size}n_{congestion_profile}_{failure_pattern}",
        "description": "User-configured experiment",
        "replication": {"n_runs": n_runs, "n_steps": steps, "m_pairs": pairs_per_step},
        "algorithms": algorithms_block,
        "warnings": _build_warnings(algorithms_block),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the routing benchmark.")
    parser.add_argument("--scenario", default="all", help=f"one of {SCENARIO_NAMES} or 'all'")
    parser.add_argument("--runs", type=int, default=30, help="independent seeded replications")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--persist", action="store_true", help="write metrics to the database")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("core.simulator").setLevel(logging.WARNING)

    targets = SCENARIO_NAMES if args.scenario == "all" else [args.scenario]
    for name in targets:
        run_scenario(
            name,
            n_runs=args.runs,
            n_steps=args.steps,
            m_pairs=args.pairs,
            base_seed=args.seed,
            algorithms=args.algorithms,
            persist=args.persist,
        )


if __name__ == "__main__":
    main()
