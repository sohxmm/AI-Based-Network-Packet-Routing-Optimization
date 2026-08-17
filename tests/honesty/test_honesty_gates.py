"""Assertions that fail loudly when a published claim stops being true.

The project already had good instincts here: ``is_fallback`` threaded through
the whole stack, a ``dijkstra_match_rate`` metric invented specifically to catch
its own models being degenerate, and a UI badge whose only job is to display
"Matches Dijkstra, no differentiation" on its own results.

What it did not have was *enforcement*. The guardrails reported problems and
nothing acted on them, which is how a documented training result ended up
contradicting the committed evaluation file for the life of the project. These
tests turn the guardrails into gates: if an algorithm becomes degenerate, or a
row is secretly a heuristic, or a metric goes structurally constant, CI fails.

They are expected to fail on results generated before the fixes. That is the
point — the failures are the bugs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"
LEARNED = {"gnn", "rl", "multi_agent"}
#: Bellman-Ford is mathematically required to match Dijkstra given identical
#: non-negative weights, and the constrained baseline is supposed to agree
#: whenever no constraint binds. Neither is a degeneracy.
DEGENERACY_EXEMPT = {"dijkstra", "bellman_ford", "constrained"}


def result_files() -> list[Path]:
    return sorted(RESULTS.glob("*.json"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payloads() -> list[tuple[str, dict]]:
    files = result_files()
    if not files:
        pytest.skip("No benchmark results. Generate them with: make bench")
    return [(f.name, load(f)) for f in files]


def test_results_exist_for_the_dashboard_to_display():
    assert result_files(), (
        "experiments/results/ is empty, so the dashboard's benchmark panel will "
        "render nothing. Run: python -m experiments.runner"
    )


def test_no_algorithm_is_silently_degenerate(payloads):
    """Degeneracy is allowed. Hiding it is not.

    An algorithm that reproduces Dijkstra's path almost every time adds no
    information, and the reader has to be told. Note that this is *expected* for
    a good learned ranker on a single additive objective: Dijkstra is provably
    optimal there, so converging to it is correct behaviour, not a defect. The
    finding only becomes dishonest if the results table presents that row as a
    distinct algorithm without saying so.

    The gate therefore requires a declaration, not an absence.
    """
    for name, data in payloads:
        warnings_text = " ".join(data.get("warnings", []))
        for algorithm, metrics in data["algorithms"].items():
            if algorithm in DEGENERACY_EXEMPT:
                continue
            rate = metrics.get("dijkstra_match_rate", 0.0)
            if rate > 0.95:
                assert algorithm in warnings_text, (
                    f"{algorithm} in {name} matches Dijkstra {rate:.1%} of the "
                    f"time but no warning declares it. Degeneracy must be "
                    f"reported, not left for the reader to notice."
                )


def test_reported_results_are_not_secretly_the_fallback(payloads):
    """A row produced entirely by the heuristic fallback is not an AI result."""
    for name, data in payloads:
        for algorithm, metrics in data["algorithms"].items():
            if algorithm not in LEARNED:
                continue
            rate = metrics.get("fallback_rate", 0.0)
            assert rate < 0.5, (
                f"{algorithm} in {name} used the heuristic fallback for "
                f"{rate:.0%} of decisions. Train the model or label this row a "
                f"heuristic, not {algorithm}."
            )


def test_no_metric_is_structurally_constant(payloads):
    """Catch hardcoded-broken metrics.

    ``diversity_index`` was 0.000 for every algorithm in every committed result
    file, because the aggregation read a ``path`` key that was never stored. A
    metric that is identical across algorithms is broken, not informative.
    """
    for name, data in payloads:
        for key in ("diversity_index", "mean_path_max_utilization"):
            values = [
                m.get(key) for m in data["algorithms"].values() if m.get(key) is not None
            ]
            if len(values) > 2:
                assert len(set(values)) > 1, (
                    f"{key} is identical ({values[0]}) for every algorithm in "
                    f"{name}. The metric is broken, not the algorithms."
                )


def test_max_utilization_is_not_pinned_at_one(payloads):
    """``max_path_utilization`` was a max over 20,000 samples, so always 1.000."""
    for name, data in payloads:
        for algorithm, metrics in data["algorithms"].items():
            value = metrics.get("p95_path_max_utilization")
            if value is not None:
                assert value < 0.999, (
                    f"{algorithm} in {name} reports p95 bottleneck {value}. A "
                    f"value pinned at 1.0 means an extreme statistic is being "
                    f"taken over the whole run."
                )


def test_p_values_are_not_exactly_zero(payloads):
    """p = 0.0 exactly is numerical underflow from pseudo-replication."""
    for name, data in payloads:
        for algorithm, metrics in data["algorithms"].items():
            comparison = metrics.get("comparison_vs_dijkstra") or {}
            p_value = comparison.get("wilcoxon_p_value")
            if p_value is not None:
                assert p_value != 0.0, (
                    f"{algorithm} in {name} reports p = 0.0 exactly. That is "
                    f"underflow from testing across correlated within-run "
                    f"decisions, not a valid p-value."
                )


def test_statistics_use_independent_runs(payloads):
    """The unit of replication must be a run, not a routing decision."""
    for name, data in payloads:
        replication = data.get("replication", {})
        assert replication.get("n_runs", 0) >= 2, (
            f"{name} reports {replication.get('n_runs')} runs. Statistics across "
            f"a single trajectory are pseudo-replicated."
        )
        for algorithm, metrics in data["algorithms"].items():
            comparison = metrics.get("comparison_vs_dijkstra")
            if comparison:
                assert comparison["n_runs"] == replication["n_runs"], (
                    f"{algorithm} in {name} compared {comparison['n_runs']} runs "
                    f"but the scenario declares {replication['n_runs']}."
                )


def test_effect_sizes_are_real_effect_sizes(payloads):
    """A percent difference in means is not an effect size."""
    for name, data in payloads:
        for algorithm, metrics in data["algorithms"].items():
            comparison = metrics.get("comparison_vs_dijkstra")
            if not comparison:
                continue
            assert "cliffs_delta" in comparison, (
                f"{algorithm} in {name} has no Cliff's delta. Reporting only a "
                f"percent difference and calling it an effect size is what the "
                f"audit flagged."
            )
            assert -1.0 <= comparison["cliffs_delta"] <= 1.0
            assert comparison.get("effect_magnitude") in {
                "negligible",
                "small",
                "medium",
                "large",
            }


def test_topology_is_recorded_and_not_a_ring(payloads):
    """Every result must say what was actually tested."""
    for name, data in payloads:
        topology = data.get("topology")
        assert topology, f"{name} does not record the topology it ran on."
        assert topology["avg_degree"] >= 3.0, (
            f"{name} ran on a degree-{topology['avg_degree']:.1f} topology. "
            f"Below degree 3 there are too few alternative paths for any "
            f"algorithm to differentiate."
        )
        assert topology["is_connected"]


def test_warnings_block_is_present(payloads):
    """The reader must not have to notice the caveats themselves."""
    for name, data in payloads:
        assert "warnings" in data, (
            f"{name} has no warnings block. Guardrail findings must be emitted "
            f"into the artifact, not left for the reader to derive."
        )


def test_model_load_status_is_recorded(payloads):
    """Whether the AI was actually running has to be part of the record."""
    for name, data in payloads:
        assert "models_loaded" in data, (
            f"{name} does not record which models were loaded. Without it, "
            f"nobody can tell whether an AI row reflects a trained policy."
        )


def test_predictive_variants_differ_from_their_base(payloads):
    """gnn_predictive identical to gnn means the forecaster never ran.

    In every previously committed result file they were byte-identical, because
    the LSTM artifact was absent and the forecast builder returned None on every
    call. Two of the eight benchmarked "algorithms" were duplicate columns.
    """
    for name, data in payloads:
        algorithms = data["algorithms"]
        for base in ("gnn", "rl"):
            predictive = f"{base}_predictive"
            if base in algorithms and predictive in algorithms:
                assert (
                    algorithms[base]["mean_latency"]
                    != algorithms[predictive]["mean_latency"]
                ), (
                    f"{predictive} is identical to {base} in {name}: the "
                    f"forecast had no effect, so predictive mode did not run."
                )
