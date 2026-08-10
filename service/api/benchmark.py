"""Benchmark results API — exposes Phase 3 benchmark data via REST endpoints."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Paths to benchmark data and limitation docs
_BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
_RESULTS_DIR = _BENCHMARK_DIR / "results"
_BENCHMARK_README = _BENCHMARK_DIR / "README.md"
_ROOT_README = Path(__file__).resolve().parents[2] / "README.md"

KNOWN_SCENARIOS = [
    "normal_traffic",
    "high_congestion",
    "link_failures_5_10pct",
    "congestion_bursts",
    "large_topology_100_nodes",
]


def _parse_json_with_nan(text: str) -> dict:
    """Parse JSON text that may contain bare NaN values (not valid JSON)."""
    return json.loads(text, parse_constant=lambda c: float("nan") if c == "NaN" else None)


def _sanitize_for_json(value: Any) -> Any:
    """Replace NaN/Inf floats with None for JSON-safe serialization."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _load_scenario_from_json(scenario: str) -> dict | None:
    """Load the most recent benchmark result JSON file for a scenario."""
    if not _RESULTS_DIR.exists():
        return None

    # Find files matching the scenario prefix, take the most recent
    pattern = f"{scenario}_*.json"
    files = sorted(_RESULTS_DIR.glob(pattern), reverse=True)
    if not files:
        return None

    with open(files[0], "r") as f:
        return _parse_json_with_nan(f.read())


def _compute_effect_size_pct(algo_mean: float, dijkstra_mean: float) -> float | None:
    """Compute effect size as percentage difference vs Dijkstra.

    Positive means the algorithm has higher latency (worse).
    Negative means lower latency (better).
    """
    if dijkstra_mean == 0 or math.isnan(dijkstra_mean):
        return None
    diff = (algo_mean - dijkstra_mean) / dijkstra_mean * 100
    return round(diff, 2)


def _format_algorithm_metrics(scenario_data: dict) -> dict:
    """Transform raw scenario JSON into the API response shape with effect_size_pct."""
    algos = scenario_data.get("algorithms", {})
    dijkstra_mean = algos.get("dijkstra", {}).get("mean_latency", 0.0)

    result = {}
    for algo, metrics in algos.items():
        algo_mean = metrics.get("mean_latency", 0.0)
        effect = _compute_effect_size_pct(algo_mean, dijkstra_mean) if algo != "dijkstra" else 0.0

        result[algo] = {
            "mean_latency": _sanitize_for_json(metrics.get("mean_latency", 0.0)),
            "p95_latency": _sanitize_for_json(metrics.get("p95_latency", 0.0)),
            "util_variance": _sanitize_for_json(metrics.get("util_variance", 0.0)),
            "success_rate": _sanitize_for_json(metrics.get("success_rate", 0.0)),
            "fallback_rate": _sanitize_for_json(metrics.get("fallback_rate", 0.0)),
            "dijkstra_match_rate": _sanitize_for_json(metrics.get("dijkstra_match_rate", 0.0)),
            "wilcoxon_p_value": _sanitize_for_json(metrics.get("wilcoxon_p_value")),
            "effect_size_pct": effect,
        }

    return result


def _load_known_limitations() -> dict:
    """Load limitation text from benchmark README and root README."""
    limitations = {}

    if _BENCHMARK_README.exists():
        limitations["benchmark_readme"] = _BENCHMARK_README.read_text(encoding="utf-8")

    if _ROOT_README.exists():
        root_text = _ROOT_README.read_text(encoding="utf-8")
        # Extract the limitation note from the MARL section
        match = re.search(
            r"\*\*Limitation\*\*:(.+?)(?=\n\n|\n>|\n---)",
            root_text,
            re.DOTALL,
        )
        if match:
            limitations["root_readme_limitation"] = match.group(0).strip()
        else:
            # Fall back to the full MARL paragraph if the exact pattern doesn't match
            match2 = re.search(
                r"### Multi-Agent.*?\n\n(.*?Limitation.*?)(?=\n\n>|\n---)",
                root_text,
                re.DOTALL,
            )
            if match2:
                limitations["root_readme_limitation"] = match2.group(1).strip()

    return limitations


@router.get("/results")
def get_benchmark_results() -> dict:
    """Return benchmark metrics for all scenarios.

    Sources data from JSON result files in backend/benchmark/results/.
    Includes per-algorithm metrics with effect_size_pct (% difference vs Dijkstra)
    and known-limitations text from project documentation.
    """
    scenarios = {}

    for scenario_name in KNOWN_SCENARIOS:
        data = _load_scenario_from_json(scenario_name)
        if data:
            scenarios[scenario_name] = {
                "scenario": scenario_name,
                "n_steps": data.get("n_steps"),
                "m_pairs": data.get("m_pairs"),
                "algorithms": _format_algorithm_metrics(data),
            }

    return {
        "scenarios": scenarios,
        "known_limitations": _load_known_limitations(),
    }


@router.get("/results/{scenario}")
def get_benchmark_results_by_scenario(scenario: str) -> dict:
    """Return benchmark metrics for a single scenario.

    Falls back to JSON result files. Includes effect_size_pct and limitations.
    """
    if scenario not in KNOWN_SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario: {scenario}. Available: {KNOWN_SCENARIOS}",
        )

    data = _load_scenario_from_json(scenario)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No benchmark results found for scenario: {scenario}",
        )

    return {
        "scenario": scenario,
        "n_steps": data.get("n_steps"),
        "m_pairs": data.get("m_pairs"),
        "algorithms": _format_algorithm_metrics(data),
        "known_limitations": _load_known_limitations(),
    }
