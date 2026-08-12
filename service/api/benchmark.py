"""Benchmark results API.

The results directory is configurable so the harness can live outside the
service image. It used to be found by walking ``Path(__file__).parents[1]``,
which coupled the web layer to the experiment layer's location on disk.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(
    os.getenv("BENCHMARK_RESULTS_DIR", _REPO_ROOT / "experiments" / "results")
)
BENCHMARK_README = _REPO_ROOT / "experiments" / "README.md"
ROOT_README = _REPO_ROOT / "README.md"


def _sanitize(value: Any) -> Any:
    """Replace NaN/Inf with None recursively so the payload is valid JSON."""
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _load_scenario(scenario: str) -> dict | None:
    """Read one scenario's result file."""
    path = RESULTS_DIR / f"{scenario}.json"
    if not path.exists():
        # Fall back to the timestamped naming the older harness produced.
        candidates = sorted(RESULTS_DIR.glob(f"{scenario}_*.json"), reverse=True)
        if not candidates:
            return None
        path = candidates[0]
    try:
        return _sanitize(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def available_scenarios() -> list[str]:
    """Scenario names that actually have a result file on disk."""
    if not RESULTS_DIR.exists():
        return []
    names = {p.stem.split("_20")[0] for p in RESULTS_DIR.glob("*.json")}
    return sorted(names)


def load_known_limitations() -> dict[str, str]:
    """Read the limitations text the dashboard displays.

    ``experiments/README.md`` is *required*: this function is what populates the
    dashboard's "Known Limitations" panel, and because that file did not exist,
    a feature built specifically to be honest rendered empty.
    """
    limitations: dict[str, str] = {}

    if BENCHMARK_README.exists():
        limitations["benchmark_readme"] = BENCHMARK_README.read_text(encoding="utf-8")
    else:
        limitations["benchmark_readme"] = (
            "experiments/README.md is missing. It documents what the benchmark "
            "measures and its known limitations, and this panel reads it."
        )
        logger.warning("experiments/README.md is missing; limitations panel degraded.")

    if ROOT_README.exists():
        text = ROOT_README.read_text(encoding="utf-8")
        match = re.search(r"\*\*Limitation\*\*:(.+?)(?=\n\n|\n>|\n---)", text, re.DOTALL)
        if match:
            limitations["root_readme_limitation"] = match.group(0).strip()

    return limitations


@router.get("/results")
def get_benchmark_results() -> dict:
    """Every scenario's results, plus the known-limitations text."""
    scenarios = {}
    for name in available_scenarios():
        data = _load_scenario(name)
        if data:
            scenarios[name] = data

    return {
        "scenarios": scenarios,
        "available": list(scenarios),
        "known_limitations": load_known_limitations(),
    }


@router.get("/scenarios")
def list_scenarios() -> dict:
    """Scenario names with results, and the full catalogue of defined ones."""
    from experiments.scenarios import SCENARIOS

    return {
        "with_results": available_scenarios(),
        "defined": [scenario.as_dict() for scenario in SCENARIOS.values()],
    }


@router.get("/results/{scenario}")
def get_benchmark_results_by_scenario(scenario: str) -> dict:
    """One scenario's results."""
    data = _load_scenario(scenario)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No benchmark results for {scenario!r}. "
                f"Available: {available_scenarios()}. "
                "Generate them with: python -m experiments.runner"
            ),
        )
    return data | {"known_limitations": load_known_limitations()}


__all__ = ["RESULTS_DIR", "available_scenarios", "load_known_limitations", "router"]
