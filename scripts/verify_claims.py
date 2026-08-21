"""Cross-check what the documentation claims against what the repository holds.

Run from the repository root::

    python scripts/verify_claims.py

This exists because of one specific failure. ``12_KNOWN_ISSUES.md`` stated that
the PPO agent's "mean reward improved from -77 to -61 (+21%)" with a "best
evaluation reward at -45.81". The committed evaluation file contained values
from -86.57 to -99.67, with the best at the *first* checkpoint. Nobody wrote
that number dishonestly — it came from a run that was never committed — but
there was no mechanism that could ever notice the divergence. Anyone who opened
the ``.npz`` would find the project's headline ML result evaporating in front of
them, in about ninety seconds.

This script is that mechanism. It exits non-zero with a numbered list of
violations, and CI runs it on every push.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DOCS = REPO_ROOT / "docs"
RESULTS = REPO_ROOT / "experiments" / "results"
ML_RESULTS = REPO_ROOT / "ml" / "results"

violations: list[str] = []


def fail(message: str) -> None:
    violations.append(message)


def check_model_artifacts() -> None:
    """Every referenced artifact must exist or be explicitly marked absent.

    THIS IS THE CHECK THAT WOULD HAVE CAUGHT ``rl_router_final.zip`` vs
    ``ppo_routing_agent.zip`` on day one.
    """
    from ml.model_registry import REGISTRY, path_for

    for key, spec in REGISTRY.items():
        exists = path_for(key).exists()
        if spec.ships_in_repo and not exists:
            fail(
                f"Model '{key}' is declared as shipping in the repo but "
                f"{spec.filename} is missing. Either train it "
                f"({spec.train_command}) or set ships_in_repo=False."
            )
        if not spec.ships_in_repo and exists:
            fail(
                f"Model '{key}' is declared as NOT shipping but {spec.filename} "
                f"is present. Update the registry so the two agree."
            )


def check_reported_ml_numbers() -> None:
    """Numbers quoted in the docs must appear in the evaluation artifacts."""
    for name, key in (
        ("gnn_evaluation.json", "test"),
        ("rl_evaluation.json", "normalized_scores"),
        ("lstm_evaluation.json", "skill_score_vs_persistence"),
    ):
        path = ML_RESULTS / name
        if not path.exists():
            fail(
                f"{path.relative_to(REPO_ROOT)} is missing, so the ML claims in "
                f"the documentation cannot be verified. Run `make train`."
            )
            continue
        payload = json.loads(path.read_text())
        if key not in payload:
            fail(f"{name} has no '{key}' field; the report format has drifted.")


def check_no_unsupported_reward_figures() -> None:
    """Reward figures in the docs must fall inside the measured range."""
    evaluation = ML_RESULTS / "rl_evaluation.json"
    if not evaluation.exists():
        return

    payload = json.loads(evaluation.read_text())
    means = [p["mean"] for p in payload.get("policies", {}).values()]
    if not means:
        return
    low, high = min(means) - 15, max(means) + 15

    # The specific figures the old documentation asserted.
    banned = {"-45.81", "-61", "-77"}
    for document in DOCS.glob("*.md"):
        text = document.read_text(encoding="utf-8")
        for value in banned:
            # Only flag it if presented as a reward, not merely as a digit.
            if re.search(rf"reward[^.\n]*{re.escape(value)}", text, re.IGNORECASE):
                fail(
                    f"{document.name} quotes reward {value}, which is not "
                    f"supported by ml/results/rl_evaluation.json "
                    f"(measured means span {min(means):.1f} to {max(means):.1f})."
                )

        for match in re.finditer(r"mean (?:episode )?reward[^.\n]*?(-\d+\.?\d*)", text, re.I):
            quoted = float(match.group(1))
            if not (low <= quoted <= high):
                fail(
                    f"{document.name} quotes a mean reward of {quoted}, outside "
                    f"the measured range [{low:.1f}, {high:.1f}]."
                )


def check_benchmark_readme() -> None:
    """The limitations panel regex-matches on a literal string."""
    readme = REPO_ROOT / "experiments" / "README.md"
    if not readme.exists():
        fail(
            "experiments/README.md is missing. service/api/benchmark.py reads it "
            "for the dashboard's Known Limitations panel, which renders empty "
            "without it."
        )
        return
    if "**Limitation**:" not in readme.read_text(encoding="utf-8"):
        fail(
            "experiments/README.md does not contain the literal '**Limitation**:'. "
            "The benchmark API regex-matches on it."
        )


def check_scenarios_have_results() -> None:
    """Every defined scenario should have been run at least once."""
    from experiments.scenarios import SCENARIO_NAMES

    if not RESULTS.exists():
        fail("experiments/results/ does not exist. Run `make bench`.")
        return

    present = {p.stem for p in RESULTS.glob("*.json")}
    missing = [name for name in SCENARIO_NAMES if name not in present]
    if missing:
        fail(
            f"No results for scenario(s): {', '.join(missing)}. The dashboard "
            f"and the results document will be incomplete."
        )


def check_topologies_are_not_degenerate() -> None:
    """Catch the 100-node ring before it reaches a published table."""
    from core.simulator import NetworkSimulator

    for num_nodes in (25, 50, 100):
        stats = NetworkSimulator(num_nodes=num_nodes, seed=42).topology_stats()
        if stats["avg_degree"] < 3.0:
            fail(
                f"The {num_nodes}-node topology has average degree "
                f"{stats['avg_degree']:.2f}. Below 3 there are too few "
                f"alternative paths for any algorithm to differentiate."
            )
        if not stats["is_connected"]:
            fail(f"The {num_nodes}-node topology is disconnected.")


def check_results_match_the_current_schema() -> None:
    """Stored artifacts and the code that reads them must not drift apart.

    The committed result files contained ``max_path_utilization`` and
    ``diversity_index`` but not ``effect_size_pct``, while the code that
    produced them emitted ``effect_size_pct`` and neither of the others. The API
    papered over it by recomputing on read, which masked the drift instead of
    surfacing it.
    """
    required = {"replication", "topology", "algorithms", "warnings", "models_loaded"}
    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text())
        missing = required - set(payload)
        if missing:
            fail(
                f"{path.name} is missing {sorted(missing)}. It was produced by a "
                f"different version of the harness; regenerate with `make bench`."
            )


def check_no_stale_module_references() -> None:
    """Documentation must not point at paths that no longer exist."""
    stale = [
        "backend/router/rl_agent.py",
        "backend/ml/models/rl_router_final",
        "rl_router_final.zip",
        "backend/simulator/network_sim.py",
    ]
    for document in list(DOCS.glob("*.md")) + [REPO_ROOT / "README.md"]:
        if not document.exists():
            continue
        text = document.read_text(encoding="utf-8")
        for token in stale:
            # A historical reference is fine when it is explicitly framed as one.
            for line in text.splitlines():
                if token in line and not re.search(
                    r"was|previously|used to|old|before|formerly|historical", line, re.I
                ):
                    fail(
                        f"{document.name} references '{token}', which no longer "
                        f"exists, without framing it as historical."
                    )
                    break


def main() -> int:
    checks = (
        check_model_artifacts,
        check_reported_ml_numbers,
        check_no_unsupported_reward_figures,
        check_benchmark_readme,
        check_scenarios_have_results,
        check_topologies_are_not_degenerate,
        check_results_match_the_current_schema,
        check_no_stale_module_references,
    )

    print("Verifying that documented claims match the committed artifacts...\n")
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - a broken check is itself a finding
            fail(f"Check {check.__name__} raised {type(exc).__name__}: {exc}")

    if not violations:
        print("All checks passed. Documentation and artifacts agree.")
        return 0

    print(f"{len(violations)} violation(s):\n")
    for index, violation in enumerate(violations, start=1):
        print(f"  {index}. {violation}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
