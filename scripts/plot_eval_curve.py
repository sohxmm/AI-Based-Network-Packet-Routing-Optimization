"""Plot the PPO evaluation curves — before the fixes, and after.

The earlier curve is the evidence for the second finding: over 500,000
timesteps the evaluation reward was statistically indistinguishable from a flat
line (r-squared 0.001, slope not significant at p = 0.87), and the best
checkpoint was the very first one taken. Meanwhile the documentation claimed a
21% improvement to -61 with a best of -45.81 — values that appear nowhere in the
committed file.

That figure belongs in the results document, so it is generated here rather than
described. The archived ``.npz`` is kept under ``docs/assets/legacy/`` precisely
so the claim stays checkable.

Usage::

    python scripts/plot_eval_curve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_NPZ = REPO_ROOT / "docs" / "assets" / "legacy" / "ppo_evaluations_baseline.npz"
# The current run writes here. Never copy it over the archived file: that
# archive is the evidence for that finding and must stay immutable.
CURRENT_NPZ = REPO_ROOT / "experiments" / "runs" / "ppo_routing" / "evaluations.npz"
RL_RESULTS = REPO_ROOT / "ml" / "results" / "rl_evaluation.json"
OUTPUT_DIR = REPO_ROOT / "docs" / "assets"


def analyse(path: Path) -> dict | None:
    """Regress an evaluation curve against timesteps."""
    if not path.exists():
        return None

    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]
    means = results.mean(axis=1)
    stds = results.std(axis=1)

    regression = stats.linregress(timesteps, means)
    return {
        "timesteps": timesteps,
        "means": means,
        "stds": stds,
        "slope_per_100k": float(regression.slope * 100_000),
        "r_squared": float(regression.rvalue**2),
        "p_value": float(regression.pvalue),
        "intercept": float(regression.intercept),
        "slope": float(regression.slope),
        "best": float(means.max()),
        "best_at": int(timesteps[int(means.argmax())]),
        "first": float(means[0]),
        "last": float(means[-1]),
    }


def plot_curve(analysis: dict, title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.errorbar(
        analysis["timesteps"],
        analysis["means"],
        yerr=analysis["stds"],
        fmt="o-",
        color="#3987e5",
        ecolor="#c3c2b7",
        elinewidth=1,
        capsize=3,
        markersize=5,
        label="Evaluation mean (± sd over 5 episodes)",
    )

    fitted = analysis["intercept"] + analysis["slope"] * analysis["timesteps"]
    ax.plot(
        analysis["timesteps"],
        fitted,
        "--",
        color="#d95926",
        linewidth=2,
        label="OLS fit",
    )

    ax.set_xlabel("Timesteps")
    # The reward function itself changed between these two runs, so the two
    # plots' vertical scales are NOT comparable. What is comparable is the
    # shape: a flat line against a rising one.
    ax.set_ylabel("Mean episode reward")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    verdict = (
        "significant at p < 0.05"
        if analysis["p_value"] < 0.05
        else "not significant at p < 0.05"
    )
    ax.annotate(
        f"slope = {analysis['slope_per_100k']:+.3f} / 100k steps\n"
        f"$r^2$ = {analysis['r_squared']:.4f}\n"
        f"p = {analysis['p_value']:.3f} ({verdict})\n"
        f"best checkpoint: {analysis['best']:.2f} at {analysis['best_at']:,} steps",
        xy=(0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f0efec", "edgecolor": "#c3c2b7"},
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def plot_current() -> Path | None:
    """Bar chart of the current policy against its floor and ceiling.

    A raw episode return conveys nothing on its own; what matters is where the
    policy sits between random and the oracle.
    """
    if not RL_RESULTS.exists():
        return None

    payload = json.loads(RL_RESULTS.read_text())
    policies = payload.get("policies", {})
    if not policies:
        return None

    order = ["random", "greedy_first_candidate", "ppo", "oracle"]
    names = [n for n in order if n in policies]
    means = [policies[n]["mean"] for n in names]
    errors = [
        [policies[n]["mean"] - policies[n]["ci95_low"] for n in names],
        [policies[n]["ci95_high"] - policies[n]["mean"] for n in names],
    ]
    labels = {
        "random": "Random\n(floor)",
        "greedy_first_candidate": "Greedy cheapest\n(≈ Dijkstra)",
        "ppo": "PPO",
        "oracle": "Greedy oracle\n(ceiling)",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#c3c2b7", "#9085e9", "#3987e5", "#199e70"]
    bars = ax.bar(
        [labels.get(n, n) for n in names],
        means,
        yerr=errors,
        capsize=4,
        color=colours[: len(names)],
        edgecolor="none",
    )

    for bar, name in zip(bars, names, strict=True):
        normalized = payload["normalized_scores"][name]
        ax.annotate(
            f"{bar.get_height():.1f}\nnorm {normalized:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#0b0b0b",
        )

    ax.set_ylabel("Mean episode return")
    ax.set_title(
        "PPO measured against a random floor and a greedy oracle ceiling",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "ppo_normalized_score.png"
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def _report(label: str, analysis: dict, path: Path) -> None:
    print(f"{label}: {path.relative_to(REPO_ROOT)}")
    print(f"  slope   {analysis['slope_per_100k']:+.4f} reward per 100k steps")
    print(f"  r^2     {analysis['r_squared']:.4f}")
    print(f"  p       {analysis['p_value']:.3f}")
    print(
        f"  first {analysis['first']:.2f} -> last {analysis['last']:.2f}; "
        f"best {analysis['best']:.2f} at {analysis['best_at']:,} steps"
    )


def main() -> int:
    legacy = analyse(LEGACY_NPZ)
    if legacy:
        path = plot_curve(
            legacy,
            "PPO evaluation curve BEFORE the fixes: the task was unobservable",
            "ppo_eval_curve_baseline.png",
        )
        _report("Baseline PPO curve", legacy, path)
    else:
        print(f"No archived evaluation file at {LEGACY_NPZ.relative_to(REPO_ROOT)}")

    current = analyse(CURRENT_NPZ)
    if current:
        path = plot_curve(
            current,
            "PPO evaluation curve AFTER the fixes: observable task, closed loop",
            "ppo_eval_curve_current.png",
        )
        _report("Current PPO curve", current, path)

    scores = plot_current()
    if scores:
        print(f"Normalized score chart: {scores.relative_to(REPO_ROOT)}")
    else:
        print("No ml/results/rl_evaluation.json yet; run `make train`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
