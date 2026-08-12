"""Statistics for benchmark comparison. The unit of replication is one seeded run.

The previous harness ran ``scipy.stats.wilcoxon`` over 1,000 steps x 20 pairs =
20,000 paired observations *from a single trajectory*. Wilcoxon assumes
independent pairs. Successive steps of the simulator are heavily autocorrelated
— step *t* and step *t+1* are almost the same network — so those 20,000 numbers
are roughly one independent observation repeated, not 20,000 samples.

With that much pseudo-replication, **any** difference becomes "significant".
That is why every committed result file reported ``wilcoxon_p_value: 0.0``: a
literal zero is numerical underflow, not a p-value.

The fix is not a different test, it is a different unit. Run N independent
seeded replications, reduce each one to a single mean per algorithm, and test
across those N numbers. Also reported here:

* **Cliff's delta**, a genuine non-parametric effect size in [-1, 1]. The old
  ``effect_size_pct`` was a raw percent difference in means, which is not an
  effect size at all and is not standardised by anything.
* **A bootstrap 95% confidence interval** on the mean difference, which conveys
  the practical size of an effect in a way a p-value never can.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

#: Below this many independent runs, a Wilcoxon test cannot reach significance
#: at alpha = 0.05 even in the best case, so reporting one would be misleading.
MIN_RUNS_FOR_TEST = 6


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]. Negative means *a* is smaller."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    greater = sum(int((x > b).sum()) for x in a)
    less = sum(int((x < b).sum()) for x in a)
    return (greater - less) / (a.size * b.size)


def interpret_delta(delta: float) -> str:
    """Standard thresholds for Cliff's delta (Romano et al.)."""
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def paired_comparison(algo_run_means, baseline_run_means, n_bootstrap: int = 5000) -> dict:
    """Compare an algorithm against the baseline across independent runs.

    Both inputs are **one value per seed**, not one value per routing decision.
    That distinction is the entire point of this module.
    """
    a = np.asarray(algo_run_means, dtype=float)
    b = np.asarray(baseline_run_means, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(a.size)

    out: dict[str, object] = {"n_runs": n}
    if n == 0:
        out["note"] = "no usable runs"
        return out

    diffs = a - b
    out["mean_diff"] = float(diffs.mean())
    out["pct_diff"] = float(diffs.mean() / b.mean() * 100) if b.mean() else None

    if n >= MIN_RUNS_FOR_TEST and not np.allclose(diffs, 0):
        try:
            statistic, p_value = stats.wilcoxon(a, b)
            out["wilcoxon_p_value"] = float(p_value)
            out["wilcoxon_statistic"] = float(statistic)
        except ValueError as exc:
            out["wilcoxon_p_value"] = None
            out["wilcoxon_note"] = str(exc)
    else:
        out["wilcoxon_p_value"] = None
        out["wilcoxon_note"] = (
            "identical to the baseline in every run"
            if np.allclose(diffs, 0)
            else f"insufficient independent runs (need >= {MIN_RUNS_FOR_TEST})"
        )

    delta = cliffs_delta(a, b)
    out["cliffs_delta"] = float(delta)
    out["effect_magnitude"] = interpret_delta(delta)

    if n > 1:
        rng = np.random.default_rng(0)
        boot = [
            float(rng.choice(diffs, size=n, replace=True).mean())
            for _ in range(n_bootstrap)
        ]
        out["ci95_low"] = float(np.percentile(boot, 2.5))
        out["ci95_high"] = float(np.percentile(boot, 97.5))
    else:
        out["ci95_low"] = out["ci95_high"] = None

    return out


def summarise(values) -> dict:
    """Mean, sd and a 95% CI for a list of per-run values."""
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None, "n": 0}
    n = int(array.size)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n": n,
    }


def path_diversity(rows: list[dict]) -> float:
    """Normalised entropy of the paths chosen for each (src, dst) pair.

    The old metric read ``r.get("path", [])`` from a dict that only ever stored
    ``path_len``, so every entry was the string ``"[]"`` and ``diversity_index``
    was 0.000 in all five committed result files. The metric that would have
    revealed whether the AI routers explore alternatives had never once worked.

    Entropy is also more informative than unique/total: it distinguishes "uses a
    second path 1% of the time" from "splits evenly between two paths".
    """
    import math
    from collections import Counter, defaultdict

    by_pair: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for row in rows:
        if row.get("success") and row.get("path"):
            by_pair[(row["src"], row["dst"])].append(tuple(row["path"]))

    scores: list[float] = []
    for paths in by_pair.values():
        if len(paths) < 2:
            continue
        counts = Counter(paths)
        total = len(paths)
        entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
        scores.append(entropy / math.log(len(paths)))

    return float(np.mean(scores)) if scores else 0.0


__all__ = [
    "MIN_RUNS_FOR_TEST",
    "cliffs_delta",
    "interpret_delta",
    "paired_comparison",
    "path_diversity",
    "summarise",
]
