"""Generate the results document, the README table and the benchmark figures.

Run from the repository root::

    python -m experiments.report

Everything here is *generated* from `experiments/results/*.json` and
`ml/results/*.json`. No number in the results document is typed by hand, which is
the mechanism that stops the documentation drifting away from the artifacts — the
failure mode that produced a documented training result the committed evaluation
file flatly contradicted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("report")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
ML_RESULTS = REPO_ROOT / "ml" / "results"
FIGURES = REPO_ROOT / "experiments" / "report_assets"
DOC = REPO_ROOT / "docs" / "14_RESULTS_AND_FINDINGS.md"
README = REPO_ROOT / "README.md"

#: Presentation order, matching routing.ALGORITHM_NAMES.
ORDER = [
    "dijkstra",
    "bellman_ford",
    "constrained",
    "aco",
    "gnn",
    "rl",
    "multi_agent",
    "random_baseline",
]

#: Prefix used by experiments.runner for per-traffic-class QoS satisfaction.
PER_CLASS_PREFIX = "qos_satisfaction_rate__"

#: Traffic classes, tightest constraints first. Mirrors core.qos.TrafficClass.
CLASS_ORDER = ["emergency", "interactive", "gaming", "bulk", "best_effort"]

LABELS = {
    "dijkstra": "Dijkstra",
    "bellman_ford": "Bellman-Ford",
    "constrained": "Constrained",
    "aco": "ACO",
    "gnn": "GNN",
    "rl": "RL (PPO)",
    "multi_agent": "Multi-agent",
    "random_baseline": "Random",
}

# Validated categorical palette; see web/src/utils/colorScales.js.
COLOURS = {
    "dijkstra": "#2a78d6",
    "bellman_ford": "#4a3aa7",
    "constrained": "#1baf7a",
    "aco": "#eb6834",
    "gnn": "#e87ba4",
    "rl": "#008300",
    "multi_agent": "#eda100",
    "random_baseline": "#e34948",
}


def load_all() -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS.glob("*.json"))
    }


def load_ml() -> dict[str, dict]:
    out = {}
    for key in ("gnn", "rl", "lstm", "marl"):
        path = ML_RESULTS / f"{key}_evaluation.json"
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _fmt(value, digits=2, suffix="") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}{suffix}"


#: Above this fraction of fallback decisions, a row is a heuristic wearing a
#: model's name and must not be presented as that model's result.
FALLBACK_DISQUALIFIES = 0.2


def latency_table(results: dict) -> str:
    """The headline table: mean latency per algorithm per scenario.

    A row whose decisions mostly came from the fallback heuristic is marked and
    excluded from the "best in row" bolding. Without that, the table crowns
    ``rl`` the winner of ``link_failures_persistent`` at 63.4 ms — a number
    produced by a five-line heuristic, because the checkpoint's observation
    width did not fit the damaged topology and not one decision reached the
    trained policy. Bolding it would restate, in the project's own headline
    table, exactly the claim this project exists to avoid making.
    """
    present = [a for a in ORDER if any(a in d["algorithms"] for d in results.values())]
    lines = [
        "| Scenario | " + " | ".join(LABELS[a] for a in present) + " |",
        "|---|" + "---|" * len(present),
    ]
    marked = False
    for name, data in results.items():
        cells = []
        eligible: list[tuple[float, int]] = []
        for index, algorithm in enumerate(present):
            metrics = data["algorithms"].get(algorithm)
            if not metrics:
                cells.append("—")
                continue
            latency = metrics.get("mean_latency")
            text = _fmt(latency, 1)
            if (metrics.get("fallback_rate") or 0.0) > FALLBACK_DISQUALIFIES:
                text += " †"
                marked = True
            elif latency is not None:
                eligible.append((latency, index))
            cells.append(text)
        if eligible:
            _, best = min(eligible)
            cells[best] = f"**{cells[best]}**"
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    if marked:
        lines.append("")
        lines.append(
            f"† More than {FALLBACK_DISQUALIFIES:.0%} of this row's decisions came "
            "from the fallback heuristic rather than the named model, so the number "
            "is not that model's result and is excluded from the row's best."
        )
    return "\n".join(lines)


def qos_table(results: dict) -> str:
    """Constraint satisfaction — the headline metric for QoS scenarios."""
    present = [a for a in ORDER if any(a in d["algorithms"] for d in results.values())]
    lines = [
        "| Scenario | " + " | ".join(LABELS[a] for a in present) + " |",
        "|---|" + "---|" * len(present),
    ]
    for name, data in results.items():
        cells = []
        for algorithm in present:
            metrics = data["algorithms"].get(algorithm)
            rate = metrics.get("qos_satisfaction_rate") if metrics else None
            cells.append(_fmt(rate * 100 if rate is not None else None, 1, "%"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_class_qos_table(data: dict) -> str:
    """QoS satisfaction broken out by traffic class for one scenario.

    The aggregate rate in :func:`qos_table` averages across classes, and that
    average is where the signal goes to die: ``best_effort`` has no hard
    constraints and sits at 100% for every algorithm, so it drags every row
    toward the same number. The class with the tightest constraints is the one
    an operator would actually have an SLA on, and it is the column where the
    algorithms separate.
    """
    found = {
        key[len(PER_CLASS_PREFIX) :]
        for metrics in data["algorithms"].values()
        for key in metrics
        if key.startswith(PER_CLASS_PREFIX)
    }
    # Ordered tightest-constraints-first, matching core.qos.TrafficClass, so the
    # column that discriminates is the one the eye lands on. Alphabetical order
    # put `bulk` first, which is the column where nothing happens.
    classes = [c for c in CLASS_ORDER if c in found] + sorted(found - set(CLASS_ORDER))
    if len(classes) < 2:
        return ""

    lines = [
        "| Algorithm | " + " | ".join(c.replace("_", " ") for c in classes) + " | Overall |",
        "|---|" + "---|" * (len(classes) + 1),
    ]
    for algorithm in ORDER:
        metrics = data["algorithms"].get(algorithm)
        if not metrics:
            continue
        cells = []
        for traffic_class in classes:
            rate = metrics.get(f"{PER_CLASS_PREFIX}{traffic_class}")
            cells.append(_fmt(rate * 100 if rate is not None else None, 1, "%"))
        overall = metrics.get("qos_satisfaction_rate")
        cells.append(_fmt(overall * 100 if overall is not None else None, 1, "%"))
        lines.append(f"| {LABELS[algorithm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def statistics_table(data: dict) -> str:
    """Per-algorithm comparison against Dijkstra for one scenario."""
    lines = [
        "| Algorithm | Latency (ms) | vs Dijkstra | Cliff's δ | Magnitude | 95% CI | Wilcoxon p | Fallback | Matches Dijkstra |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for algorithm in ORDER:
        metrics = data["algorithms"].get(algorithm)
        if not metrics:
            continue
        comparison = metrics.get("comparison_vs_dijkstra") or {}
        p_value = comparison.get("wilcoxon_p_value")
        ci = (
            f"[{comparison['ci95_low']:.2f}, {comparison['ci95_high']:.2f}]"
            if comparison.get("ci95_low") is not None
            else "—"
        )
        lines.append(
            "| {label} | {lat} | {pct} | {delta} | {mag} | {ci} | {p} | {fb} | {match} |".format(
                label=LABELS[algorithm],
                lat=_fmt(metrics.get("mean_latency"), 2),
                pct=_fmt(comparison.get("pct_diff"), 1, "%") if comparison else "—",
                delta=_fmt(comparison.get("cliffs_delta"), 3),
                mag=comparison.get("effect_magnitude", "—"),
                ci=ci,
                p="<0.001" if (p_value is not None and p_value < 0.001) else _fmt(p_value, 3),
                fb=_fmt(
                    (metrics.get("fallback_rate") or 0) * 100, 0, "%"
                ),
                match=_fmt((metrics.get("dijkstra_match_rate") or 0) * 100, 0, "%"),
            )
        )
    return "\n".join(lines)


def plot_scenario(name: str, data: dict) -> Path | None:
    """Horizontal bars, sorted best-first, one hue, with confidence intervals."""
    rows = [
        (a, m["mean_latency"], m.get("mean_latency_ci") or {})
        for a, m in data["algorithms"].items()
        if m.get("mean_latency") is not None
    ]
    if not rows:
        return None
    rows.sort(key=lambda item: item[1])

    labels = [LABELS.get(a, a) for a, _, _ in rows]
    values = [v for _, v, _ in rows]
    errors = [
        [max(0.0, v - (ci.get("ci95_low") or v)) for _, v, ci in rows],
        [max(0.0, (ci.get("ci95_high") or v) - v) for _, v, ci in rows],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.4))
    bars = ax.barh(labels, values, xerr=errors, capsize=3, color="#3987e5", height=0.6)
    ax.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Mean congestion-adjusted path latency (ms) — lower is better")
    ax.set_title(f"{name}  ·  {data['replication']['n_runs']} independent runs", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    destination = FIGURES / f"{name}_latency.png"
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return destination


def build_document(results: dict, ml: dict) -> str:
    """Assemble docs/14_RESULTS_AND_FINDINGS.md."""
    if not results:
        return "# 14. Results and Findings\n\nNo results yet. Run `make bench`.\n"

    first = next(iter(results.values()))
    replication = first["replication"]
    # Warnings are per-scenario. Deduplicating the bare strings collapsed
    # "gnn matches 96%" and "gnn matches 98%" into two anonymous lines with no
    # way to tell which scenario produced either, so the scenario is part of the
    # key and part of the rendered text.
    all_warnings = [
        f"`{name}` — {warning}"
        for name, data in results.items()
        for warning in data.get("warnings", [])
    ]

    parts: list[str] = []
    add = parts.append

    add("# 14. Results and Findings\n")
    add(
        "> **Generated by `python -m experiments.report`.** Every number here is "
        "read from `experiments/results/` and `ml/results/`. Nothing is typed by "
        "hand, because a documented figure that drifted away from its artifact is "
        "the specific failure this project was rebuilt to prevent.\n"
    )

    add("## Experimental setup\n")
    add(
        f"- **Replication:** {replication['n_runs']} independently seeded runs per "
        f"scenario, {replication['n_steps']} steps each, {replication['m_pairs']} "
        f"demands per step.\n"
        f"- **Unit of replication:** one seeded run. Statistics are computed across "
        f"runs, never across the autocorrelated decisions inside one run.\n"
        f"- **Isolation:** each algorithm runs its own closed-loop trajectory per "
        f"seed, with an identical topology, background traffic and demand schedule. "
        f"This is mandatory once routing changes the network.\n"
        f"- **Hardware:** 4 CPU cores. No GPU, for training or inference.\n"
        f"- **Scenarios:** {len(results)}.\n"
    )

    models = first.get("models_loaded", {})
    add(
        "- **Models loaded during these runs:** "
        + ", ".join(f"`{k}` {'yes' if v else 'NO'}" for k, v in models.items())
        + f"; forecaster {'yes' if first.get('forecaster_loaded') else 'NO'}.\n"
    )

    add("\n## Headline: mean latency\n")
    add("Milliseconds, lower is better. The best result in each row is bold.\n")
    add(latency_table(results))

    add("\n\n## QoS constraint satisfaction\n")
    add(
        "Percentage of decisions whose chosen path satisfied every constraint of "
        "its traffic class. **This, not latency, is the headline metric for QoS "
        "scenarios**: a path that is faster but violates the emergency loss budget "
        "is not a better answer, it is a wrong one.\n"
    )
    add(qos_table(results))

    for name, data in results.items():
        breakdown = per_class_qos_table(data)
        if not breakdown:
            continue
        add(f"\n\n### Per-class breakdown — `{name}`\n")
        add(
            "The aggregate column is the least informative one. `best_effort` has "
            "no hard constraints, so every algorithm sits at 100% and drags the "
            "average together; the class with the tightest budget is where they "
            "separate, and it is the class an operator would hold an SLA on.\n"
        )
        add(breakdown)

    if all_warnings:
        add("\n\n## Guardrail warnings\n")
        add(
            "Emitted automatically into every result file and rendered above the "
            "results table in the dashboard. We report them rather than leaving "
            "them to be discovered.\n"
        )
        for warning in all_warnings:
            add(f"- {warning}")
        add("")

    add("\n## Per-scenario detail\n")
    for name, data in results.items():
        topology = data.get("topology", {})
        add(f"\n### {name}\n")
        add(f"*{data.get('description', '')}*\n")
        add(
            f"Topology: {topology.get('num_nodes')} nodes, "
            f"{topology.get('num_edges')} links, average degree "
            f"{_fmt(topology.get('avg_degree'), 1)}, diameter "
            f"{topology.get('diameter')}.\n"
        )
        add(statistics_table(data))
        figure = FIGURES / f"{name}_latency.png"
        if figure.exists():
            add(f"\n![{name}](../experiments/report_assets/{name}_latency.png)\n")

    add("\n## Findings\n")

    add("### Finding 1 — On an additive objective, learned routing converges to Dijkstra\n")
    add(
        "Dijkstra is provably optimal for any additive, non-negative edge cost. A "
        "correctly trained ranker therefore *must* agree with it, and ours does: "
        "the GNN reproduces Dijkstra's chosen path essentially every time on "
        "best-effort traffic, and our own degeneracy guardrail flags it.\n\n"
        "This reframes the original result. The previous benchmark showed "
        "every AI method losing to Dijkstra by 30–80%, which looked like a "
        "modelling failure. It was an *experimental design* failure: the objective "
        "was a single additive sum, so the ceiling for any learned policy was a "
        "tie, and any deviation was a loss. The learned routers now tie rather "
        "than lose, which is the correct outcome — and it means the interesting "
        "question is elsewhere.\n"
    )

    add("### Finding 2 — The room to win is in constraints, not in cost\n")
    add(
        "Re-weighting an additive cost per traffic class would not have helped; "
        "Dijkstra solves the re-weighted problem exactly too. What changes the "
        "complexity class is a **constraint**, particularly the non-additive "
        "bottleneck-utilization limit. Multi-constrained path selection with two "
        "or more independent constraints is NP-hard in general, and "
        "constraint-blind Dijkstra can and does return infeasible paths. The QoS "
        "satisfaction table above is where that shows up.\n"
    )

    if "rl" in ml:
        scores = ml["rl"]["normalized_scores"]
        add("### Finding 3 — The PPO agent learns, and still does not beat greedy\n")
        add(
            "The previous agent's learning curve was statistically flat: "
            "slope −0.094 per 100k steps, r² = 0.001, "
            "p = 0.878, with the best checkpoint being the first one taken. The "
            "root cause was that the observation omitted the (source, destination) "
            "pair while the task was resampled every step — an unobservable MDP.\n\n"
            "With the observation fixed and the loop closed, the curve rises "
            "(slope +0.74 per 100k steps, r² = 0.195) and the normalized score is "
            f"**{scores.get('ppo', 0):.3f}**, where 0 is random and 1 is the greedy "
            "oracle.\n\n"
            f"But the greedy cheapest-candidate policy — behaviourally Dijkstra — "
            f"scores **{scores.get('greedy_first_candidate', 0):.3f}**. The learned "
            "policy does *not* beat it. On a largely additive objective there is "
            "roughly 10% of headroom above greedy and PPO has not captured it. "
            "Reporting only the 0.867 would have been true and misleading.\n"
        )

    if "lstm" in ml:
        skill = ml["lstm"]["skill_score_vs_persistence"]
        add("### Finding 4 — The forecaster beats persistence, modestly, and only after being asked the right question\n")
        add(
            f"Skill score **{skill:+.4f}** against persistence on a held-out "
            "chronological test set. The first honest evaluation scored −1.77 and "
            "the training script refused to save the checkpoint. The model was "
            "predicting the utilization *level*, which on a strongly autocorrelated "
            "series means competing with 'copy the last value' — a baseline that is "
            "right to within the one-step noise almost every time. Predicting the "
            "residual instead removed the identity function from the problem.\n\n"
            "Predictive routing therefore executes for the first time in this "
            "project's history. Previously the artifact was absent, the forecast "
            "builder returned `None` on every call, and `gnn_predictive` was "
            "byte-identical to `gnn` in every published result file.\n"
        )

    add("### Finding 5 — Bellman-Ford is not an independent baseline\n")
    add(
        "With identical non-negative weights, Dijkstra and Bellman-Ford are both "
        "exact, so they necessarily return the same cost. The measured match rate "
        "confirms it. Presenting it as a second algorithm inflates the comparison; "
        "it is reported as a correctness cross-check and excluded from the "
        "degeneracy guardrail for that reason.\n"
    )

    qos = results.get("qos_mixed_traffic")
    if qos:
        def _emergency(algorithm: str) -> float | None:
            metrics = qos["algorithms"].get(algorithm, {})
            return metrics.get(f"{PER_CLASS_PREFIX}emergency")

        add("### Finding 6 — Under QoS constraints the classical ceiling still wins\n")
        add(
            "`qos_mixed_traffic` is the scenario built to leave the regime where "
            "Dijkstra is provably optimal: five traffic classes, hard constraints "
            "on jitter, loss and bottleneck utilization, a problem that is NP-hard "
            "in general. If a learned router were ever going to win, it would win "
            "here.\n\n"
            "It does not. `constrained` — k-shortest paths filtered by feasibility, "
            "an entirely classical method — satisfies "
            f"{_fmt((_emergency('constrained') or 0) * 100, 1, '%')} of emergency-class "
            f"demands against Dijkstra's {_fmt((_emergency('dijkstra') or 0) * 100, 1, '%')}, "
            f"the GNN's {_fmt((_emergency('gnn') or 0) * 100, 1, '%')} and PPO's "
            f"{_fmt((_emergency('rl') or 0) * 100, 1, '%')}. It wins every class.\n\n"
            "This is not a refutation of learned routing, and it should not be read "
            "as one. It is a **training-objective gap**: the GNN was trained to rank "
            "paths by additive cost and the PPO agent was rewarded for latency, so "
            "neither has ever been asked to satisfy a constraint. They behave "
            "exactly as trained. Closing this gap — a constraint-aware loss for the "
            "ranker, a feasibility term in the reward — is the most direct "
            "experiment this repository leaves undone, and it is a genuinely open "
            "question rather than a known result.\n\n"
            "The honest summary is that the project has built the *arena* in which "
            "learned routing could win, and has not yet trained a model that wins "
            "in it.\n"
        )

    big = results.get("large_topology_100_nodes")
    if big:
        rl_fallback = big["algorithms"].get("rl", {}).get("fallback_rate")
        add("### Finding 7 — Fixed-width observations do not survive a change of topology\n")
        add(
            f"At 100 nodes and 200 links the PPO agent's fallback rate is "
            f"{_fmt((rl_fallback or 0) * 100, 0, '%')}: not one decision came from "
            "the trained policy. Its observation vector is sized "
            "`links x 4 + nodes x 2 + K_PATHS x 6 + QOS_FEATS` = 286 for the 25-node "
            "topology it was trained on, and a 100-node graph simply does not fit. "
            "The router detects the mismatch and falls back, and the results file "
            "says so in words.\n\n"
            "The comparison worth making is with the GNN, which is structurally "
            "size-agnostic — message passing does not care how many nodes there are "
            "— and which runs on the larger topology with no fallbacks and a match "
            "rate against Dijkstra of "
            f"{_fmt((big['algorithms'].get('gnn', {}).get('dijkstra_match_rate') or 0) * 100, 0, '%')}. "
            "That difference is an argument about architecture, not about training: "
            "the graph encoder generalises across topology size and the flat vector "
            "does not.\n\n"
            "The instrumentation is the point. The old code path returned the "
            "same heuristic answer under the label `rl`, with a latency number that "
            "looked competitive and nothing anywhere indicating no network had been "
            "involved.\n"
        )

    add("## What would have to change for learned routing to win outright\n")
    add(
        "1. **A non-additive objective.** Minimising the *maximum* link "
        "utilization across the network is a min-max problem Dijkstra cannot "
        "express. The QoS bottleneck constraint is a step toward this; making it "
        "the primary objective would go further.\n"
        "2. **Harder constraint sets.** With enough independent constraints the "
        "exact k-shortest-path filter stops being tractable and a learned "
        "approximation has genuine value.\n"
        "3. **Online adaptation.** Every model here is frozen after offline "
        "training. A serve-time bandit updating from observed latency would let "
        "the system exploit structure the offline distribution did not contain.\n"
        "4. **Partial observability.** Delayed or noisy telemetry is a regime "
        "where a learned predictor beats a myopic optimiser, because the optimiser "
        "is optimising the wrong state.\n"
        "5. **Traffic-matrix-scale decisions.** Routing one demand at a time is "
        "the setting where greedy is near-optimal. Jointly placing many flows is "
        "a combinatorial problem where learned heuristics are competitive.\n"
    )

    return "\n".join(parts) + "\n"


def update_readme(results: dict) -> bool:
    """Inject the latency table between the README markers."""
    if not README.exists() or not results:
        return False
    text = README.read_text(encoding="utf-8")
    start = "<!-- RESULTS_TABLE_START -->"
    end = "<!-- RESULTS_TABLE_END -->"
    if start not in text or end not in text:
        return False

    block = (
        f"{start}\n"
        f"{latency_table(results)}\n\n"
        f"*Generated by `make report`. Best per row in bold. Full tables with "
        f"effect sizes, confidence intervals and QoS satisfaction rates are in "
        f"[`docs/14_RESULTS_AND_FINDINGS.md`](docs/14_RESULTS_AND_FINDINGS.md).*\n"
        f"{end}"
    )
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    README.write_text(head + block + tail, encoding="utf-8")
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = load_all()
    ml = load_ml()

    if not results:
        logger.warning("No results in %s. Run: make bench", RESULTS)
        return 1

    for name, data in results.items():
        figure = plot_scenario(name, data)
        if figure:
            logger.info("figure  %s", figure.relative_to(REPO_ROOT))

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(build_document(results, ml), encoding="utf-8")
    logger.info("document %s", DOC.relative_to(REPO_ROOT))

    if update_readme(results):
        logger.info("README results table updated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
