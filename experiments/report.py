import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("report")

def generate_report():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    assets_dir = base_dir / "report_assets"
    assets_dir.mkdir(exist_ok=True)
    
    if not results_dir.exists():
        logger.error("No results directory found.")
        return
        
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        logger.error("No JSON results found.")
        return
        
    report_md = ["# Routing Algorithms Benchmark Report\n"]
    
    # Process each scenario
    for rf in result_files:
        with open(rf, "r") as f:
            data = json.load(f)
            
        scenario = data["scenario"]
        report_md.append(f"## Scenario: {scenario}")
        report_md.append(f"Steps: {data['n_steps']} | Pairs per step: {data['m_pairs']}\n")
        
        if scenario == "large_topology_100_nodes":
            report_md.append("> [!IMPORTANT]")
            report_md.append("> **Note on Large Topology:** AI algorithms fell back to heuristic routing here — models were trained on the 25-node network and were not evaluated as trained policies on this topology.\n")
        
        algos = data["algorithms"]
        
        # Guardrails check
        warnings = []
        for algo, metrics in algos.items():
            # 1. Fallback tracking
            if algo not in ["dijkstra", "bellman_ford", "aco"]:
                if metrics["fallback_rate"] > 0.05:
                    warnings.append(f"> [!WARNING]\n> **{algo}**: High fallback rate ({metrics['fallback_rate']*100:.1f}%). Model failed to load or triggered heuristic.")
                    
            # 2. Degeneracy check
            if algo not in ("dijkstra", "bellman_ford"):
                if metrics["dijkstra_match_rate"] > 0.90:
                    warnings.append(f"> [!WARNING]\n> **{algo}**: Possible degenerate/non-differentiated policy. Matches Dijkstra on {metrics['dijkstra_match_rate']*100:.1f}% of pairs.")
                    
            # 3. Variance sanity check
            # We approximated std dev from variance
            std_dev = math.sqrt(metrics["util_variance"]) if metrics["util_variance"] >= 0 else 0
            if std_dev < 1e-4 and algo != "dijkstra":
                 warnings.append(f"> [!WARNING]\n> **{algo}**: Policy may be collapsing to a constant action (utilization std dev = {std_dev:.6f}).")

        if warnings:
            report_md.append("### Guardrail Warnings")
            report_md.extend(warnings)
            report_md.append("\n")
            
        # Table
        report_md.append("### Metrics Table\n")
        report_md.append("| Algorithm | Mean Latency | p95 Latency | Util Variance | Success Rate | Wilcoxon p-value |")
        report_md.append("|---|---|---|---|---|---|")
        
        latencies = {}
        
        for algo, metrics in algos.items():
            latencies[algo] = metrics["mean_latency"]
            
            p_val = metrics.get("wilcoxon_p_value", float("nan"))
            if p_val == -1.0:
                err_msg = metrics.get("wilcoxon_error", "Unknown error")
                p_str = f"ERROR ({err_msg})"
            elif math.isnan(p_val):
                p_str = "NaN — algorithms produced statistically identical results"
            else:
                p_str = f"{p_val:.4e}" if p_val != 0 else "0.0"
                if algo == "dijkstra":
                    p_str = "N/A"
                    
            report_md.append(
                f"| {algo} | {metrics['mean_latency']:.2f}ms | {metrics['p95_latency']:.2f}ms | "
                f"{metrics['util_variance']:.5f} | {metrics['success_rate']*100:.1f}% | {p_str} |"
            )
        report_md.append("\n")
        
        # Chart
        plt.figure(figsize=(10, 6))
        names = list(latencies.keys())
        vals = list(latencies.values())
        plt.bar(names, vals, color='skyblue')
        plt.title(f"Mean Latency by Algorithm - {scenario}")
        plt.ylabel("Latency (ms)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        chart_path = assets_dir / f"{scenario}_latency.png"
        plt.savefig(chart_path)
        plt.close()
        
        report_md.append(f"![Latency Chart]({chart_path.absolute().as_posix()})\n")
        
    out_file = base_dir / "report.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))
        
    logger.info(f"Report generated at {out_file}")

if __name__ == "__main__":
    generate_report()
