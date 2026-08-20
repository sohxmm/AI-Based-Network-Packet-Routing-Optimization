import { useMemo } from "react";

import AlgorithmMetricsTable from "./benchmark/AlgorithmMetricsTable.jsx";
import LatencyChart from "./benchmark/LatencyChart.jsx";
import StatisticalSummary from "./benchmark/StatisticalSummary.jsx";
import WarningsCallout from "./benchmark/WarningsCallout.jsx";
import { scenarioLabel } from "../utils/colorScales.js";

/**
 * Composition only: fetching lives in BenchmarkReport, and each piece of the
 * presentation lives in ./benchmark/. This file was 543 lines — about three
 * times the next-largest component — and held the selector, the table, the
 * chart and the statistics in one place.
 */
export function BenchmarkResultView({
  scenarioData,
  scenarioLabel: labelOverride,
  showLimitations = true,
  limitations,
  isDark = true,
}) {
  const rows = useMemo(() => flattenAlgorithms(scenarioData), [scenarioData]);

  if (!scenarioData) {
    return (
      <p className="text-sm text-app-muted">
        No benchmark results yet. Generate them with{" "}
        <code className="font-mono">make bench</code>.
      </p>
    );
  }

  const topology = scenarioData.topology;

  return (
    <div className="flex flex-col gap-4">
      <header>
        <h2 className="text-sm font-semibold text-app-text">
          {labelOverride ?? scenarioLabel(scenarioData.scenario ?? "")}
        </h2>
        {scenarioData.description && (
          <p className="mt-0.5 text-xs text-app-muted">{scenarioData.description}</p>
        )}
        {topology && (
          <p className="mt-1 font-mono text-[11px] text-app-muted">
            {topology.num_nodes} nodes · {topology.num_edges} links · avg degree{" "}
            {topology.avg_degree?.toFixed(1)} · diameter {topology.diameter ?? "—"}
          </p>
        )}
      </header>

      <WarningsCallout warnings={scenarioData.warnings} />

      <LatencyChart rows={rows} isDark={isDark} />

      <AlgorithmMetricsTable rows={rows} isDark={isDark} />

      <StatisticalSummary rows={rows} replication={scenarioData.replication} />

      {showLimitations && limitations?.benchmark_readme && (
        <details className="rounded border border-app-border bg-app-input-bg p-3">
          <summary className="cursor-pointer text-xs font-semibold text-app-text">
            Known limitations of these results
          </summary>
          <pre className="mt-2 whitespace-pre-wrap text-[11px] leading-relaxed text-app-muted">
            {limitations.benchmark_readme}
          </pre>
        </details>
      )}
    </div>
  );
}

/** Turn the nested result payload into flat rows the presentation can consume. */
function flattenAlgorithms(scenarioData) {
  const algorithms = scenarioData?.algorithms ?? {};
  return Object.entries(algorithms).map(([algorithm, metrics]) => ({
    algorithm,
    mean_latency: metrics.mean_latency,
    ci: metrics.mean_latency_ci,
    success_rate: metrics.success_rate,
    fallback_rate: metrics.fallback_rate,
    qos_satisfaction_rate: metrics.qos_satisfaction_rate,
    diversity_index: metrics.diversity_index,
    p95_path_max_utilization: metrics.p95_path_max_utilization,
    dijkstra_match_rate: metrics.dijkstra_match_rate,
    comparison: metrics.comparison_vs_dijkstra,
    pct_diff: metrics.comparison_vs_dijkstra?.pct_diff,
  }));
}

export default BenchmarkResultView;
