import React from "react";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";

const ALGO_LABELS = {
  dijkstra: "Dijkstra",
  bellman_ford: "B-Ford",
  aco: "ACO",
  gnn: "GNN",
  gnn_predictive: "GNN-P",
  rl: "RL",
  rl_predictive: "RL-P",
  multi_agent: "MARL",
};

/**
 * Shared result renderer — used by both the fixed benchmark report and experiment sandbox.
 */
export function BenchmarkResultView({
  scenarioData,
  scenarioLabel,
  showLimitations = false,
  limitations = null,
}) {
  if (!scenarioData || !scenarioData.algorithms) {
    return (
      <div className="rounded border border-app-border bg-app-panel p-6 text-center text-app-muted">
        No benchmark data available for this scenario.
      </div>
    );
  }

  const algos = scenarioData.algorithms;
  const algoNames = Object.keys(algos);
  const isLargeTopology = scenarioData.scenario?.includes("large_topology");

  // ⚠️ Guardrail warnings ⚠️
  const warnings = [];
  for (const [algo, m] of Object.entries(algos)) {
    if (!["dijkstra", "bellman_ford", "aco"].includes(algo) && m.fallback_rate > 0.05) {
      warnings.push({
        type: "error",
        message: `${ALGO_LABELS[algo] || algo}: High fallback (${(m.fallback_rate * 100).toFixed(1)}%). Policy failed to differentiate.`,
      });
    }
    if (!["dijkstra", "bellman_ford"].includes(algo) && m.dijkstra_match_rate > 0.9) {
      warnings.push({
        type: "warning",
        message: `${ALGO_LABELS[algo] || algo}: Degenerate policy — matches Dijkstra ${(m.dijkstra_match_rate * 100).toFixed(1)}% of time.`,
      });
    }
  }

  // 🏆 Find Best Performers 🏆
  let bestLatency = null;
  let bestVariance = null;
  
  for (const [algo, m] of Object.entries(algos)) {
    if (m.fallback_rate > 0.05) continue;
    
    if (m.mean_latency != null && (!bestLatency || m.mean_latency < bestLatency.val)) {
      bestLatency = { algo, val: m.mean_latency };
    }
    if (m.utilization_variance != null && (!bestVariance || m.utilization_variance < bestVariance.val)) {
      bestVariance = { algo, val: m.utilization_variance };
    }
  }

  const chartData = algoNames.map((algo) => ({
    algorithm: ALGO_LABELS[algo] || algo,
    "Mean Latency (ms)": algos[algo].mean_latency,
    "Util Variance (x1000)": algos[algo].utilization_variance * 1000,
  }));

  return (
    <div className="flex flex-col gap-5 animate-fade-in">
      <div className="flex flex-wrap items-center gap-3 border-b border-app-border pb-3">
        <h3 className="text-sm font-semibold text-app-text">
          Scenario Results: <span className="text-app-accent">{scenarioLabel}</span>
        </h3>
        
        {bestLatency && (
          <span className="ml-auto rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-400 border border-emerald-500/20 shadow-sm">
            Fastest: {ALGO_LABELS[bestLatency.algo] || bestLatency.algo}
          </span>
        )}
        {bestVariance && (
          <span className="rounded-full bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-500 border border-amber-500/20 shadow-sm">
            Best Load Balance: {ALGO_LABELS[bestVariance.algo] || bestVariance.algo}
          </span>
        )}
      </div>

      {warnings.length > 0 && (
        <div className="flex flex-col gap-2">
          {warnings.map((w, i) => (
            <div
              key={i}
              className={`flex items-start gap-2 rounded border px-3 py-2 text-sm ${
                w.type === "error"
                  ? "border-red-500/40 bg-red-500/10 text-red-400"
                  : "border-amber-500/40 bg-amber-500/10 text-amber-400"
              }`}
            >
              <span className="mt-0.5 text-base">
                {w.type === "error" ? "🚨" : "⚠️"}
              </span>
              <span>{w.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* Metrics Table */}
      <div className="overflow-x-auto rounded border border-app-border bg-app-panel shadow-sm">
        <table className="w-full text-left text-sm">
          <thead className="bg-app-input-bg text-xs uppercase text-app-muted border-b border-app-border">
            <tr>
              <th className="px-3 py-2">Algorithm</th>
              <th className="px-3 py-2">Mean Latency</th>
              <th className="px-3 py-2">Success Rate</th>
              <th className="px-3 py-2">Fallback Rate</th>
              <th className="px-3 py-2">Dijkstra Match</th>
              <th className="px-3 py-2">p-value (vs Dijk)</th>
              <th className="px-3 py-2">Effect Size</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-app-border">
            {algoNames.map((algo) => {
              const m = algos[algo];
              const pStr = m.p_value_vs_dijkstra != null
                ? m.p_value_vs_dijkstra < 0.001
                  ? "< 0.001"
                  : m.p_value_vs_dijkstra.toFixed(3)
                : "—";

              return (
                <tr key={algo} className="hover:bg-app-input-bg transition-colors duration-200">
                  <td className="px-3 py-2 font-semibold text-app-text">
                    {ALGO_LABELS[algo] || algo}
                  </td>
                  <td className="px-3 py-2 text-app-text font-mono">
                    {m.mean_latency != null
                      ? `${m.mean_latency.toFixed(1)} ms`
                      : "—"}
                  </td>
                  <td className="px-3 py-2 text-app-text">
                    {(m.success_rate * 100).toFixed(1)}%
                  </td>
                  <td
                    className={`px-3 py-2 ${
                      m.fallback_rate > 0.05
                        ? "text-red-400 font-semibold"
                        : "text-app-text"
                    }`}
                  >
                    {(m.fallback_rate * 100).toFixed(1)}%
                  </td>
                  <td
                    className={`px-3 py-2 ${
                      m.dijkstra_match_rate > 0.9 &&
                      algo !== "dijkstra" &&
                      algo !== "bellman_ford"
                        ? "text-amber-400 font-semibold"
                        : "text-app-text"
                    }`}
                  >
                    {(m.dijkstra_match_rate * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 text-app-text text-xs font-mono">
                    {pStr}
                  </td>
                  <td
                    className={`px-3 py-2 font-mono text-xs ${
                      m.effect_size_pct > 0
                        ? "text-red-400"
                        : m.effect_size_pct < 0
                        ? "text-emerald-400"
                        : "text-app-muted"
                    }`}
                  >
                    {algo === "dijkstra"
                      ? "baseline"
                      : m.effect_size_pct != null
                      ? `${m.effect_size_pct > 0 ? "+" : ""}${m.effect_size_pct.toFixed(1)}%`
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Grouped bar chart: latency + variance trade-off */}
      <div className="rounded border border-app-border bg-app-input-bg p-3 shadow-sm">
        <h3 className="mb-2 text-xs font-semibold text-app-muted">
          Latency vs. Utilization Variance Trade-off
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 10, right: 20, left: -10, bottom: 5 }}
            >
              <XAxis
                dataKey="algorithm"
                stroke="currentColor"
                className="text-app-muted"
                tick={{ fontSize: 10 }}
                interval={0}
              />
              <YAxis
                yAxisId="left"
                stroke="currentColor"
                className="text-app-muted"
                tick={{ fontSize: 10 }}
                tickFormatter={(v) => Math.round(v)}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                stroke="currentColor"
                className="text-app-muted"
                tick={{ fontSize: 10 }}
                tickFormatter={(v) => v.toFixed(1)}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--color-panel)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
                labelStyle={{ color: "var(--color-text-main)" }}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px", color: "var(--color-text-muted)" }}
              />
              <Bar
                yAxisId="left"
                dataKey="Mean Latency (ms)"
                fill="var(--color-accent, #3b82f6)"
                radius={[3, 3, 0, 0]}
                opacity={0.85}
              />
              <Bar
                yAxisId="right"
                dataKey="Util Variance (x1000)"
                fill="#f59e0b"
                radius={[3, 3, 0, 0]}
                opacity={0.7}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Known limitations */}
      {showLimitations && limitations && (
        <details className="rounded border border-app-border bg-app-panel shadow-sm">
          <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-app-text hover:bg-app-input-bg transition-colors">
            📘 Known Limitations & Methodology Notes
          </summary>
          <div className="border-t border-app-border px-4 py-3 text-xs text-app-muted leading-relaxed whitespace-pre-wrap">
            {limitations.root_readme_limitation && (
              <div className="mb-3">
                <p className="font-semibold text-app-text mb-1">
                  Root README Limitation Note:
                </p>
                <p>{limitations.root_readme_limitation}</p>
              </div>
            )}
            {limitations.benchmark_readme && (
              <div>
                <p className="font-semibold text-app-text mb-1">
                  Benchmark Suite Notes:
                </p>
                <p>{limitations.benchmark_readme}</p>
              </div>
            )}
          </div>
        </details>
      )}
    </div>
  );
}
