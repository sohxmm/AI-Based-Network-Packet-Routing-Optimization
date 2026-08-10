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
        message: `${ALGO_LABELS[algo] || algo}: Degenerate policy, matches Dijkstra ${(m.dijkstra_match_rate * 100).toFixed(1)}% of time.`,
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
                : "-";

              return (
                <tr key={algo} className="hover:bg-app-input-bg transition-colors duration-200">
                  <td className="px-3 py-2 font-semibold text-app-text">
                    {ALGO_LABELS[algo] || algo}
                  </td>
                  <td className="px-3 py-2 text-app-text font-mono">
                    {m.mean_latency != null
                      ? `${m.mean_latency.toFixed(1)} ms`
                      : "-"}
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
                      : "-"}
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

      {/* ── Results Insights ── */}
      <ResultsInsights algos={algos} algoNames={algoNames} />

    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════════════════
 * ResultsInsights — dynamic, plain-language explanation of the benchmark
 * numbers for every algorithm tested in this experiment run.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Rating levels used for colour-coded badges */
const LEVEL = { GOOD: "good", OK: "ok", WARN: "warn", CRIT: "crit" };

const LEVEL_STYLES = {
  good: "bg-emerald-500/10 text-emerald-400 border-emerald-500/25",
  ok:   "bg-sky-500/10 text-sky-400 border-sky-500/25",
  warn: "bg-amber-500/10 text-amber-500 border-amber-500/25",
  crit: "bg-red-500/10 text-red-400 border-red-500/25",
};

const LEVEL_LABELS = {
  good: "Excellent",
  ok:   "Acceptable",
  warn: "Warning",
  crit: "Critical",
};

const LEVEL_ICONS = {
  good: "✅",
  ok:   "ℹ️",
  warn: "⚠️",
  crit: "🚨",
};

/* ── Per-metric rating functions ─────────────────────────────────────────── */

function rateSuccessRate(v) {
  if (v >= 0.98) return { level: LEVEL.GOOD, text: `${(v * 100).toFixed(1)}%. Nearly all routes succeeded.` };
  if (v >= 0.90) return { level: LEVEL.OK,   text: `${(v * 100).toFixed(1)}%. Most routes succeeded, some failures present.` };
  if (v >= 0.75) return { level: LEVEL.WARN, text: `${(v * 100).toFixed(1)}%. Noticeable routing failures; network may be under stress.` };
  return { level: LEVEL.CRIT, text: `${(v * 100).toFixed(1)}%. Many routes failed. Algorithm may not cope with current conditions.` };
}

function rateFallbackRate(v, isClassic) {
  if (isClassic) return { level: LEVEL.GOOD, text: `${(v * 100).toFixed(1)}%. N/A for classical algorithms (they never fall back).` };
  if (v <= 0.02) return { level: LEVEL.GOOD, text: `${(v * 100).toFixed(1)}%. AI model produced its own routes almost every time.` };
  if (v <= 0.05) return { level: LEVEL.OK,   text: `${(v * 100).toFixed(1)}%. Occasional fallback to Dijkstra; still within tolerance (≤5%).` };
  if (v <= 0.15) return { level: LEVEL.WARN, text: `${(v * 100).toFixed(1)}%. Elevated fallback. The AI model struggled to produce valid routes.` };
  return { level: LEVEL.CRIT, text: `${(v * 100).toFixed(1)}%. High fallback rate. The AI policy is essentially not working.` };
}

function rateDijkstraMatch(v, algo) {
  if (algo === "dijkstra") return { level: LEVEL.GOOD, text: "100.0%. This is the Dijkstra baseline." };
  if (algo === "bellman_ford") return { level: LEVEL.OK, text: `${(v * 100).toFixed(1)}%. Expected: Bellman-Ford often finds the same shortest path.` };
  if (v <= 0.40) return { level: LEVEL.GOOD, text: `${(v * 100).toFixed(1)}%. The AI found distinctly different routes from Dijkstra.` };
  if (v <= 0.70) return { level: LEVEL.OK,   text: `${(v * 100).toFixed(1)}%. Some overlap with Dijkstra, but the AI still differentiates.` };
  if (v <= 0.90) return { level: LEVEL.WARN, text: `${(v * 100).toFixed(1)}%. High overlap. The AI is not adding much value over Dijkstra.` };
  return { level: LEVEL.CRIT, text: `${(v * 100).toFixed(1)}%. Degenerate: the model copies Dijkstra nearly every time.` };
}

function rateTradeoff(algo, m, dijkstraM) {
  if (algo === "dijkstra") return { level: LEVEL.OK, text: "Baseline. All other algorithms are compared against this." };
  if (m.effect_size_pct == null || !dijkstraM) return { level: LEVEL.OK, text: "No baseline data to compute trade-off." };

  const effect = m.effect_size_pct;
  const utilVar = (m.utilization_variance ?? m.util_variance ?? 0);
  const dUtilVar = (dijkstraM.utilization_variance ?? dijkstraM.util_variance ?? 0);
  
  const varChangePct = dUtilVar > 0 ? ((utilVar - dUtilVar) / dUtilVar * 100) : 0;
  
  if (effect > 5 && Math.abs(varChangePct) < 5) {
    return { 
      level: LEVEL.OK, 
      text: `Added +${effect.toFixed(1)}% latency cost with negligible balancing benefit in this specific low-congestion condition.` 
    };
  }
  
  const latencyStr = effect > 0 ? `+${effect.toFixed(1)}%` : `${effect.toFixed(1)}%`;
  const varStr = varChangePct > 0 ? `+${varChangePct.toFixed(1)}%` : `${varChangePct.toFixed(1)}%`;
  
  let suffix = "";
  if (effect > 0 && varChangePct > 0) {
    suffix = " (worse on both metrics, no clear benefit)";
  }
  
  return {
    level: LEVEL.OK,
    text: `Costs ${latencyStr} latency to change ${dUtilVar.toFixed(3)}→${utilVar.toFixed(3)} utilization variance (${varStr} change)${suffix}.`
  };
}

function ratePValue(pv, algo) {
  if (algo === "dijkstra") return null; // skip for baseline
  if (pv == null) return { level: LEVEL.OK, text: "No statistical test available for this comparison." };
  if (pv < 0) return { level: LEVEL.WARN, text: "Statistical test encountered an error." };
  if (pv < 0.001) return { level: LEVEL.GOOD, text: `p < 0.001. Highly significant difference from Dijkstra.` };
  if (pv < 0.05)  return { level: LEVEL.GOOD, text: `p = ${pv.toFixed(3)}. Statistically significant difference (p < 0.05).` };
  if (pv < 0.10)  return { level: LEVEL.OK,   text: `p = ${pv.toFixed(3)}. Marginal significance; results may be noise.` };
  return { level: LEVEL.WARN, text: `p = ${pv.toFixed(3)}. Not statistically significant. Difference could be due to chance.` };
}

/** Build a full set of insights for a single algorithm */
function buildAlgoInsights(algo, m, algos) {
  const isClassic = ["dijkstra", "bellman_ford", "aco"].includes(algo);
  const lines = [];

  lines.push({ metric: "Success Rate",    ...rateSuccessRate(m.success_rate) });
  lines.push({ metric: "Fallback Rate",   ...rateFallbackRate(m.fallback_rate, isClassic) });
  lines.push({ metric: "Dijkstra Match",  ...rateDijkstraMatch(m.dijkstra_match_rate, algo) });
  
  const dijkstraM = algos["dijkstra"];
  lines.push({ metric: "Performance Trade-off", ...rateTradeoff(algo, m, dijkstraM) });

  // p-value (skip for dijkstra)
  const pVal = m.p_value_vs_dijkstra ?? m.wilcoxon_p_value ?? null;
  const pInsight = ratePValue(pVal, algo);
  if (pInsight) lines.push({ metric: "p-value (vs Dijkstra)", ...pInsight });

  return lines;
}


function ResultsInsights({ algos, algoNames }) {
  if (!algos || algoNames.length === 0) return null;

  return (
    <div className="rounded border border-app-border bg-app-panel shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-app-border bg-app-input-bg px-4 py-3">
        <span className="text-base">🔍</span>
        <h3 className="text-sm font-semibold text-app-text">Results Insights</h3>
        <span className="ml-auto text-[10px] text-app-muted">
          Auto-generated from benchmark metrics
        </span>
      </div>

      {/* Metric Guide Legend */}
      <div className="border-b border-app-border px-4 py-2.5 flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-app-muted">
          Rating Guide:
        </span>
        {Object.entries(LEVEL_LABELS).map(([lvl, label]) => (
          <span
            key={lvl}
            className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium ${LEVEL_STYLES[lvl]}`}
          >
            {LEVEL_ICONS[lvl]} {label}
          </span>
        ))}
      </div>

      {/* Per-algorithm insight cards */}
      <div className="divide-y divide-app-border">
        {algoNames.map((algo) => {
          const m = algos[algo];
          const insights = buildAlgoInsights(algo, m, algos);
          // Determine an overall vibe
          const hasAnyCrit = insights.some((i) => i.level === LEVEL.CRIT);
          const hasAnyWarn = insights.some((i) => i.level === LEVEL.WARN);
          const allGood = insights.every((i) => i.level === LEVEL.GOOD || i.level === LEVEL.OK);

          let overallIcon = "✅";
          let overallText = "All metrics look healthy.";
          if (hasAnyCrit) {
            overallIcon = "🚨";
            overallText = "One or more metrics are in critical range — review this algorithm's suitability.";
          } else if (hasAnyWarn) {
            overallIcon = "⚠️";
            overallText = "Some metrics need attention but the algorithm may still be viable.";
          }

          return (
            <details key={algo} className="group">
              <summary className="flex cursor-pointer items-center gap-3 px-4 py-3 hover:bg-app-input-bg transition-colors">
                <span className="text-sm">{overallIcon}</span>
                <span className="text-sm font-semibold text-app-text">
                  {ALGO_LABELS[algo] || algo}
                </span>
                {/* Mini pills showing metric levels at a glance */}
                <span className="ml-auto flex items-center gap-1">
                  {insights.map((ins, idx) => (
                    <span
                      key={idx}
                      title={`${ins.metric}: ${LEVEL_LABELS[ins.level]}`}
                      className={`inline-block h-2 w-2 rounded-full border ${LEVEL_STYLES[ins.level]}`}
                      style={{
                        backgroundColor:
                          ins.level === "good" ? "rgb(16 185 129 / 0.6)" :
                          ins.level === "ok"   ? "rgb(56 189 248 / 0.6)" :
                          ins.level === "warn" ? "rgb(245 158 11 / 0.6)" :
                                                 "rgb(239 68 68 / 0.6)",
                      }}
                    />
                  ))}
                </span>
                <span className="text-[10px] text-app-muted group-open:rotate-90 transition-transform">▶</span>
              </summary>
              <div className="px-4 pb-3 pt-1">
                <p className="text-[11px] text-app-muted mb-2 italic">{overallText}</p>
                <div className="flex flex-col gap-1.5">
                  {insights.map((ins, idx) => (
                    <div
                      key={idx}
                      className={`flex items-start gap-2 rounded border px-3 py-2 text-xs ${LEVEL_STYLES[ins.level]}`}
                    >
                      <span className="mt-px shrink-0 text-[11px]">{LEVEL_ICONS[ins.level]}</span>
                      <div>
                        <span className="font-semibold">{ins.metric}:</span>{" "}
                        <span className="opacity-90">{ins.text}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </details>
          );
        })}
      </div>

      {/* Reference table: what ranges mean */}
      <div className="border-t border-app-border bg-app-input-bg px-4 py-3">
        <h4 className="text-[10px] font-semibold uppercase tracking-wider text-app-muted mb-2">
          Metric Reference
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] text-app-muted">
            <thead>
              <tr className="border-b border-app-border text-left">
                <th className="pb-1.5 pr-4 font-semibold text-app-text">Metric</th>
                <th className="pb-1.5 pr-4 font-semibold text-emerald-400">Excellent</th>
                <th className="pb-1.5 pr-4 font-semibold text-sky-400">Acceptable</th>
                <th className="pb-1.5 pr-4 font-semibold text-amber-500">Warning</th>
                <th className="pb-1.5 font-semibold text-red-400">Critical</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-app-border/50">
              <tr>
                <td className="py-1.5 pr-4 text-app-text font-medium">Success Rate</td>
                <td className="py-1.5 pr-4">≥ 98%</td>
                <td className="py-1.5 pr-4">90 – 97%</td>
                <td className="py-1.5 pr-4">75 – 89%</td>
                <td className="py-1.5">&lt; 75%</td>
              </tr>
              <tr>
                <td className="py-1.5 pr-4 text-app-text font-medium">Fallback Rate</td>
                <td className="py-1.5 pr-4">≤ 2%</td>
                <td className="py-1.5 pr-4">2 – 5%</td>
                <td className="py-1.5 pr-4">5 – 15%</td>
                <td className="py-1.5">&gt; 15%</td>
              </tr>
              <tr>
                <td className="py-1.5 pr-4 text-app-text font-medium">Dijkstra Match</td>
                <td className="py-1.5 pr-4">≤ 40%</td>
                <td className="py-1.5 pr-4">40 – 70%</td>
                <td className="py-1.5 pr-4">70 – 90%</td>
                <td className="py-1.5">&gt; 90%</td>
              </tr>
              <tr>
                <td className="py-1.5 pr-4 text-app-text font-medium">Performance Trade-off</td>
                <td className="py-1.5 pr-4 text-app-muted" colSpan="4">Evaluated holistically based on latency cost vs. utilization variance benefit.</td>
              </tr>
              <tr>
                <td className="py-1.5 pr-4 text-app-text font-medium">p-value</td>
                <td className="py-1.5 pr-4">&lt; 0.05</td>
                <td className="py-1.5 pr-4">0.05 – 0.10</td>
                <td className="py-1.5 pr-4">&gt; 0.10</td>
                <td className="py-1.5">Error</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
