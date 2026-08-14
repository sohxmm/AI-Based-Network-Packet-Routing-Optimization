import GuardrailBadge from "../GuardrailBadge.jsx";
import { algorithmColor, algorithmLabel } from "../../utils/colorScales.js";

const DEGENERACY_THRESHOLD = 0.95;
const FALLBACK_THRESHOLD = 0.5;
const LEARNED = new Set(["gnn", "rl", "multi_agent"]);

/**
 * The table view. It is not optional decoration next to the charts: three of
 * the light-mode categorical slots fall below the 3:1 contrast floor, so a
 * non-colour route to every number has to exist. It is also the honest place to
 * put the guardrails, since a badge next to a row is harder to skim past than a
 * footnote.
 */
function AlgorithmMetricsTable({ rows, isDark }) {
  if (!rows?.length) return null;

  const sorted = [...rows].sort(
    (a, b) => (a.mean_latency ?? Infinity) - (b.mean_latency ?? Infinity)
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-xs">
        <caption className="sr-only">
          Per-algorithm benchmark metrics, sorted by mean latency
        </caption>
        <thead>
          <tr className="border-b border-app-border text-left text-app-muted">
            <th className="py-1.5 font-medium">Algorithm</th>
            <th className="py-1.5 text-right font-medium">Latency (ms)</th>
            <th className="py-1.5 text-right font-medium">vs Dijkstra</th>
            <th className="py-1.5 text-right font-medium">QoS met</th>
            <th className="py-1.5 text-right font-medium">Success</th>
            <th className="py-1.5 text-right font-medium">Fallback</th>
            <th className="py-1.5 text-right font-medium">Diversity</th>
            <th className="py-1.5 text-right font-medium">Bottleneck p95</th>
            <th className="py-1.5 font-medium">Guardrails</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const degenerate =
              row.dijkstra_match_rate > DEGENERACY_THRESHOLD &&
              !["dijkstra", "bellman_ford", "constrained"].includes(row.algorithm);
            const fellBack = LEARNED.has(row.algorithm) && row.fallback_rate > FALLBACK_THRESHOLD;

            return (
              <tr key={row.algorithm} className="border-b border-app-border/50">
                <td className="py-1.5">
                  <span className="flex items-center gap-1.5">
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ backgroundColor: algorithmColor(row.algorithm, isDark) }}
                      aria-hidden="true"
                    />
                    <span className="text-app-text">{algorithmLabel(row.algorithm)}</span>
                  </span>
                </td>
                <td className="py-1.5 text-right font-mono text-app-text">
                  {fmt(row.mean_latency, 1)}
                </td>
                <td
                  className={`py-1.5 text-right font-mono ${
                    row.pct_diff == null
                      ? "text-app-muted"
                      : row.pct_diff > 0
                        ? "text-orange-400"
                        : "text-blue-400"
                  }`}
                >
                  {row.pct_diff == null
                    ? "—"
                    : `${row.pct_diff > 0 ? "+" : ""}${row.pct_diff.toFixed(1)}%`}
                </td>
                <td className="py-1.5 text-right font-mono text-app-text">
                  {pct(row.qos_satisfaction_rate)}
                </td>
                <td className="py-1.5 text-right font-mono text-app-text">
                  {pct(row.success_rate)}
                </td>
                <td
                  className={`py-1.5 text-right font-mono ${
                    fellBack ? "text-red-400" : "text-app-text"
                  }`}
                >
                  {pct(row.fallback_rate)}
                </td>
                <td className="py-1.5 text-right font-mono text-app-text">
                  {fmt(row.diversity_index, 3)}
                </td>
                <td className="py-1.5 text-right font-mono text-app-text">
                  {pct(row.p95_path_max_utilization)}
                </td>
                <td className="py-1.5">
                  <span className="flex flex-wrap gap-1">
                    {fellBack && <GuardrailBadge type="fallback" compact />}
                    {degenerate && <GuardrailBadge type="dijkstra-match" compact />}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function fmt(value, digits) {
  return value == null || Number.isNaN(value) ? "—" : value.toFixed(digits);
}

function pct(value) {
  return value == null || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(0)}%`;
}

export default AlgorithmMetricsTable;
