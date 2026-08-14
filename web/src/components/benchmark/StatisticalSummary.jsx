import { algorithmLabel } from "../../utils/colorScales.js";

/**
 * The statistics, presented so they cannot be over-read.
 *
 * The previous dashboard displayed a raw percent difference in means under the
 * label "effect size", and p-values that were literally 0.0 — the result of
 * running a paired test over 20,000 autocorrelated decisions from a single
 * trajectory. Both are gone. What is shown now is the number of *independent
 * runs*, a real non-parametric effect size (Cliff's delta) with its
 * interpretation band, and a bootstrap confidence interval on the difference,
 * which conveys practical size in a way a p-value never does.
 */
function StatisticalSummary({ rows, replication }) {
  const comparisons = rows.filter((row) => row.comparison);
  if (!comparisons.length) return null;

  return (
    <section>
      <h3 className="text-xs font-semibold text-app-text">
        Statistical comparison against Dijkstra
      </h3>
      {replication && (
        <p className="mt-0.5 text-[11px] text-app-muted">
          {replication.n_runs} independent seeded runs × {replication.n_steps} steps ×{" "}
          {replication.m_pairs} demands. The unit of replication is one run, not one
          routing decision — decisions inside a run are autocorrelated and testing
          across them produces invalid p-values.
        </p>
      )}

      <div className="mt-2 overflow-x-auto">
        <table className="w-full min-w-[640px] text-xs">
          <thead>
            <tr className="border-b border-app-border text-left text-app-muted">
              <th className="py-1.5 font-medium">Algorithm</th>
              <th className="py-1.5 text-right font-medium">Runs</th>
              <th className="py-1.5 text-right font-medium">Mean diff (ms)</th>
              <th className="py-1.5 text-right font-medium">95% CI</th>
              <th className="py-1.5 text-right font-medium">Cliff&apos;s δ</th>
              <th className="py-1.5 font-medium">Magnitude</th>
              <th className="py-1.5 text-right font-medium">Wilcoxon p</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {comparisons.map((row) => {
              const c = row.comparison;
              return (
                <tr key={row.algorithm} className="border-b border-app-border/50">
                  <td className="py-1.5 font-sans text-app-text">
                    {algorithmLabel(row.algorithm)}
                  </td>
                  <td className="py-1.5 text-right text-app-text">{c.n_runs}</td>
                  <td
                    className={`py-1.5 text-right ${
                      c.mean_diff > 0 ? "text-orange-400" : "text-blue-400"
                    }`}
                  >
                    {c.mean_diff > 0 ? "+" : ""}
                    {c.mean_diff?.toFixed(2) ?? "—"}
                  </td>
                  <td className="py-1.5 text-right text-app-muted">
                    {c.ci95_low != null
                      ? `[${c.ci95_low.toFixed(2)}, ${c.ci95_high.toFixed(2)}]`
                      : "—"}
                  </td>
                  <td className="py-1.5 text-right text-app-text">
                    {c.cliffs_delta?.toFixed(3) ?? "—"}
                  </td>
                  <td className="py-1.5 font-sans text-app-muted">
                    {c.effect_magnitude ?? "—"}
                  </td>
                  <td className="py-1.5 text-right text-app-text">
                    {c.wilcoxon_p_value == null
                      ? c.wilcoxon_note
                        ? "n/a"
                        : "—"
                      : c.wilcoxon_p_value < 0.001
                        ? "<0.001"
                        : c.wilcoxon_p_value.toFixed(3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="mt-1.5 text-[11px] text-app-muted">
        Cliff&apos;s δ is a non-parametric effect size in [-1, 1]; negative means the
        algorithm is faster than Dijkstra. Bands: &lt;0.147 negligible, &lt;0.33 small,
        &lt;0.474 medium, otherwise large.
      </p>
    </section>
  );
}

export default StatisticalSummary;
