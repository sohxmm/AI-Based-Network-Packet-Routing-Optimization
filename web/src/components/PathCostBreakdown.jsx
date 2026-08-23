import { useState } from "react";

import GuardrailBadge from "./GuardrailBadge.jsx";
import { algorithmColor, algorithmLabel } from "../utils/colorScales.js";

/**
 * Explain *why* a path cost what it did.
 *
 * The dashboard used to show the network but never the argument. A reviewer saw
 * a pretty graph and a table of numbers and had to infer the point. This makes
 * one algorithm's decision legible hop by hop: the base latency of each link,
 * how loaded it was, and what that combination cost.
 */
function PathCostBreakdown({ result, baseline, trafficClass }) {
  const [expanded, setExpanded] = useState(false);

  if (!result) return null;

  const color = algorithmColor(result.algorithm);
  const hops = result.hops ?? [];
  // `result.qos` is the server's own evaluation of the returned path and is
  // present for every algorithm. `diagnostics.qos` is whatever the router chose
  // to report about itself, and only the constraint-aware ones report anything
  // — so reading it first left Dijkstra and Bellman-Ford, the two
  // constraint-blind routers, as the only rows with no feasibility verdict.
  const qos = result.qos ?? result.diagnostics?.qos;

  const delta =
    baseline?.total_latency != null && result.total_latency != null
      ? result.total_latency - baseline.total_latency
      : null;
  const deltaPct =
    delta != null && baseline.total_latency
      ? (delta / baseline.total_latency) * 100
      : null;

  const samePathAsBaseline =
    baseline && JSON.stringify(baseline.path) === JSON.stringify(result.path);

  return (
    <div className="rounded border border-app-border bg-app-input-bg">
      <button
        type="button"
        onClick={() => setExpanded((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left"
        aria-expanded={expanded}
      >
        <span className="flex min-w-0 items-center gap-2">
          <span
            className="h-3 w-3 shrink-0 rounded-full"
            style={{ backgroundColor: color }}
            aria-hidden="true"
          />
          <span className="truncate text-sm font-medium text-app-text">
            {algorithmLabel(result.algorithm)}
          </span>
        </span>

        <span className="flex shrink-0 items-center gap-2 text-xs">
          <span className="font-mono text-app-text">
            {result.total_latency == null
              ? "no path"
              : `${result.total_latency.toFixed(1)} ms`}
          </span>
          {delta != null && delta !== 0 && (
            <span
              className={`rounded px-1.5 py-0.5 font-mono ${
                delta > 0
                  ? "bg-red-500/15 text-red-400"
                  : "bg-emerald-500/15 text-emerald-400"
              }`}
            >
              {delta > 0 ? "+" : ""}
              {delta.toFixed(1)} ms
              {deltaPct != null && ` (${deltaPct > 0 ? "+" : ""}${deltaPct.toFixed(1)}%)`}
            </span>
          )}
          <span className="text-app-muted">{expanded ? "▲" : "▼"}</span>
        </span>
      </button>

      <div className="flex flex-wrap gap-1.5 px-3 pb-2">
        {result.is_fallback && (
          <GuardrailBadge
            tone="warn"
            label="Heuristic fallback"
            title="No trained model was loaded, so this decision came from a heuristic, not from AI."
          />
        )}
        {samePathAsBaseline && result.algorithm !== "dijkstra" && (
          <GuardrailBadge
            tone="neutral"
            label="Matches Dijkstra"
            title="Identical to the baseline path, so this algorithm added no differentiation here."
          />
        )}
        {qos && !qos.feasible && (
          <GuardrailBadge
            tone="bad"
            label={`Violates ${trafficClass} QoS`}
            title={(qos.violations ?? []).join("; ")}
          />
        )}
        {qos && qos.feasible && (
          <GuardrailBadge
            tone="ok"
            label="QoS satisfied"
            title={`Bottleneck ${(qos.bottleneck_utilization * 100).toFixed(0)}%, ${qos.hops} hops`}
          />
        )}
      </div>

      {expanded && (
        <div className="border-t border-app-border px-3 py-2">
          {hops.length === 0 ? (
            <p className="text-xs text-app-muted">No path to break down.</p>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-app-muted">
                  <th className="pb-1 font-medium">Hop</th>
                  <th className="pb-1 text-right font-medium">Base</th>
                  <th className="pb-1 text-right font-medium">Util</th>
                  <th className="pb-1 text-right font-medium">Cost</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {hops.map((hop, index) => (
                  <tr key={`${hop.from}-${hop.to}-${index}`} className="text-app-text">
                    <td className="py-0.5">
                      {hop.from} → {hop.to}
                    </td>
                    <td className="py-0.5 text-right">{hop.base_latency.toFixed(0)} ms</td>
                    <td
                      className={`py-0.5 text-right ${
                        hop.utilization > 0.7 ? "text-red-400" : "text-app-muted"
                      }`}
                    >
                      {(hop.utilization * 100).toFixed(0)}%
                    </td>
                    <td className="py-0.5 text-right">{hop.cost.toFixed(1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {qos && (
            <dl className="mt-2 grid grid-cols-3 gap-2 border-t border-app-border pt-2 text-xs">
              <div>
                <dt className="text-app-muted">Hops</dt>
                <dd className="font-mono text-app-text">{qos.hops}</dd>
              </div>
              <div>
                <dt className="text-app-muted">Bottleneck</dt>
                <dd className="font-mono text-app-text">
                  {(qos.bottleneck_utilization * 100).toFixed(0)}%
                </dd>
              </div>
              <div>
                <dt className="text-app-muted">Path loss</dt>
                <dd className="font-mono text-app-text">
                  {(qos.total_loss * 100).toFixed(2)}%
                </dd>
              </div>
            </dl>
          )}
        </div>
      )}
    </div>
  );
}

export default PathCostBreakdown;
