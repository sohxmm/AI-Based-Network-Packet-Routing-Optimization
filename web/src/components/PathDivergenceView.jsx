import { useEffect, useMemo, useState } from "react";

import PathCostBreakdown from "./PathCostBreakdown.jsx";
import TopologyGraph from "./TopologyGraph.jsx";
import { TRAFFIC_CLASS_LABELS, algorithmLabel } from "../utils/colorScales.js";

const DEFAULT_OVERLAY = ["dijkstra", "gnn", "rl"];

/**
 * The view that shows the argument rather than just the network.
 *
 * Same source, same destination, same instant: every selected algorithm's
 * chosen route overlaid on one topology in its own colour, with a one-line
 * verdict and a per-hop cost breakdown underneath. It makes "the AI chose
 * differently, and here is what that cost" immediately legible — including,
 * importantly, when the AI is the one that is wrong.
 */
function PathDivergenceView({
  networkState,
  comparison,
  trafficClass,
  onTrafficClassChange,
  trafficClasses = [],
  isLoading,
  onCompare,
  isDark,
}) {
  const nodes = useMemo(() => networkState?.nodes ?? [], [networkState]);
  const [source, setSource] = useState("");
  const [destination, setDestination] = useState("");
  const [selected, setSelected] = useState(DEFAULT_OVERLAY);

  useEffect(() => {
    if (!nodes.length) return;
    setSource((current) => (current && nodes.includes(current) ? current : nodes[0]));
    setDestination((current) =>
      current && nodes.includes(current)
        ? current
        : nodes[Math.min(13, nodes.length - 1)]
    );
  }, [nodes]);

  const results = useMemo(() => comparison?.results ?? [], [comparison]);

  const overlays = useMemo(
    () =>
      results
        .filter((result) => selected.includes(result.algorithm) && result.path?.length)
        .map((result) => ({
          algorithm: result.algorithm,
          path: result.path,
          total_latency: result.total_latency,
          is_fallback: result.is_fallback,
        })),
    [results, selected]
  );

  const baseline = results.find((result) => result.algorithm === "dijkstra");

  // Ordered best-first, which is the order a reader wants to read them in.
  const ranked = useMemo(
    () =>
      [...results]
        .filter((result) => selected.includes(result.algorithm))
        .sort((a, b) => (a.total_latency ?? Infinity) - (b.total_latency ?? Infinity)),
    [results, selected]
  );

  const verdict = useMemo(() => buildVerdict(ranked, baseline), [ranked, baseline]);

  function toggleAlgorithm(name) {
    setSelected((current) =>
      current.includes(name)
        ? current.filter((item) => item !== name)
        : [...current, name]
    );
  }

  const availableAlgorithms = results.map((result) => result.algorithm);

  return (
    <section className="flex flex-col gap-3 rounded border border-app-border bg-app-panel p-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-app-text">Path divergence</h2>
          <p className="text-xs text-app-muted">
            Where the algorithms disagree, and what disagreeing costs.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col text-xs text-app-muted">
            From
            <select
              value={source}
              onChange={(event) => setSource(event.target.value)}
              className="mt-0.5 rounded border border-app-border bg-app-input-bg px-2 py-1 text-app-text"
            >
              {nodes.map((node) => (
                <option key={node} value={node}>
                  {node}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col text-xs text-app-muted">
            To
            <select
              value={destination}
              onChange={(event) => setDestination(event.target.value)}
              className="mt-0.5 rounded border border-app-border bg-app-input-bg px-2 py-1 text-app-text"
            >
              {nodes.map((node) => (
                <option key={node} value={node}>
                  {node}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col text-xs text-app-muted">
            Traffic class
            <select
              value={trafficClass}
              onChange={(event) => onTrafficClassChange(event.target.value)}
              className="mt-0.5 rounded border border-app-border bg-app-input-bg px-2 py-1 text-app-text"
            >
              {(trafficClasses.length
                ? trafficClasses
                : Object.keys(TRAFFIC_CLASS_LABELS).map((name) => ({ name }))
              ).map((item) => (
                <option key={item.name} value={item.name}>
                  {TRAFFIC_CLASS_LABELS[item.name] ?? item.name}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            disabled={isLoading || !source || !destination || source === destination}
            onClick={() => onCompare(source, destination)}
            className="rounded bg-app-accent px-3 py-1.5 text-xs font-semibold text-app-accent-text disabled:opacity-50"
          >
            {isLoading ? "Routing…" : "Compare"}
          </button>
        </div>
      </header>

      {availableAlgorithms.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {availableAlgorithms.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => toggleAlgorithm(name)}
              aria-pressed={selected.includes(name)}
              className={`rounded-full border px-2.5 py-0.5 text-[11px] transition-colors ${
                selected.includes(name)
                  ? "border-app-accent bg-app-accent/15 text-app-text"
                  : "border-app-border text-app-muted hover:text-app-text"
              }`}
            >
              {algorithmLabel(name)}
            </button>
          ))}
        </div>
      )}

      {verdict && (
        <p className="rounded border border-app-border bg-app-input-bg px-3 py-2 text-xs text-app-text">
          {verdict}
        </p>
      )}

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_320px]">
        <TopologyGraph
          networkState={networkState}
          highlightedPaths={overlays}
          isDark={isDark}
          height={380}
        />

        <div className="flex flex-col gap-2 overflow-y-auto" style={{ maxHeight: 460 }}>
          {ranked.length === 0 ? (
            <p className="text-xs text-app-muted">
              Run a comparison to see how each algorithm routes this demand.
            </p>
          ) : (
            ranked.map((result) => (
              <PathCostBreakdown
                key={result.algorithm}
                result={result}
                baseline={baseline}
                trafficClass={trafficClass}
              />
            ))
          )}

          {comparison?.oracle?.path?.length > 0 && (
            <p className="rounded border border-dashed border-app-border px-3 py-2 text-[11px] text-app-muted">
              <strong className="text-app-text">QoS oracle</strong> for this class
              chose {comparison.oracle.path.join(" → ")}
              {comparison.oracle.feasible
                ? " (constraints satisfied)."
                : " — no candidate satisfies every constraint here."}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function buildVerdict(ranked, baseline) {
  if (!ranked.length) return null;

  const distinct = new Set(ranked.map((result) => JSON.stringify(result.path)));
  if (distinct.size === 1) {
    return "All selected algorithms chose the identical path — no differentiation on this demand.";
  }

  const best = ranked[0];
  if (!baseline || best.algorithm === "dijkstra") {
    const runnerUp = ranked.find((result) => result.algorithm !== "dijkstra");
    if (!runnerUp || runnerUp.total_latency == null || baseline?.total_latency == null) {
      return `${algorithmLabel(best.algorithm)} produced the lowest-cost route.`;
    }
    const delta = runnerUp.total_latency - baseline.total_latency;
    const pct = (delta / baseline.total_latency) * 100;
    return (
      `Dijkstra won here. ${algorithmLabel(runnerUp.algorithm)} diverged and ` +
      `arrived ${delta.toFixed(1)} ms slower (${pct > 0 ? "+" : ""}${pct.toFixed(1)}%).`
    );
  }

  if (best.total_latency == null || baseline.total_latency == null) {
    return `${algorithmLabel(best.algorithm)} produced the lowest-cost route.`;
  }

  const delta = best.total_latency - baseline.total_latency;
  const pct = (delta / baseline.total_latency) * 100;
  const divergedAt = firstDivergence(baseline.path, best.path);
  return (
    `${algorithmLabel(best.algorithm)} diverged from Dijkstra at hop ${divergedAt} ` +
    `and arrived ${Math.abs(delta).toFixed(1)} ms ${delta < 0 ? "faster" : "slower"} ` +
    `(${pct > 0 ? "+" : ""}${pct.toFixed(1)}%).`
  );
}

function firstDivergence(a = [], b = []) {
  const limit = Math.min(a.length, b.length);
  for (let i = 0; i < limit; i += 1) {
    if (a[i] !== b[i]) return i;
  }
  return limit;
}

export default PathDivergenceView;
