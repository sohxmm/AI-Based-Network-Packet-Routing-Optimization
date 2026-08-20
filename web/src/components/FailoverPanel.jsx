import { useEffect, useMemo, useState } from "react";

import { API_BASE_URL } from "../config.js";
import { algorithmLabel } from "../utils/colorScales.js";

/**
 * Fault-tolerant rerouting, and how fast each algorithm actually recovers.
 *
 * Automatic rerouting on its own is not interesting — every router here
 * recomputes from the current state, so they all "recover". What is worth
 * measuring is the *cost* of recovery: how many ticks until service is restored
 * under the traffic class's constraints, and what the route costs afterwards. A
 * fast recovery onto a saturated link is not better than a slower one onto a
 * good path, so both numbers are shown together.
 */
function FailoverPanel({ networkState, failoverEvents, onWatch, onConvergence, isLoading }) {
  const nodes = useMemo(() => networkState?.nodes ?? [], [networkState]);
  const links = useMemo(() => networkState?.links ?? [], [networkState]);

  const [source, setSource] = useState("");
  const [destination, setDestination] = useState("");
  const [linkIndex, setLinkIndex] = useState(0);
  const [watched, setWatched] = useState([]);
  const [convergence, setConvergence] = useState(null);

  useEffect(() => {
    if (!nodes.length) return;
    setSource((current) => (nodes.includes(current) ? current : nodes[0]));
    setDestination((current) =>
      nodes.includes(current) ? current : nodes[Math.min(13, nodes.length - 1)]
    );
  }, [nodes]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/network/failover`)
      .then((response) => response.json())
      .then((data) => setWatched(data.watched ?? []))
      .catch(() => undefined);
  }, [failoverEvents]);

  async function handleWatch() {
    const result = await onWatch({ source, destination, traffic_class: "best_effort" });
    if (result?.watching) setWatched(result.watching);
  }

  async function handleConvergence() {
    const link = links[linkIndex];
    if (!link) return;
    const result = await onConvergence({
      source,
      destination,
      link_source: link.source,
      link_target: link.target,
      traffic_class: "best_effort",
    });
    if (result) setConvergence(result);
  }

  return (
    <section className="rounded border border-app-border bg-app-panel p-4">
      <h2 className="text-sm font-semibold text-app-text">Fault tolerance</h2>
      <p className="mt-0.5 text-xs text-app-muted">
        Watch a flow, break a link, measure the recovery.
      </p>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <label className="text-xs text-app-muted">
          From
          <select
            value={source}
            onChange={(event) => setSource(event.target.value)}
            className="mt-0.5 w-full rounded border border-app-border bg-app-input-bg px-2 py-1 text-app-text"
          >
            {nodes.map((node) => (
              <option key={node} value={node}>
                {node}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-app-muted">
          To
          <select
            value={destination}
            onChange={(event) => setDestination(event.target.value)}
            className="mt-0.5 w-full rounded border border-app-border bg-app-input-bg px-2 py-1 text-app-text"
          >
            {nodes.map((node) => (
              <option key={node} value={node}>
                {node}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label className="mt-2 block text-xs text-app-muted">
        Link to fail
        <select
          value={linkIndex}
          onChange={(event) => setLinkIndex(Number(event.target.value))}
          className="mt-0.5 w-full rounded border border-app-border bg-app-input-bg px-2 py-1 font-mono text-app-text"
        >
          {links.map((link, index) => (
            <option key={`${link.source}-${link.target}`} value={index}>
              {link.source}–{link.target} ({Math.round(link.utilization * 100)}%)
            </option>
          ))}
        </select>
      </label>

      <div className="mt-2 flex gap-2">
        <button
          type="button"
          onClick={handleWatch}
          disabled={isLoading || !source || source === destination}
          className="flex-1 rounded border border-app-border bg-app-input-bg px-2 py-1.5 text-xs text-app-text disabled:opacity-50"
        >
          Watch flow
        </button>
        <button
          type="button"
          onClick={handleConvergence}
          disabled={isLoading || !source || source === destination}
          className="flex-1 rounded bg-app-accent px-2 py-1.5 text-xs font-semibold text-app-accent-text disabled:opacity-50"
        >
          {isLoading ? "Measuring…" : "Measure recovery"}
        </button>
      </div>

      {watched.length > 0 && (
        <div className="mt-3 border-t border-app-border pt-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-app-muted">
            Watched flows
          </h3>
          <ul className="mt-1 space-y-0.5 font-mono text-[11px] text-app-text">
            {watched.map((flow) => (
              <li key={`${flow.source}-${flow.destination}-${flow.traffic_class}`}>
                {flow.source} → {flow.destination}{" "}
                <span className="text-app-muted">
                  via {flow.path?.length ? flow.path.join("→") : "no route"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {convergence && (
        <div className="mt-3 border-t border-app-border pt-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-app-muted">
            Recovery after {convergence.failed_link.join("–")} failed
          </h3>
          <table className="mt-1 w-full text-[11px]">
            <thead>
              <tr className="text-left text-app-muted">
                <th className="py-1 font-medium">Algorithm</th>
                <th className="py-1 text-right font-medium">Ticks</th>
                <th className="py-1 text-right font-medium">Before</th>
                <th className="py-1 text-right font-medium">After</th>
              </tr>
            </thead>
            <tbody className="font-mono text-app-text">
              {convergence.results.map((row) => (
                <tr key={row.algorithm}>
                  <td className="py-0.5 font-sans">{algorithmLabel(row.algorithm)}</td>
                  <td
                    className={`py-0.5 text-right ${
                      row.converged ? "" : "text-red-400"
                    }`}
                  >
                    {row.converged ? row.convergence_steps : "no recovery"}
                  </td>
                  <td className="py-0.5 text-right">
                    {row.latency_before?.toFixed(0) ?? "—"}
                  </td>
                  <td className="py-0.5 text-right">
                    {row.latency_after?.toFixed(0) ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1.5 text-[11px] text-app-muted">{convergence.note}</p>
        </div>
      )}

      {failoverEvents?.length > 0 && (
        <div className="mt-3 border-t border-app-border pt-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wide text-app-muted">
            Automatic reroutes
          </h3>
          <ul className="mt-1 space-y-1 text-[11px]">
            {failoverEvents.slice(0, 5).map((event, index) => (
              <li key={`${event.step}-${index}`} className="text-app-text">
                <span className="text-app-muted">step {event.step}</span>{" "}
                {event.source}→{event.destination}{" "}
                <span className="text-amber-400">{event.reason}</span>{" "}
                {event.recovered ? (
                  <span className="font-mono text-app-muted">
                    → {event.new_path.join("→")}
                  </span>
                ) : (
                  <span className="text-red-400">unrecoverable</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

export default FailoverPanel;
