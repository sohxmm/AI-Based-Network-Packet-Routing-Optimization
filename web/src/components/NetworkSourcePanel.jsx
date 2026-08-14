import { useEffect, useState } from "react";

import { API_BASE_URL } from "../config.js";

const DEFAULT_HOSTS = "1.1.1.1, 8.8.8.8, 9.9.9.9";

/**
 * Switch what the platform is actually looking at.
 *
 * Three sources, and the differences between them matter enough to state in the
 * UI rather than bury in docs:
 *
 * - **Simulator** — synthetic and closed-loop. Routing decisions change the
 *   network, so this is the only source any published benchmark number uses.
 * - **Trace** — replayed measurements. Deterministic and repeatable, but open
 *   loop: our routing cannot affect a recording.
 * - **Live** — real ICMP round-trip measurements from this machine, so anyone
 *   can point the dashboard at their own network. Read-only, explicit targets
 *   only, and it produces a star topology with exactly one path per
 *   destination — so it demonstrates telemetry and congestion detection, not
 *   routing quality. The panel says so, because a reviewer should not have to
 *   work that out for themselves.
 */
function NetworkSourcePanel({ onSourceChange, isLoading }) {
  const [source, setSource] = useState(null);
  const [kind, setKind] = useState("simulated");
  const [hosts, setHosts] = useState(DEFAULT_HOSTS);
  const [tracePath, setTracePath] = useState("datasets/example_trace.jsonl");
  const [message, setMessage] = useState(null);
  const [liveHealth, setLiveHealth] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/network/source`)
      .then((response) => response.json())
      .then((data) => {
        setSource(data);
        setKind(data.kind ?? "simulated");
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (source?.kind !== "live") {
      setLiveHealth(null);
      return undefined;
    }
    const poll = () =>
      fetch(`${API_BASE_URL}/sim/source/health`)
        .then((response) => response.json())
        .then((data) => setLiveHealth(data.targets ?? null))
        .catch(() => undefined);
    poll();
    const timer = window.setInterval(poll, 5000);
    return () => window.clearInterval(timer);
  }, [source?.kind]);

  async function apply() {
    setMessage(null);
    const payload = { kind };
    if (kind === "trace") payload.trace_path = tracePath;
    if (kind === "live") {
      payload.targets = hosts
        .split(",")
        .map((host) => host.trim())
        .filter(Boolean);
    }

    const result = await onSourceChange(payload);
    if (result?.source) {
      setSource(result.source);
      setMessage(null);
    } else {
      setMessage("Could not switch source — see the error above.");
    }
  }

  return (
    <section className="rounded border border-app-border bg-app-panel p-4">
      <h2 className="text-sm font-semibold text-app-text">Network source</h2>
      <p className="mt-0.5 text-xs text-app-muted">
        What the dashboard is measuring.
      </p>

      <div className="mt-3 flex gap-1.5">
        {["simulated", "trace", "live"].map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => setKind(option)}
            aria-pressed={kind === option}
            className={`flex-1 rounded border px-2 py-1 text-xs capitalize transition-colors ${
              kind === option
                ? "border-app-accent bg-app-accent/15 text-app-text"
                : "border-app-border text-app-muted hover:text-app-text"
            }`}
          >
            {option}
          </button>
        ))}
      </div>

      {kind === "trace" && (
        <label className="mt-2 block text-xs text-app-muted">
          Trace file
          <input
            value={tracePath}
            onChange={(event) => setTracePath(event.target.value)}
            className="mt-0.5 w-full rounded border border-app-border bg-app-input-bg px-2 py-1 font-mono text-app-text"
            placeholder="datasets/example_trace.jsonl"
          />
        </label>
      )}

      {kind === "live" && (
        <div className="mt-2">
          <label className="block text-xs text-app-muted">
            Hosts to probe (comma separated)
            <input
              value={hosts}
              onChange={(event) => setHosts(event.target.value)}
              className="mt-0.5 w-full rounded border border-app-border bg-app-input-bg px-2 py-1 font-mono text-app-text"
              placeholder={DEFAULT_HOSTS}
            />
          </label>
          <p className="mt-1.5 rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-400">
            Sends one ICMP echo per host per tick, read-only, no root required.
            Only probe networks you are authorised to measure. Requires
            <code className="mx-1 font-mono">LIVE_PROBE_ENABLED=1</code>
            on the backend.
          </p>
        </div>
      )}

      <button
        type="button"
        onClick={apply}
        disabled={isLoading}
        className="mt-2 w-full rounded bg-app-accent px-3 py-1.5 text-xs font-semibold text-app-accent-text disabled:opacity-50"
      >
        {isLoading ? "Switching…" : "Apply"}
      </button>

      {message && <p className="mt-1.5 text-[11px] text-amber-400">{message}</p>}

      {source && (
        <dl className="mt-3 space-y-1 border-t border-app-border pt-2 text-[11px]">
          <div className="flex justify-between gap-2">
            <dt className="text-app-muted">Active</dt>
            <dd className="font-mono text-app-text">{source.kind}</dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt className="text-app-muted">Closed loop</dt>
            <dd className="font-mono text-app-text">
              {source.closed_loop ? "yes" : "no"}
            </dd>
          </div>
          {source.num_nodes != null && (
            <div className="flex justify-between gap-2">
              <dt className="text-app-muted">Nodes / edges</dt>
              <dd className="font-mono text-app-text">
                {source.num_nodes} / {source.num_edges}
              </dd>
            </div>
          )}
          {source.avg_degree != null && (
            <div className="flex justify-between gap-2">
              <dt className="text-app-muted">Avg degree / diameter</dt>
              <dd className="font-mono text-app-text">
                {source.avg_degree.toFixed(1)} / {source.diameter ?? "—"}
              </dd>
            </div>
          )}
          {source.warning && (
            <p className="pt-1 text-[11px] text-amber-400">{source.warning}</p>
          )}
          {source.note && (
            <p className="pt-1 text-[11px] text-app-muted">{source.note}</p>
          )}
        </dl>
      )}

      {liveHealth && (
        <table className="mt-2 w-full border-t border-app-border pt-2 text-[11px]">
          <thead>
            <tr className="text-left text-app-muted">
              <th className="py-1 font-medium">Target</th>
              <th className="py-1 text-right font-medium">RTT</th>
              <th className="py-1 text-right font-medium">Jitter</th>
              <th className="py-1 text-right font-medium">Loss</th>
            </tr>
          </thead>
          <tbody className="font-mono text-app-text">
            {Object.entries(liveHealth).map(([label, stats]) => (
              <tr key={label}>
                <td className="py-0.5">{label}</td>
                <td className="py-0.5 text-right">
                  {stats.reachable ? `${stats.last_rtt_ms.toFixed(1)} ms` : "—"}
                </td>
                <td className="py-0.5 text-right">{stats.jitter_ms.toFixed(1)} ms</td>
                <td
                  className={`py-0.5 text-right ${
                    stats.loss_rate > 0 ? "text-red-400" : ""
                  }`}
                >
                  {(stats.loss_rate * 100).toFixed(0)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export default NetworkSourcePanel;
