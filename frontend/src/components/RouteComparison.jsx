import { useMemo, useState } from "react";

import { GitCompareArrows } from "lucide-react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function RouteComparison({ networkState, comparison, isLoading, onCompare }) {
  const nodes = networkState?.nodes ?? [];
  const [source, setSource] = useState("R1");
  const [destination, setDestination] = useState("R2");
  const [expandedAlgorithm, setExpandedAlgorithm] = useState(null);
  const [useForecast, setUseForecast] = useState(false);

  const bestAlgorithm = useMemo(() => {
    const successful = comparison?.results?.filter((result) => result.success) ?? [];
    return successful.reduce((best, result) => {
      if (!best) {
        return result;
      }
      return result.total_latency < best.total_latency ? result : best;
    }, null)?.algorithm;
  }, [comparison]);

  const LABEL_MAP = {
    dijkstra: "Dijkstra",
    bellman_ford: "B-Ford",
    aco: "ACO",
    rl: "RL",
    gnn: "GNN",
    gnn_predictive: "GNN-P",
    rl_predictive: "RL-P",
  };

  const chartData = (comparison?.results ?? []).map((result) => {
    const rawName = result.algorithm.toLowerCase();
    const shortName = LABEL_MAP[rawName] || rawName;

    return {
      algorithm: shortName,
      latency: result.total_latency ?? 0
    };
  });

  function handleSubmit(event) {
    event.preventDefault();
    onCompare(source, destination, useForecast);
  }

  function togglePath(algorithm) {
    setExpandedAlgorithm(expandedAlgorithm === algorithm ? null : algorithm);
  }

  function formatAlgorithmName(name) {
    const displayName = LABEL_MAP[name.toLowerCase()] || name.replace("_", " ");
    const isPredictive = name.toLowerCase().includes("predictive");
    return (
      <span className="flex items-center gap-1.5">
        <span className="capitalize">{displayName}</span>
        {isPredictive && (
          <span className="rounded bg-blue-500/20 px-1 py-0.5 text-[10px] font-medium text-blue-400">
            forecast
          </span>
        )}
      </span>
    );
  }

  return (
    <section className="rounded border border-app-border bg-app-panel p-4">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold text-app-text">Route Comparison</h2>

        <div className="grid grid-cols-2 gap-2">
          <label className="text-xs text-app-muted">
            Source
            <select
              className="mt-1 h-9 w-full rounded border border-app-border bg-app-input-bg px-2 text-sm text-app-text outline-none focus:border-app-accent"
              value={source}
              onChange={(event) => setSource(event.target.value)}
            >
              {nodes.map((node) => (
                <option key={node} value={node}>{node}</option>
              ))}
            </select>
          </label>
          <label className="text-xs text-app-muted">
            Destination
            <select
              className="mt-1 h-9 w-full rounded border border-app-border bg-app-input-bg px-2 text-sm text-app-text outline-none focus:border-app-accent"
              value={destination}
              onChange={(event) => setDestination(event.target.value)}
            >
              {nodes.map((node) => (
                <option key={node} value={node}>{node}</option>
              ))}
            </select>
          </label>
        </div>

        {/* Forecast toggle */}
        <label
          id="forecast-toggle"
          className="flex cursor-pointer items-center gap-2 text-xs text-app-muted select-none"
        >
          <span
            role="switch"
            aria-checked={useForecast}
            tabIndex={0}
            className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors duration-200 ${
              useForecast ? "bg-app-accent" : "bg-app-border"
            }`}
            onClick={() => setUseForecast(!useForecast)}
            onKeyDown={(e) => {
              if (e.key === " " || e.key === "Enter") {
                e.preventDefault();
                setUseForecast(!useForecast);
              }
            }}
          >
            <span
              className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform duration-200 ${
                useForecast ? "translate-x-[18px]" : "translate-x-[3px]"
              }`}
            />
          </span>
          Use congestion forecast
        </label>

        <button
          className="inline-flex h-9 items-center justify-center gap-2 rounded bg-app-accent px-3 text-sm font-medium text-app-accent-text disabled:cursor-not-allowed disabled:opacity-60"
          type="submit"
          disabled={isLoading || nodes.length < 2}
        >
          <GitCompareArrows className="h-4 w-4" aria-hidden="true" />
          {isLoading ? "Comparing" : "Compare All"}
        </button>
      </form>

      <div className="mt-4 overflow-hidden rounded border border-app-border">
        <table className="w-full table-fixed text-left text-sm">
          <thead className="bg-app-input-bg text-xs text-app-muted border-b border-app-border">
            <tr>
              <th className="px-3 py-2 font-medium">Algorithm</th>
              <th className="px-3 py-2 font-medium">Latency</th>
              <th className="px-3 py-2 font-medium">Path</th>
            </tr>
          </thead>
          <tbody>
            {(comparison?.results ?? []).map((result) => (
              <tr
                key={result.algorithm}
                className={result.algorithm === bestAlgorithm ? "bg-app-accent/20" : "border-t border-app-border"}
              >
                <td className="px-3 py-2 text-app-text">{formatAlgorithmName(result.algorithm)}</td>
                <td className="px-3 py-2 text-app-text">
                  {result.total_latency == null ? "--" : `${result.total_latency.toFixed(1)} ms`}
                </td>
                <td 
                  className={`px-3 py-2 cursor-pointer text-app-muted transition-all hover:text-app-text ${expandedAlgorithm === result.algorithm ? "break-words whitespace-normal" : "truncate"}`}
                  onClick={() => togglePath(result.algorithm)}
                  title={expandedAlgorithm === result.algorithm ? "Click to collapse" : "Click to expand"}
                >
                  {result.path?.length ? result.path.join(" -> ") : "No path"}
                </td>
              </tr>
            ))}
            {!comparison?.results?.length && (
              <tr>
                <td className="px-3 py-5 text-app-muted" colSpan={3}>
                  Choose routers and compare all algorithms.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {chartData.length > 0 && (
        <div className="mt-4 h-44 rounded border border-app-border bg-app-input-bg p-2">
          <ResponsiveContainer width="100%" height="100%">
            {/* Margins balanced back out for horizontal labels */}
            <BarChart data={chartData} margin={{ top: 15, right: 5, left: -20, bottom: 5 }}>
              {/* FIXED: Straight labels at 0 angle with slightly smaller font size */}
              <XAxis 
                dataKey="algorithm" 
                stroke="currentColor" 
                className="text-app-muted" 
                tick={{ fontSize: 10, angle: 0, textAnchor: 'middle', dy: 4 }}
                interval={0}
              />
              <YAxis 
                stroke="currentColor" 
                className="text-app-muted" 
                tick={{ fontSize: 11 }} 
                domain={[0, 'dataMax + 2']} 
                tickFormatter={(value) => Math.round(value)}
              />
              <Tooltip
                contentStyle={{ background: "var(--color-panel)", border: "1px solid var(--color-border)" }}
                labelStyle={{ color: "var(--color-text-main)" }}
              />
              <Bar 
                dataKey="latency" 
                name="Latency ms" 
                fill="var(--color-accent, #3b82f6)" 
                radius={[4, 4, 0, 0]} 
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

export default RouteComparison;