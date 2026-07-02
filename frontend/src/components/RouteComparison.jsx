import { useMemo, useState } from "react";

import { GitCompareArrows } from "lucide-react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function RouteComparison({ networkState, comparison, isLoading, onCompare }) {
  const nodes = networkState?.nodes ?? [];
  const [source, setSource] = useState("R1");
  const [destination, setDestination] = useState("R2");
  const [expandedAlgorithm, setExpandedAlgorithm] = useState(null);

  const bestAlgorithm = useMemo(() => {
    const successful = comparison?.results?.filter((result) => result.success) ?? [];
    return successful.reduce((best, result) => {
      if (!best) {
        return result;
      }
      return result.total_latency < best.total_latency ? result : best;
    }, null)?.algorithm;
  }, [comparison]);
  const chartData = (comparison?.results ?? []).map((result) => ({
    algorithm: result.algorithm.replace("_", " "),
    latency: result.total_latency ?? 0
  }));

  function handleSubmit(event) {
    event.preventDefault();
    onCompare(source, destination);
  }

  function togglePath(algorithm) {
    setExpandedAlgorithm(expandedAlgorithm === algorithm ? null : algorithm);
  }

  return (
    <section className="rounded border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900/90">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Route Comparison</h2>

        <div className="grid grid-cols-2 gap-2">
          <label className="text-xs text-slate-500 dark:text-slate-400">
            Source
            <select
              className="mt-1 h-9 w-full rounded border border-slate-300 bg-slate-50 px-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              value={source}
              onChange={(event) => setSource(event.target.value)}
            >
              {nodes.map((node) => (
                <option key={node} value={node}>{node}</option>
              ))}
            </select>
          </label>
          <label className="text-xs text-slate-500 dark:text-slate-400">
            Destination
            <select
              className="mt-1 h-9 w-full rounded border border-slate-300 bg-slate-50 px-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              value={destination}
              onChange={(event) => setDestination(event.target.value)}
            >
              {nodes.map((node) => (
                <option key={node} value={node}>{node}</option>
              ))}
            </select>
          </label>
        </div>

        <button
          className="inline-flex h-9 items-center justify-center gap-2 rounded bg-cyan-600 px-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60 dark:bg-cyan-400 dark:text-slate-950"
          type="submit"
          disabled={isLoading || nodes.length < 2}
        >
          <GitCompareArrows className="h-4 w-4" aria-hidden="true" />
          {isLoading ? "Comparing" : "Compare All"}
        </button>
      </form>

      <div className="mt-4 overflow-hidden rounded border border-slate-200 dark:border-slate-800">
        <table className="w-full table-fixed text-left text-sm">
          <thead className="bg-slate-50 text-xs text-slate-500 dark:bg-slate-950 dark:text-slate-400">
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
                className={result.algorithm === bestAlgorithm ? "bg-emerald-50 dark:bg-emerald-400/10" : "border-t border-slate-200 dark:border-slate-800"}
              >
                <td className="px-3 py-2 capitalize">{result.algorithm.replace("_", " ")}</td>
                <td className="px-3 py-2">
                  {result.total_latency == null ? "--" : `${result.total_latency.toFixed(1)} ms`}
                </td>
                <td 
                  className={`px-3 py-2 cursor-pointer text-slate-600 transition-all dark:text-slate-300 ${expandedAlgorithm === result.algorithm ? "break-words whitespace-normal" : "truncate"}`}
                  onClick={() => togglePath(result.algorithm)}
                  title={expandedAlgorithm === result.algorithm ? "Click to collapse" : "Click to expand"}
                >
                  {result.path?.length ? result.path.join(" -> ") : "No path"}
                </td>
              </tr>
            ))}
            {!comparison?.results?.length && (
              <tr>
                <td className="px-3 py-5 text-slate-500" colSpan={3}>
                  Choose routers and compare all algorithms.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {chartData.length > 0 && (
        <div className="mt-4 h-44 rounded border border-slate-200 bg-slate-50 p-2 dark:border-slate-800 dark:bg-slate-950">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <XAxis dataKey="algorithm" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "var(--tw-colors-slate-900, #0f172a)", border: "1px solid var(--tw-colors-slate-700, #334155)" }}
                labelStyle={{ color: "var(--tw-colors-slate-200, #e2e8f0)" }}
              />
              <Bar dataKey="latency" name="Latency ms" fill="#22d3ee" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

export default RouteComparison;
