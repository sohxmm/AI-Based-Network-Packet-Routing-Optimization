import { useMemo, useState } from "react";

import ControlPanel from "./components/ControlPanel.jsx";
import CongestionHeatmap from "./components/CongestionHeatmap.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import RouteComparison from "./components/RouteComparison.jsx";
import TopologyGraph from "./components/TopologyGraph.jsx";
import { useNetworkStream } from "./hooks/useNetworkStream.js";
import { useRouteRequest } from "./hooks/useRouteRequest.js";

function App() {
  const { networkState, isConnected, error: streamError } = useNetworkStream();
  const { compareRoutes, postSimulatorAction, isLoading, error: actionError } = useRouteRequest();
  const [comparison, setComparison] = useState(null);

  const highlightedPath = useMemo(() => {
    const successful = comparison?.results?.filter((result) => result.success) ?? [];
    return successful.reduce((best, result) => {
      if (!best) {
        return result;
      }
      return result.total_latency < best.total_latency ? result : best;
    }, null)?.path ?? [];
  }, [comparison]);

  async function handleCompareRoutes(source, destination) {
    const result = await compareRoutes({ source, destination });
    if (result) {
      setComparison(result);
    }
  }

  return (
    <main className="min-h-screen bg-[#0b1220] text-slate-100">
      <section className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <h1 className="text-xl font-semibold">
              AI-Based Network Packet Routing Optimization
            </h1>
            <p className="text-sm text-slate-400">
              Step {networkState?.step_count ?? "--"} | {networkState?.links?.length ?? "--"} active links
            </p>
          </div>
          <span
            className={`rounded px-2 py-1 text-xs font-medium ${
              isConnected ? "bg-emerald-400 text-slate-950" : "bg-amber-300 text-slate-950"
            }`}
          >
            {isConnected ? "Live stream connected" : "Waiting for backend"}
          </span>
        </header>

        {(streamError || actionError) && (
          <div className="rounded border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-sm text-amber-100">
            {streamError || actionError}
          </div>
        )}

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.7fr)_minmax(320px,1fr)]">
          <TopologyGraph networkState={networkState} highlightedPath={highlightedPath} />
          <div className="flex flex-col gap-4">
            <MetricsPanel networkState={networkState} comparison={comparison} />
            <RouteComparison
              networkState={networkState}
              comparison={comparison}
              isLoading={isLoading}
              onCompare={handleCompareRoutes}
            />
          </div>
        </div>

        <CongestionHeatmap networkState={networkState} />
        <ControlPanel
          networkState={networkState}
          isLoading={isLoading}
          onSimulatorAction={postSimulatorAction}
        />
      </section>
    </main>
  );
}

export default App;
