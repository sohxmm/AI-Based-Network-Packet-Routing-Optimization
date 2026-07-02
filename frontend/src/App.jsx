import { useMemo, useState, useEffect } from "react";
import { Sun, Moon } from "lucide-react";

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
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDark]);

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
    <main className="min-h-screen bg-slate-50 text-slate-900 transition-colors duration-300 dark:bg-[#0b1220] dark:text-slate-100">
      <section className="mx-auto flex min-h-screen max-w-[1600px] flex-col gap-4 px-4 py-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3 dark:border-slate-800">
          <div>
            <h1 className="text-xl font-semibold">
              AI-Based Network Packet Routing Optimization
            </h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Step {networkState?.step_count ?? "--"} | {networkState?.links?.length ?? "--"} active links
            </p>
          </div>
          <div className="flex items-center gap-4">
            <span
              className={`rounded px-2 py-1 text-xs font-medium ${
                isConnected ? "bg-emerald-400 text-emerald-950 dark:bg-emerald-400 dark:text-slate-950" : "bg-amber-300 text-amber-950 dark:bg-amber-300 dark:text-slate-950"
              }`}
            >
              {isConnected ? "Live stream connected" : "Waiting for backend"}
            </span>
            <button
              onClick={() => setIsDark(!isDark)}
              className="rounded-full p-2 text-slate-600 hover:bg-slate-200 dark:text-slate-300 dark:hover:bg-slate-800"
              aria-label="Toggle theme"
            >
              {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
          </div>
        </header>

        {(streamError || actionError) && (
          <div className="rounded border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-sm text-amber-900 dark:text-amber-100">
            {streamError || actionError}
          </div>
        )}

        <div className="grid flex-1 gap-4 lg:grid-cols-[250px_minmax(0,1fr)_320px]">
          {/* Left Column */}
          <div className="flex flex-col gap-4">
            <MetricsPanel networkState={networkState} comparison={comparison} />
          </div>

          {/* Center Column */}
          <div className="flex flex-col gap-4">
            <TopologyGraph networkState={networkState} highlightedPath={highlightedPath} isDark={isDark} />
            <CongestionHeatmap networkState={networkState} isDark={isDark} />
          </div>

          {/* Right Column */}
          <div className="flex flex-col gap-4">
            <RouteComparison
              networkState={networkState}
              comparison={comparison}
              isLoading={isLoading}
              onCompare={handleCompareRoutes}
            />
            <ControlPanel
              networkState={networkState}
              isLoading={isLoading}
              onSimulatorAction={postSimulatorAction}
            />
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
