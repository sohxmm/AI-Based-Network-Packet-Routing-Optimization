import { useMemo, useState, useEffect } from "react";
import { Settings } from "lucide-react";

import ControlPanel from "./components/ControlPanel.jsx";
import CongestionHeatmap from "./components/CongestionHeatmap.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import RouteComparison from "./components/RouteComparison.jsx";
import TopologyGraph from "./components/TopologyGraph.jsx";
import LeftPanel from "./components/LeftPanel.jsx";
import RightPanel from "./components/RightPanel.jsx";
import { useNetworkStream } from "./hooks/useNetworkStream.js";
import { useRouteRequest } from "./hooks/useRouteRequest.js";

function App() {
  const { networkState, isConnected, error: streamError } = useNetworkStream();
  const { compareRoutes, postSimulatorAction, isLoading, error: actionError } = useRouteRequest();
  const [comparison, setComparison] = useState(null);
  const [theme, setTheme] = useState("dracula");

  useEffect(() => {
    const root = document.documentElement;
    const allThemes = [
      "theme-solarized-light", "theme-solarized-dark", "theme-dracula",
      "theme-monokai", "theme-nord", "theme-tokyo-night", "theme-one-dark",
      "theme-gruvbox", "theme-synthwave", "theme-github-dark"
    ];
    root.classList.remove(...allThemes, "dark", "theme-maroon", "theme-green");
    root.classList.add(`theme-${theme}`);
  }, [theme]);

  const highlightedPath = useMemo(() => {
    const successful = comparison?.results?.filter((result) => result.success) ?? [];
    return successful.reduce((best, result) => {
      if (!best) {
        return result;
      }
      return result.total_latency < best.total_latency ? result : best;
    }, null)?.path ?? [];
  }, [comparison]);

  async function handleCompareRoutes(source, destination, useForecast) {
    const result = await compareRoutes({ source, destination, use_forecast: useForecast });
    if (result) {
      setComparison(result);
    }
  }

  return (
    <main className="min-h-screen transition-colors duration-300">
      <section className="mx-auto flex min-h-screen max-w-[1600px] flex-col gap-4 px-4 py-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-app-border pb-3">
          <div>
            <h1 className="text-xl font-semibold">
              AI-Based Network Packet Routing Optimization
            </h1>
            <p className="text-sm text-app-muted">
              Step {networkState?.step_count ?? "--"} | {networkState?.links?.length ?? "--"} active links
            </p>
          </div>
          <div className="flex items-center gap-4">
            <span
              className={`rounded px-2 py-1 text-xs font-medium ${
                isConnected ? "bg-emerald-400 text-emerald-950" : "bg-amber-300 text-amber-950"
              }`}
            >
              {isConnected ? "Live stream connected" : "Waiting for backend"}
            </span>
            <div className="flex items-center gap-2 text-sm rounded border border-app-border bg-app-input-bg px-2 h-8">
              <span className="text-app-muted">theme:</span>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value)}
                className="bg-transparent text-app-text outline-none cursor-pointer"
                aria-label="Select theme"
              >
                <option value="solarized-light">solarized-light</option>
                <option value="solarized-dark">solarized-dark</option>
                <option value="dracula">dracula</option>
                <option value="monokai">monokai</option>
                <option value="nord">nord</option>
                <option value="tokyo-night">tokyo-night</option>
                <option value="one-dark">one-dark</option>
                <option value="gruvbox">gruvbox</option>
                <option value="synthwave">synthwave</option>
                <option value="github-dark">github-dark</option>
              </select>
            </div>
          </div>
        </header>

        {(streamError || actionError) && (
          <div className="rounded border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-sm text-amber-900">
            {streamError || actionError}
          </div>
        )}

        <div className="grid flex-1 gap-4 lg:grid-cols-[250px_minmax(0,1fr)_320px]">
          {/* Left Column */}
          <div className="flex flex-col gap-4">
            <MetricsPanel networkState={networkState} comparison={comparison} />
            <LeftPanel networkState={networkState} />
          </div>

          {/* Center Column */}
          <div className="flex flex-col gap-4">
            <TopologyGraph networkState={networkState} highlightedPath={highlightedPath} isDark={theme !== 'solarized-light'} />
            <CongestionHeatmap networkState={networkState} isDark={theme !== 'solarized-light'} />
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
            <RightPanel networkState={networkState} />
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
