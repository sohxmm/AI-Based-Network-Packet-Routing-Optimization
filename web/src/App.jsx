import { useEffect, useMemo, useState } from "react";

import BenchmarkReport from "./components/BenchmarkReport.jsx";
import { BenchmarkResultView } from "./components/BenchmarkResultView.jsx";
import CongestionHeatmap from "./components/CongestionHeatmap.jsx";
import ControlPanel from "./components/ControlPanel.jsx";
import ExperimentBuilder from "./components/ExperimentBuilder.jsx";
import FailoverPanel from "./components/FailoverPanel.jsx";
import LeftPanel from "./components/LeftPanel.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import ModelStatusBanner from "./components/ModelStatusBanner.jsx";
import NetworkSourcePanel from "./components/NetworkSourcePanel.jsx";
import PathDivergenceView from "./components/PathDivergenceView.jsx";
import RightPanel from "./components/RightPanel.jsx";
import TopologyGraph from "./components/TopologyGraph.jsx";
import { API_BASE_URL } from "./config.js";
import { useNetworkStream } from "./hooks/useNetworkStream.js";
import { useRouteRequest } from "./hooks/useRouteRequest.js";

/**
 * Four themes, not ten.
 *
 * The ten-theme engine worked and demonstrated real CSS architecture, but ten
 * themes cost about the same effort as fixing the model-loading bug did. Four
 * keep the engine and the light/dark handling honest without reading as
 * padding. `isDark` drives chart palettes, which are validated separately for
 * each mode rather than flipped automatically.
 */
const THEMES = [
  { id: "github-dark", label: "GitHub Dark", dark: true },
  { id: "nord", label: "Nord", dark: true },
  { id: "dracula", label: "Dracula", dark: true },
  { id: "solarized-light", label: "Solarized Light", dark: false },
];

const ALL_THEME_CLASSES = [
  "theme-solarized-light",
  "theme-solarized-dark",
  "theme-dracula",
  "theme-monokai",
  "theme-nord",
  "theme-tokyo-night",
  "theme-one-dark",
  "theme-gruvbox",
  "theme-synthwave",
  "theme-github-dark",
];

const TABS = [
  { id: "live", label: "Live network" },
  { id: "divergence", label: "Path divergence" },
  { id: "benchmark", label: "Benchmark" },
  { id: "lab", label: "Lab" },
];

function App() {
  const {
    networkState,
    failoverEvents,
    isConnected,
    isStale,
    error: streamError,
  } = useNetworkStream();

  const {
    compareRoutes,
    postSimulatorAction,
    setNetworkSource,
    watchFlow,
    runConvergenceTest,
    isLoading,
    error: actionError,
  } = useRouteRequest();

  const [comparison, setComparison] = useState(null);
  const [experimentResults, setExperimentResults] = useState(null);
  const [trafficClass, setTrafficClass] = useState("best_effort");
  const [trafficClasses, setTrafficClasses] = useState([]);
  const [theme, setTheme] = useState("github-dark");
  const [tab, setTab] = useState("live");

  const isDark = THEMES.find((entry) => entry.id === theme)?.dark ?? true;

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove(...ALL_THEME_CLASSES);
    root.classList.add(`theme-${theme}`);
  }, [theme]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/network/algorithms`)
      .then((response) => response.json())
      .then((data) => setTrafficClasses(data.traffic_classes ?? []))
      .catch(() => undefined);
  }, []);

  // The single best route, overlaid on the live topology.
  const bestPath = useMemo(() => {
    const successful = comparison?.results?.filter((result) => result.success) ?? [];
    const best = successful.reduce(
      (winner, result) =>
        !winner || result.total_latency < winner.total_latency ? result : winner,
      null
    );
    return best ? [{ algorithm: best.algorithm, path: best.path, is_fallback: best.is_fallback }] : [];
  }, [comparison]);

  async function handleCompare(source, destination) {
    const result = await compareRoutes({ source, destination, trafficClass });
    if (result) setComparison(result);
  }

  return (
    <main className="min-h-screen transition-colors duration-300">
      <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col gap-3 px-4 py-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-app-border pb-3">
          <div>
            <h1 className="text-lg font-semibold text-app-text">
              AI-Based Network Packet Routing Optimization
            </h1>
            <p className="text-xs text-app-muted">
              Step {networkState?.step_count ?? "--"} ·{" "}
              {networkState?.links?.length ?? "--"} links ·{" "}
              {networkState?.nodes?.length ?? "--"} nodes
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <ConnectionPill isConnected={isConnected} isStale={isStale} />
            <label className="flex h-8 items-center gap-2 rounded border border-app-border bg-app-input-bg px-2 text-xs">
              <span className="text-app-muted">Theme</span>
              <select
                value={theme}
                onChange={(event) => setTheme(event.target.value)}
                className="cursor-pointer bg-transparent text-app-text outline-none"
                aria-label="Select theme"
              >
                {THEMES.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </header>

        <ModelStatusBanner />

        {(streamError || actionError) && (
          <div
            role="alert"
            className="rounded border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-sm text-amber-400"
          >
            {streamError || actionError}
          </div>
        )}

        {isStale && isConnected && (
          <div
            role="alert"
            className="rounded border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-sm text-amber-400"
          >
            Connected, but no update for over 5 seconds. The backend&apos;s simulator
            loop may have stopped — check <code className="font-mono">/health</code>.
          </div>
        )}

        <nav className="flex gap-1" role="tablist" aria-label="Dashboard section">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              aria-selected={tab === entry.id}
              onClick={() => setTab(entry.id)}
              className={`rounded-t border-b-2 px-3 py-1.5 text-sm transition-colors ${
                tab === entry.id
                  ? "border-app-accent text-app-text"
                  : "border-transparent text-app-muted hover:text-app-text"
              }`}
            >
              {entry.label}
            </button>
          ))}
        </nav>

        {tab === "live" && (
          <div className="grid flex-1 gap-3 lg:grid-cols-[300px_minmax(0,1fr)_320px]">
            <div className="flex flex-col gap-3">
              <NetworkSourcePanel
                onSourceChange={setNetworkSource}
                isLoading={isLoading}
              />
              <LeftPanel networkState={networkState} />
            </div>

            <div className="flex flex-col gap-3">
              <div className="min-h-[480px] flex-1">
                <TopologyGraph
                  networkState={networkState}
                  highlightedPaths={bestPath}
                  isDark={isDark}
                />
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <MetricsPanel networkState={networkState} comparison={comparison} />
                <CongestionHeatmap networkState={networkState} isDark={isDark} />
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <ControlPanel
                networkState={networkState}
                isLoading={isLoading}
                onSimulatorAction={postSimulatorAction}
              />
              <FailoverPanel
                networkState={networkState}
                failoverEvents={failoverEvents}
                onWatch={watchFlow}
                onConvergence={runConvergenceTest}
                isLoading={isLoading}
              />
              <RightPanel networkState={networkState} />
            </div>
          </div>
        )}

        {tab === "divergence" && (
          <PathDivergenceView
            networkState={networkState}
            comparison={comparison}
            trafficClass={trafficClass}
            onTrafficClassChange={setTrafficClass}
            trafficClasses={trafficClasses}
            isLoading={isLoading}
            onCompare={handleCompare}
            isDark={isDark}
          />
        )}

        {tab === "benchmark" && (
          <section className="rounded border border-app-border bg-app-panel p-4">
            <BenchmarkReport isDark={isDark} />
          </section>
        )}

        {tab === "lab" && (
          <div className="grid gap-3 lg:grid-cols-[340px_minmax(0,1fr)]">
            <ExperimentBuilder onResults={setExperimentResults} />
            <section className="rounded border border-app-border bg-app-panel p-4">
              {experimentResults ? (
                <BenchmarkResultView
                  scenarioData={experimentResults}
                  scenarioLabel="Custom experiment"
                  showLimitations={false}
                  isDark={isDark}
                />
              ) : (
                <p className="text-sm text-app-muted">
                  Configure an experiment on the left and run it. Results use the
                  same engine, the same guardrails and the same statistics as the
                  fixed benchmark scenarios.
                </p>
              )}
            </section>
          </div>
        )}
      </div>
    </main>
  );
}

function ConnectionPill({ isConnected, isStale }) {
  const tone = !isConnected
    ? "bg-amber-400/20 text-amber-400 border-amber-400/40"
    : isStale
      ? "bg-amber-400/20 text-amber-400 border-amber-400/40"
      : "bg-emerald-400/20 text-emerald-400 border-emerald-400/40";

  const label = !isConnected
    ? "Waiting for backend"
    : isStale
      ? "Connected but stale"
      : "Live stream connected";

  return (
    <span className={`rounded border px-2 py-1 text-xs font-medium ${tone}`}>
      {label}
    </span>
  );
}

export default App;
