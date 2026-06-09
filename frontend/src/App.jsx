// TODO: implement
import ControlPanel from "./components/ControlPanel.jsx";
import CongestionHeatmap from "./components/CongestionHeatmap.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import RouteComparison from "./components/RouteComparison.jsx";
import TopologyGraph from "./components/TopologyGraph.jsx";

function App() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <section className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <h1 className="text-xl font-semibold">
              AI-Based Network Packet Routing Optimization
            </h1>
            <p className="text-sm text-slate-400">Phase 0 scaffold</p>
          </div>
          <span className="rounded bg-emerald-500 px-2 py-1 text-xs font-medium text-slate-950">
            Ready for Phase 1
          </span>
        </header>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.7fr)_minmax(320px,1fr)]">
          <TopologyGraph />
          <div className="flex flex-col gap-4">
            <MetricsPanel />
            <RouteComparison />
          </div>
        </div>

        <CongestionHeatmap />
        <ControlPanel />
      </section>
    </main>
  );
}

export default App;
