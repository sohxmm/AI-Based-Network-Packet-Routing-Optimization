// TODO: implement
import { GitCompareArrows } from "lucide-react";

function RouteComparison() {
  return (
    <section className="rounded border border-slate-800 bg-slate-900 p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-200">Route Comparison</h2>
        <button className="inline-flex h-9 items-center gap-2 rounded bg-cyan-400 px-3 text-sm font-medium text-slate-950">
          <GitCompareArrows className="h-4 w-4" aria-hidden="true" />
          Compare
        </button>
      </div>
      <div className="mt-4 rounded border border-dashed border-slate-700 p-4 text-sm text-slate-500">
        Algorithm result table placeholder
      </div>
    </section>
  );
}

export default RouteComparison;
