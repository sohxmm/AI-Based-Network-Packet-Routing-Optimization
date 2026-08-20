import { useModelHealth } from "../hooks/useModelHealth.js";

/**
 * Say out loud when an "AI" row is not actually AI.
 *
 * The single worst defect in this project's history was three of four learned
 * routers silently serving heuristics because of a filename mismatch, with no
 * log line and nothing in the UI. This banner is the permanent fix for the
 * *visibility* half of that problem: if a model is missing or failed to load,
 * the dashboard says so above the fold, unprompted.
 */
function ModelStatusBanner() {
  const { health, missing, notLoaded } = useModelHealth();

  if (!health) return null;

  const problems = [...missing, ...notLoaded];
  if (problems.length === 0) {
    return (
      <div className="flex items-center gap-2 rounded border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-400">
        <span aria-hidden="true">✓</span>
        <span>
          All {Object.keys(health.models).length} model artifacts loaded — every
          AI result below is genuine model output.
        </span>
      </div>
    );
  }

  return (
    <div className="rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-400">
      <p className="font-semibold">
        <span aria-hidden="true">⚠ </span>
        {problems.length} model{problems.length > 1 ? "s are" : " is"} not loaded.
      </p>
      <p className="mt-1 text-amber-300/90">
        Results shown for {problems.join(", ")} are congestion-aware heuristics,
        not AI. Train them with <code className="font-mono">make train</code>.
      </p>
      <ul className="mt-1.5 space-y-0.5 font-mono text-[11px] text-amber-300/80">
        {problems.map((key) => (
          <li key={key}>
            {key}: {health.models[key].file_present ? "present but failed to load" : "artifact missing"}
            {" → "}
            {health.models[key].train_command}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ModelStatusBanner;
