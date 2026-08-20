/**
 * ExperimentBuilder — "Design an Experiment" view.
 *
 * Lets users configure and run custom benchmark scenarios against all 8
 * algorithms using the same verified Phase 3 benchmark engine. Results
 * render using the shared BenchmarkResultView component from BenchmarkReport.
 */

import { useState, useEffect, useRef } from "react";

import { API_BASE_URL as API_BASE } from "../config.js";

const ALL_ALGORITHMS = [
  { id: "dijkstra", label: "Dijkstra" },
  { id: "bellman_ford", label: "Bellman-Ford" },
  { id: "constrained", label: "Constrained k-shortest" },
  { id: "aco", label: "Ant Colony" },
  { id: "gnn", label: "GNN" },
  { id: "rl", label: "RL (PPO)" },
  { id: "multi_agent", label: "Multi-Agent RL" },
  { id: "random_baseline", label: "Random baseline" },
];

const TRAFFIC_CLASSES = [
  { id: "best_effort", label: "Best effort" },
  { id: "emergency", label: "Emergency" },
  { id: "interactive", label: "Voice / video" },
  { id: "gaming", label: "Gaming" },
  { id: "bulk", label: "Bulk transfer" },
];

const MAX_TOTAL_DECISIONS = 3000;

// Observed throughput: ~8000 decisions in ~2.5 min = 3200 decisions/min
const DECISIONS_PER_MINUTE = 3200;

export default function ExperimentBuilder({ onResults }) {
  // ── Config form state ─────────────────────────────────────────────────
  const [topologySize, setTopologySize] = useState(25);
  const [congestionProfile, setCongestionProfile] = useState("normal");
  const [failureRate, setFailureRate] = useState(0);
  const [failurePattern, setFailurePattern] = useState("none");
  const [steps, setSteps] = useState(50);
  const [pairsPerStep, setPairsPerStep] = useState(5);
  const [runs, setRuns] = useState(3);
  const [trafficClasses, setTrafficClasses] = useState(["best_effort"]);
  const [selectedAlgos, setSelectedAlgos] = useState(
    ALL_ALGORITHMS.map((a) => a.id)
  );

  // ── Job state ─────────────────────────────────────────────────────────
  const [jobState, setJobState] = useState(null); // queued | running | done | failed
  const [progress, setProgress] = useState({ runs_completed: 0, total: 1 });
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const pollRef = useRef(null);

  // ── Derived values ────────────────────────────────────────────────────
  const totalDecisions = steps * pairsPerStep * runs;
  const exceedsCap = totalDecisions > MAX_TOTAL_DECISIONS;
  const totalDecisionsWithAlgos = totalDecisions * selectedAlgos.length;
  const estimatedMinutes = totalDecisionsWithAlgos / DECISIONS_PER_MINUTE;
  const estimatedTimeStr =
    estimatedMinutes < 1
      ? `~${Math.max(1, Math.round(estimatedMinutes * 60))}s`
      : `~${estimatedMinutes.toFixed(1)} min`;

  // ── Algorithm checkbox toggle ─────────────────────────────────────────
  function toggleAlgorithm(algoId) {
    setSelectedAlgos((prev) =>
      prev.includes(algoId)
        ? prev.filter((a) => a !== algoId)
        : [...prev, algoId]
    );
  }

  // ── Submit experiment ─────────────────────────────────────────────────
  async function handleSubmit(e) {
    e.preventDefault();
    if (exceedsCap || selectedAlgos.length === 0) return;

    setSubmitting(true);
    setError(null);
    setJobState(null);

    try {
      const res = await fetch(`${API_BASE}/experiments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topology_size: topologySize,
          congestion_profile: congestionProfile,
          failure_rate: failureRate,
          failure_pattern: failurePattern,
          steps,
          pairs_per_step: pairsPerStep,
          runs,
          algorithms: selectedAlgos,
          traffic_classes: trafficClasses,
        }),
      });

      if (!res.ok) {
        const body = await res.json();
        const detail = body.detail;
        // Handle pydantic validation errors (array of objects)
        if (Array.isArray(detail)) {
          throw new Error(detail.map((d) => d.msg).join("; "));
        }
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }

      const data = await res.json();
      setJobState("queued");
      startPolling(data.job_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  // ── Poll job status ───────────────────────────────────────────────────
  function startPolling(id) {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/experiments/${id}/status`);
        if (!res.ok) return;
        const data = await res.json();
        setJobState(data.state);
        setProgress(data.progress || { runs_completed: 0, total: 1 });

        if (data.state === "done") {
          clearInterval(pollRef.current);
          pollRef.current = null;
          // Fetch results
          const rRes = await fetch(`${API_BASE}/experiments/${id}/results`);
          if (rRes.ok) {
            const resultData = await rRes.json();
            if (onResults) {
              onResults(resultData);
            }
          }
        } else if (data.state === "failed") {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setError(data.error || "Experiment failed");
        }
      } catch {
        // Network error, keep polling
      }
    }, 2000);
  }

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const progressPct =
    progress.total > 0
      ? Math.round(((progress.runs_completed ?? 0) / (progress.total || 1)) * 100)
      : 0;

  return (
    <div className="flex flex-col gap-4 rounded border border-app-border bg-app-panel p-4 h-full">
      <div>
        <h2 className="text-sm font-semibold text-app-text flex items-center gap-2">
          <span>🧪</span> Experiment Builder
        </h2>
        <p className="mt-1 text-xs text-app-muted leading-relaxed">
          Configure a custom scenario to evaluate algorithms in the background.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3 flex-1 overflow-y-auto pr-1">
        
        {/* Step 1: Topology */}
        <fieldset className="rounded border border-app-border bg-app-input-bg p-3">
          <legend className="px-1 text-[11px] font-semibold text-app-muted uppercase tracking-wider">
            1. Topology Size
          </legend>
          <div className="flex gap-2 mt-1">
            {[25, 50, 100].map((size) => (
              <label
                key={size}
                className={`flex-1 text-center cursor-pointer rounded px-2 py-1 text-xs transition-colors ${
                  topologySize === size
                    ? "bg-app-accent text-app-accent-text font-medium"
                    : "bg-app-panel text-app-text hover:bg-app-border"
                }`}
              >
                <input
                  type="radio"
                  name="topology_size"
                  value={size}
                  checked={topologySize === size}
                  onChange={() => setTopologySize(size)}
                  className="sr-only"
                />
                {size} nodes
              </label>
            ))}
          </div>
        </fieldset>

        {/* Step 2: Traffic */}
        <fieldset className="rounded border border-app-border bg-app-input-bg p-3">
          <legend className="px-1 text-[11px] font-semibold text-app-muted uppercase tracking-wider">
            2. Traffic Profile
          </legend>
          <select
            className="mt-1 h-8 w-full rounded border border-app-border bg-app-panel px-2 text-xs text-app-text outline-none focus:border-app-accent"
            value={congestionProfile}
            onChange={(e) => setCongestionProfile(e.target.value)}
          >
            <option value="normal">Normal</option>
            <option value="high">High (30% congested)</option>
            <option value="bursty">Bursty (random spikes)</option>
          </select>
        </fieldset>

        {/* Step 3: Failures */}
        <fieldset className="rounded border border-app-border bg-app-input-bg p-3">
          <legend className="px-1 text-[11px] font-semibold text-app-muted uppercase tracking-wider">
            3. Link Failures
          </legend>
          <div className="flex flex-col gap-2 mt-1">
            <label className="text-xs text-app-muted flex items-center justify-between">
              <span>Rate</span>
              <span className="font-mono text-app-text">{failureRate}%</span>
            </label>
            <input
              type="range"
              min={0}
              max={30}
              step={1}
              value={failureRate}
              onChange={(e) => setFailureRate(Number(e.target.value))}
              className="w-full accent-app-accent"
            />
            <label className="text-xs text-app-muted mt-1">
              Pattern
              <select
                className="mt-1 h-8 w-full rounded border border-app-border bg-app-panel px-2 text-xs text-app-text outline-none focus:border-app-accent"
                value={failurePattern}
                onChange={(e) => setFailurePattern(e.target.value)}
              >
                <option value="none">None</option>
                <option value="random">Random</option>
                <option value="targeted">Targeted (Hubs first)</option>
              </select>
            </label>
          </div>
        </fieldset>

        {/* Step 4: Scale */}
        <fieldset className="rounded border border-app-border bg-app-input-bg p-3">
          <legend className="px-1 text-[11px] font-semibold text-app-muted uppercase tracking-wider">
            4. Simulation Scale
          </legend>
          <div className="grid grid-cols-3 gap-2 mt-1">
            <label className="text-[10px] text-app-muted">
              Steps (≤300)
              <input
                type="number"
                min={1}
                max={300}
                value={steps}
                onChange={(e) =>
                  setSteps(Math.max(1, Math.min(300, Number(e.target.value) || 1)))
                }
                className="mt-1 h-7 w-full rounded border border-app-border bg-app-panel px-2 text-xs text-app-text outline-none focus:border-app-accent"
              />
            </label>
            <label className="text-[10px] text-app-muted">
              Pairs/step (≤10)
              <input
                type="number"
                min={1}
                max={10}
                value={pairsPerStep}
                onChange={(e) =>
                  setPairsPerStep(
                    Math.max(1, Math.min(10, Number(e.target.value) || 1))
                  )
                }
                className="mt-1 h-7 w-full rounded border border-app-border bg-app-panel px-2 text-xs text-app-text outline-none focus:border-app-accent"
              />
            </label>
            <label className="text-[10px] text-app-muted" title="Independent seeded replications. Statistics are computed across runs, not across the correlated decisions inside one run.">
              Runs (≤10)
              <input
                type="number"
                min={1}
                max={10}
                value={runs}
                onChange={(e) =>
                  setRuns(Math.max(1, Math.min(10, Number(e.target.value) || 1)))
                }
                className="mt-1 h-7 w-full rounded border border-app-border bg-app-panel px-2 text-xs text-app-text outline-none focus:border-app-accent"
              />
            </label>
          </div>

          <fieldset className="mt-2 border-0 p-0">
            <legend className="text-[10px] text-app-muted">Traffic classes</legend>
            <div className="mt-1 flex flex-wrap gap-1">
              {TRAFFIC_CLASSES.map((cls) => (
                <button
                  key={cls.id}
                  type="button"
                  aria-pressed={trafficClasses.includes(cls.id)}
                  onClick={() =>
                    setTrafficClasses((current) =>
                      current.includes(cls.id)
                        ? current.filter((item) => item !== cls.id) || []
                        : [...current, cls.id]
                    )
                  }
                  className={`rounded-full border px-2 py-0.5 text-[10px] ${
                    trafficClasses.includes(cls.id)
                      ? "border-app-accent bg-app-accent/15 text-app-text"
                      : "border-app-border text-app-muted"
                  }`}
                >
                  {cls.label}
                </button>
              ))}
            </div>
          </fieldset>

          <div
            className={`mt-2 text-[10px] ${
              exceedsCap ? "text-red-400 font-semibold" : "text-app-muted"
            }`}
          >
            Total decisions: {totalDecisions.toLocaleString()}
            {exceedsCap && " (Max 3,000)"}
          </div>
        </fieldset>

        {/* Step 5: Algorithms */}
        <fieldset className="rounded border border-app-border bg-app-input-bg p-3">
          <legend className="px-1 text-[11px] font-semibold text-app-muted uppercase tracking-wider">
            5. Algorithms
          </legend>
          <div className="flex flex-wrap gap-1.5 mt-1">
            {ALL_ALGORITHMS.map((a) => (
              <label
                key={a.id}
                className={`flex cursor-pointer items-center gap-1 rounded px-2 py-1 text-[10px] transition-colors select-none ${
                  selectedAlgos.includes(a.id)
                    ? "bg-app-accent/20 text-app-accent border border-app-accent/40"
                    : "bg-app-panel text-app-muted border border-app-border hover:border-app-accent/30"
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedAlgos.includes(a.id)}
                  onChange={() => toggleAlgorithm(a.id)}
                  className="sr-only"
                />
                {a.label}
              </label>
            ))}
          </div>
        </fieldset>

        {/* Submit */}
        <div className="mt-auto pt-2 border-t border-app-border">
          {error && (
            <div className="mb-2 rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-[11px] text-red-400">
              {error}
            </div>
          )}

          {(jobState === "queued" || jobState === "running") ? (
            <div className="rounded border border-app-border bg-app-input-bg p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-app-text">
                  {jobState === "queued" ? "Queued…" : "Running…"}
                </span>
                <span className="text-[10px] text-app-muted">
                  {progressPct}%
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-app-panel">
                <div
                  className="h-full rounded-full bg-app-accent transition-all duration-500 ease-out"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          ) : (
            <button
              type="submit"
              disabled={
                exceedsCap ||
                selectedAlgos.length === 0 ||
                submitting
              }
              className="w-full h-8 flex items-center justify-center gap-2 rounded bg-app-accent text-xs font-semibold text-app-accent-text transition-opacity disabled:cursor-not-allowed disabled:opacity-50 hover:opacity-90"
            >
              {submitting ? "Submitting…" : "▶ Run Experiment"}
            </button>
          )}
          
          <div className="mt-2 text-center text-[10px] text-app-muted">
            {estimatedTimeStr} • {totalDecisionsWithAlgos.toLocaleString()} evaluations
          </div>
        </div>
      </form>
    </div>
  );
}
