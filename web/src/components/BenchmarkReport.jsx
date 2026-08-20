import { useEffect, useMemo, useState } from "react";

import ScenarioSelector from "./benchmark/ScenarioSelector.jsx";
import { BenchmarkResultView } from "./BenchmarkResultView.jsx";
import { API_BASE_URL } from "../config.js";

export { BenchmarkResultView };

/**
 * Fetching and scenario selection. Presentation lives in BenchmarkResultView
 * and ./benchmark/.
 */
function BenchmarkReport({ isDark = true }) {
  const [payload, setPayload] = useState(null);
  const [active, setActive] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE_URL}/benchmark/results`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setPayload(data);
        const names = Object.keys(data.scenarios ?? {});
        setActive((current) => current ?? names[0] ?? null);
      })
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setIsLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const scenarios = useMemo(
    () => Object.keys(payload?.scenarios ?? {}),
    [payload]
  );

  if (isLoading) {
    return <p className="text-sm text-app-muted">Loading benchmark results…</p>;
  }

  if (error) {
    return (
      <p className="text-sm text-amber-400">
        Could not load benchmark results ({error}). Generate them with{" "}
        <code className="font-mono">make bench</code>.
      </p>
    );
  }

  if (!scenarios.length) {
    return (
      <div className="rounded border border-app-border bg-app-input-bg p-4 text-sm text-app-muted">
        <p className="font-medium text-app-text">No benchmark results found.</p>
        <p className="mt-1">
          Run <code className="font-mono">make bench</code> to generate them. Results
          are written to <code className="font-mono">experiments/results/</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <ScenarioSelector scenarios={scenarios} active={active} onSelect={setActive} />
      <BenchmarkResultView
        scenarioData={payload.scenarios[active]}
        limitations={payload.known_limitations}
        isDark={isDark}
      />
    </div>
  );
}

export default BenchmarkReport;
