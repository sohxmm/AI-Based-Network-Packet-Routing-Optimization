import { useCallback, useEffect, useState } from "react";

import { API_BASE_URL } from "../config.js";

/**
 * Poll which model artifacts are present and which actually loaded.
 *
 * This exists because the project's worst defect was three of four AI features
 * silently serving heuristics after a filename mismatch. The dashboard should
 * be able to say "the RL model is not loaded, so this row is not AI" without
 * anyone having to read a log file.
 */
export function useModelHealth(pollMs = 30000) {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health/models`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setHealth(await response.json());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, pollMs);
    return () => window.clearInterval(timer);
  }, [refresh, pollMs]);

  const missing = health
    ? Object.entries(health.models)
        .filter(([, model]) => model.expected_in_repo && !model.file_present)
        .map(([key]) => key)
    : [];

  const notLoaded = health
    ? Object.entries(health.models)
        .filter(([, model]) => model.file_present && model.loaded_in_memory === false)
        .map(([key]) => key)
    : [];

  return { health, missing, notLoaded, error, refresh };
}
