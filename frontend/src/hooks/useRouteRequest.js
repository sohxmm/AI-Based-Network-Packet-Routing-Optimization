import { useCallback, useState } from "react";
import { API_BASE_URL } from "../config.js";
export function useRouteRequest() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const compareRoutes = useCallback(async ({ source, destination, algorithms, use_forecast } = {}) => {
    if (!source || !destination || source === destination) {
      setError("Choose two different routers.");
      return null;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/network/route/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, destination, algorithms, use_forecast: !!use_forecast })
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail?.message || payload.detail || "Route comparison failed.");
      }

      return await response.json();
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const postSimulatorAction = useCallback(async (path, payload) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload ? JSON.stringify(payload) : undefined
      });

      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || "Simulator action failed.");
      }

      return await response.json();
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { compareRoutes, postSimulatorAction, isLoading, error };
}
