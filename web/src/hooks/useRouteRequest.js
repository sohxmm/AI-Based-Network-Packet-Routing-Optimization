import { useCallback, useState } from "react";

import { API_BASE_URL } from "../config.js";
import { extractApiError } from "../utils/apiError.js";

/**
 * Every mutating call the dashboard makes, with uniform error extraction.
 * Both handlers previously parsed error bodies differently and one of them
 * rendered structured errors as "[object Object]".
 */
export function useRouteRequest() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const request = useCallback(async (path, { method = "GET", body } = {}) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}${path}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!response.ok) throw new Error(await extractApiError(response));
      return await response.json();
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const compareRoutes = useCallback(
    async ({ source, destination, algorithms, trafficClass, useForecast } = {}) => {
      if (!source || !destination || source === destination) {
        setError("Choose two different routers.");
        return null;
      }
      return request("/network/route/compare", {
        method: "POST",
        body: {
          source,
          destination,
          algorithms,
          traffic_class: trafficClass ?? "best_effort",
          use_forecast: Boolean(useForecast),
        },
      });
    },
    [request]
  );

  const fetchCandidates = useCallback(
    ({ source, destination, trafficClass = "best_effort" }) =>
      request(
        `/network/candidates?source=${encodeURIComponent(source)}` +
          `&destination=${encodeURIComponent(destination)}` +
          `&traffic_class=${encodeURIComponent(trafficClass)}`
      ),
    [request]
  );

  const postSimulatorAction = useCallback(
    (path, payload) => request(path, { method: "POST", body: payload }),
    [request]
  );

  const setNetworkSource = useCallback(
    (payload) => request("/sim/source", { method: "POST", body: payload }),
    [request]
  );

  const watchFlow = useCallback(
    (payload) => request("/network/failover/watch", { method: "POST", body: payload }),
    [request]
  );

  const runConvergenceTest = useCallback(
    (payload) =>
      request("/network/failover/convergence", { method: "POST", body: payload }),
    [request]
  );

  return {
    compareRoutes,
    fetchCandidates,
    postSimulatorAction,
    setNetworkSource,
    watchFlow,
    runConvergenceTest,
    request,
    isLoading,
    error,
  };
}
