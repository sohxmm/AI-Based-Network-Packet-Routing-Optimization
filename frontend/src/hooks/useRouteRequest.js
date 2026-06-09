// TODO: implement
import { useCallback, useState } from "react";

export function useRouteRequest() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const compareRoutes = useCallback(async () => {
    setIsLoading(false);
    setError(null);
    return null;
  }, []);

  return { compareRoutes, isLoading, error };
}
