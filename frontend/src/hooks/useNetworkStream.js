// TODO: implement
import { useEffect, useState } from "react";

export function useNetworkStream(url = "ws://localhost:8000/ws/stream") {
  const [networkState, setNetworkState] = useState(null);
  const [lastRoutingEvent, setLastRoutingEvent] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    void url;
    setNetworkState(null);
    setLastRoutingEvent(null);
    setIsConnected(false);
    setError(null);
  }, [url]);

  return { networkState, lastRoutingEvent, isConnected, error };
}
