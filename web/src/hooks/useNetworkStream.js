import { useEffect, useRef, useState } from "react";

import { wsUrl } from "../config.js";

/** Frames older than this mean the backend is connected but not producing. */
const STALE_AFTER_MS = 5000;

/**
 * Subscribe to the live network stream.
 *
 * Two fixes over the original:
 *
 * - `JSON.parse` is guarded. One malformed frame used to throw inside the
 *   message listener, which silently kills the handler for the rest of the
 *   connection's life.
 * - `isStale` is exposed. A backend whose simulator loop has died looks
 *   *identical* to an idle one from the client's point of view: the socket
 *   stays open and no frames arrive. Now the UI can say so.
 */
export function useNetworkStream(url = wsUrl("/ws/stream")) {
  const [networkState, setNetworkState] = useState(null);
  const [lastRoutingEvent, setLastRoutingEvent] = useState(null);
  const [lastComparison, setLastComparison] = useState(null);
  const [failoverEvents, setFailoverEvents] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isStale, setIsStale] = useState(false);
  const [error, setError] = useState(null);

  const lastFrameAtRef = useRef(0);

  useEffect(() => {
    let socket = null;
    let reconnectTimer = null;
    let reconnectDelay = 1000;
    let shouldReconnect = true;

    function connect() {
      socket = new WebSocket(url);

      socket.addEventListener("open", () => {
        reconnectDelay = 1000;
        setIsConnected(true);
        setError(null);
        lastFrameAtRef.current = Date.now();
      });

      socket.addEventListener("message", (event) => {
        let message;
        try {
          message = JSON.parse(event.data);
        } catch {
          console.warn("Dropped a malformed WebSocket frame");
          return;
        }

        lastFrameAtRef.current = Date.now();
        setIsStale(false);

        switch (message.type) {
          case "state_update":
            setNetworkState(message.payload);
            break;
          case "routing_event":
            setLastRoutingEvent(message.payload);
            break;
          case "routing_comparison":
            setLastComparison(message.payload);
            break;
          case "failover":
            setFailoverEvents((previous) =>
              [...message.payload, ...previous].slice(0, 20)
            );
            break;
          default:
            break;
        }
      });

      socket.addEventListener("close", () => {
        setIsConnected(false);
        if (!shouldReconnect) return;
        reconnectTimer = window.setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 30000);
      });

      socket.addEventListener("error", () => {
        setError("Live stream connection failed.");
        socket.close();
      });
    }

    connect();

    // Watchdog: connected but silent is a different failure from disconnected,
    // and it is the one that used to be invisible.
    const watchdog = window.setInterval(() => {
      if (!lastFrameAtRef.current) return;
      setIsStale(Date.now() - lastFrameAtRef.current > STALE_AFTER_MS);
    }, 1000);

    return () => {
      shouldReconnect = false;
      window.clearTimeout(reconnectTimer);
      window.clearInterval(watchdog);
      if (socket) socket.close();
    };
  }, [url]);

  return {
    networkState,
    lastRoutingEvent,
    lastComparison,
    failoverEvents,
    isConnected,
    isStale,
    error,
  };
}
