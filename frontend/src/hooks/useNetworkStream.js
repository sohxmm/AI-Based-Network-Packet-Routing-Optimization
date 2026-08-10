import { useEffect, useState } from "react";

import { wsUrl } from "../config.js";

export function useNetworkStream(url = wsUrl("/ws/stream")) {
  const [networkState, setNetworkState] = useState(null);
  const [lastRoutingEvent, setLastRoutingEvent] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);

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
      });

      socket.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);

        if (message.type === "state_update") {
          setNetworkState(message.payload);
        }

        if (message.type === "routing_event") {
          setLastRoutingEvent(message.payload);
        }
      });

      socket.addEventListener("close", () => {
        setIsConnected(false);

        if (!shouldReconnect) {
          return;
        }

        reconnectTimer = window.setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 30000);
      });

      socket.addEventListener("error", () => {
        setError("Live stream connection failed.");
        socket.close();
      });
    }

    connect();

    return () => {
      shouldReconnect = false;
      window.clearTimeout(reconnectTimer);
      if (socket) {
        socket.close();
      }
    };
  }, [url]);

  return { networkState, lastRoutingEvent, isConnected, error };
}
