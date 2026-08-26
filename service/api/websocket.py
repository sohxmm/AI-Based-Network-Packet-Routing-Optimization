"""WebSocket streaming for the live dashboard.

The broadcast used to send to each client sequentially with ``await``, so one
slow client delayed every other client *and* the 1 Hz simulator loop that was
awaiting it. Sends now run concurrently with a per-client timeout.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from service.schemas.responses import state_to_dict
from service.state import get_source

logger = logging.getLogger(__name__)
router = APIRouter()

#: A client that cannot accept a frame within this window is dropped.
SEND_TIMEOUT_SECONDS = 2.0


class ConnectionManager:
    """Track active dashboard WebSocket clients."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug("WebSocket connected (%d active)", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.debug("WebSocket closed (%d active)", len(self.active_connections))

    async def broadcast(self, message: dict[str, object]) -> None:
        """Send to every client concurrently; drop the ones that fail."""
        targets = list(self.active_connections)
        if not targets:
            return

        results = await asyncio.gather(
            *(self._safe_send(ws, message) for ws in targets),
            return_exceptions=True,
        )
        for websocket, result in zip(targets, results, strict=True):
            if isinstance(result, Exception):
                self.disconnect(websocket)

    async def _safe_send(self, websocket: WebSocket, message: dict[str, object]) -> None:
        await asyncio.wait_for(websocket.send_json(message), timeout=SEND_TIMEOUT_SECONDS)


manager = ConnectionManager()


@router.websocket("/ws/stream")
async def stream_network_state(websocket: WebSocket) -> None:
    """Accept the connection, push the current state, then hold it open."""
    await manager.connect(websocket)

    try:
        await websocket.send_json(
            {"type": "state_update", "payload": state_to_dict(get_source().get_state())}
        )
    except Exception:  # noqa: BLE001 - the client may vanish mid-handshake
        manager.disconnect(websocket)
        return

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:  # noqa: BLE001
        logger.debug("WebSocket receive failed; closing", exc_info=True)
        manager.disconnect(websocket)


__all__ = ["ConnectionManager", "manager", "router"]
