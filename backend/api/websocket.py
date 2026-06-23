from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.state import get_simulator

router = APIRouter()


class ConnectionManager:
    """Track active dashboard WebSocket clients."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and remember a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Forget a WebSocket connection that has closed."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@router.websocket("/ws/stream")
async def stream_network_state(websocket: WebSocket) -> None:
    """Push the current network state to one dashboard client every second."""
    await manager.connect(websocket)

    try:
        while True:
            await websocket.send_json(
                {
                    "type": "state_update",
                    "payload": _state_to_dict(),
                }
            )
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def _state_to_dict() -> dict[str, object]:
    """Serialize the latest simulator state for WebSocket clients."""
    state = get_simulator().get_state()
    return {
        "nodes": state.nodes,
        "links": [asdict(link) for link in state.links],
        "timestamp": state.timestamp,
        "step_count": state.step_count,
    }
