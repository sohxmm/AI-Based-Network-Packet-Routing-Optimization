"""Simulator control and the per-tick broadcast/persist hook."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from core.models import NetworkState
from service.api.websocket import manager
from service.db.models import NetworkSnapshot
from service.db.writes import save_snapshot
from service.schemas.requests import LinkRequest, SourceRequest
from service.schemas.responses import state_to_dict
from service.state import LIVE_PROBE_ENABLED, app_state, get_failover, get_simulator, get_source

logger = logging.getLogger(__name__)
router = APIRouter()

#: Persist one snapshot per N ticks. At 1 Hz with ~10 KB rows, writing every
#: tick produced roughly 860 MB/day with no retention.
SNAPSHOT_EVERY_N_STEPS = 10


async def handle_simulator_step(state: NetworkState) -> None:
    """Broadcast a tick, run failover checks, and persist occasionally.

    Persistence is fire-and-forget: it used to be awaited from inside the 1 Hz
    loop, so a slow database stalled the simulation itself.
    """
    await manager.broadcast({"type": "state_update", "payload": state_to_dict(state)})

    events = get_failover().tick(state)
    if events:
        await manager.broadcast(
            {"type": "failover", "payload": [event.as_dict() for event in events]}
        )

    if state.step_count % SNAPSHOT_EVERY_N_STEPS != 0:
        return

    links = state.links
    snapshot = NetworkSnapshot(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC),
        state_json=state_to_dict(state),
        avg_utilization=(
            sum(link.utilization for link in links) / len(links) if links else 0.0
        ),
        congested_links=sum(1 for link in links if link.utilization >= 0.7),
        step_count=state.step_count,
    )
    asyncio.create_task(save_snapshot(snapshot))


@router.post("/sim/step")
async def step_simulation() -> dict[str, object]:
    """Advance the network source by one tick."""
    state = get_source().step()
    await handle_simulator_step(state)
    return state_to_dict(state)


@router.post("/sim/reset")
async def reset_simulation() -> dict[str, object]:
    """Return the network source to its initial state."""
    state = get_source().reset()
    await handle_simulator_step(state)
    return state_to_dict(state)


@router.post("/sim/inject-failure")
def inject_failure(request: LinkRequest) -> dict[str, object]:
    """Fail a link, to demonstrate rerouting."""
    simulator = _require_simulator()
    try:
        simulator.inject_failure(request.source, request.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return state_to_dict(simulator.get_state())


@router.post("/sim/restore-link")
def restore_link(request: LinkRequest) -> dict[str, object]:
    """Restore a previously failed link."""
    simulator = _require_simulator()
    try:
        simulator.restore_link(request.source, request.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return state_to_dict(simulator.get_state())


@router.post("/sim/source")
def set_source(request: SourceRequest) -> dict[str, object]:
    """Switch between the simulator, a recorded trace and live measurement.

    Live mode measures real hosts, so it is gated behind ``LIVE_PROBE_ENABLED``
    and requires the targets to be listed explicitly.
    """
    try:
        if request.kind == "simulated":
            source = app_state.use_simulator(request.num_nodes, request.seed)
        elif request.kind == "trace":
            if not request.trace_path:
                raise HTTPException(
                    status_code=422, detail="trace_path is required for kind='trace'."
                )
            source = app_state.use_trace(request.trace_path)
        else:
            source = app_state.use_live(request.targets)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"source": source.describe(), "state": state_to_dict(source.get_state())}


@router.get("/sim/source/health")
def source_health() -> dict[str, object]:
    """Per-target reachability, when the active source is a live probe."""
    source = get_source()
    if not hasattr(source, "health"):
        return {
            "kind": source.kind,
            "live": False,
            "live_probe_enabled": LIVE_PROBE_ENABLED,
        }
    return {
        "kind": source.kind,
        "live": True,
        "live_probe_enabled": LIVE_PROBE_ENABLED,
        "targets": source.health(),
    }


def _require_simulator():
    """Failure injection only makes sense against the simulator."""
    simulator = get_simulator()
    if simulator is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "The active network source is measured or replayed, so links "
                "cannot be failed or restored. Switch to the simulator with "
                "POST /sim/source {\"kind\": \"simulated\"}."
            ),
        )
    return simulator


__all__ = ["SNAPSHOT_EVERY_N_STEPS", "handle_simulator_step", "router"]
