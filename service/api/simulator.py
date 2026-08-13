import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from api.state import get_simulator
from api.websocket import manager
from db.database import AsyncSessionLocal
from db.models import NetworkSnapshot
from simulator.data_models import NetworkState
from .common import LinkRequest, _state_to_dict

router = APIRouter()

async def handle_simulator_step(state: NetworkState) -> None:
    await manager.broadcast({
        "type": "state_update",
        "payload": _state_to_dict(state)
    })

    links = state.links
    avg_utilization = (
        sum(link.utilization for link in links) / len(links)
        if links
        else 0.0
    )
    congested_links = sum(1 for link in links if link.utilization >= 0.7)
    
    snapshot = NetworkSnapshot(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        state_json=_state_to_dict(state),
        avg_utilization=avg_utilization,
        congested_links=congested_links,
        step_count=state.step_count
    )
    
    try:
        async with AsyncSessionLocal() as session:
            session.add(snapshot)
            await session.commit()
    except Exception as exc:
        print(f"[DB] Error saving network snapshot: {exc}")

@router.post("/sim/step")
async def step_simulation() -> dict[str, object]:
    state = get_simulator().step()
    await handle_simulator_step(state)
    return _state_to_dict(state)

@router.post("/sim/reset")
async def reset_simulation() -> dict[str, object]:
    state = get_simulator().reset()
    await handle_simulator_step(state)
    return _state_to_dict(state)

@router.post("/sim/inject-failure")
def inject_failure(request: LinkRequest) -> dict[str, object]:
    simulator = get_simulator()
    try:
        simulator.inject_failure(request.source, request.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _state_to_dict(simulator.get_state())

@router.post("/sim/restore-link")
def restore_link(request: LinkRequest) -> dict[str, object]:
    simulator = get_simulator()
    try:
        simulator.restore_link(request.source, request.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _state_to_dict(simulator.get_state())
