from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from api.state import get_simulator
from db.database import get_db
from .common import (
    RouteRequest, RouteCompareRequest, DEFAULT_ALGORITHMS,
    _validate_nodes, _build_forecast_state, _run_algorithm,
    save_routing_event, broadcast_routing_event, _decision_to_dict, _state_to_dict,
    forecast_history, congestion_predictor
)

router = APIRouter()

@router.get("/network/state")
def get_network_state() -> dict[str, object]:
    return _state_to_dict(get_simulator().get_state())

@router.get("/network/topology")
def get_network_topology() -> dict[str, object]:
    state = get_simulator().get_state()
    return {
        "nodes": [{"id": node, "label": node} for node in state.nodes],
        "links": [
            {
                "source": link.source,
                "target": link.target,
                "base_latency": link.base_latency,
                "bandwidth": link.bandwidth,
            }
            for link in state.links
        ],
        "step_count": state.step_count,
    }

@router.post("/network/route")
async def route_packet(
    request: RouteRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    state = get_simulator().get_state()
    _validate_nodes(state, request.source, request.destination)

    routing_state = state
    if request.use_forecast and request.algorithm in ("gnn", "rl"):
        forecast_state = _build_forecast_state(state)
        if forecast_state is not None:
            routing_state = forecast_state

    decision = _run_algorithm(request.algorithm, routing_state, request.source, request.destination)
    
    await save_routing_event(decision, state.step_count)
    await broadcast_routing_event(decision)
    
    if not decision.success:
        raise HTTPException(
            status_code=400,
            detail=f"No path exists between {request.source} and {request.destination} using {request.algorithm}."
        )
        
    return _decision_to_dict(decision)

@router.post("/network/route/compare")
async def compare_routes(
    request: RouteCompareRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    state = get_simulator().get_state()
    _validate_nodes(state, request.source, request.destination)
    algorithms = request.algorithms or DEFAULT_ALGORITHMS
    
    decisions = []
    has_any_success = False
    
    for algorithm in algorithms:
        decision = _run_algorithm(algorithm, state, request.source, request.destination)
        await save_routing_event(decision, state.step_count)
        await broadcast_routing_event(decision)
        if decision.success: has_any_success = True
        decisions.append(_decision_to_dict(decision))

    if request.use_forecast:
        forecast_state = _build_forecast_state(state)
        if forecast_state is not None:
            for algo_name in ("gnn", "rl"):
                if algo_name in algorithms:
                    pred_decision = _run_algorithm(algo_name, forecast_state, request.source, request.destination)
                    pred_dict = _decision_to_dict(pred_decision)
                    pred_dict["algorithm"] = f"{algo_name}_predictive"
                    if pred_decision.success: has_any_success = True
                    decisions.append(pred_dict)

    if not has_any_success:
        raise HTTPException(
            status_code=400,
            detail=f"No path exists between {request.source} and {request.destination} using any selected algorithms."
        )
        
    return {
        "source": request.source,
        "destination": request.destination,
        "step_count": state.step_count,
        "use_forecast": request.use_forecast,
        "results": decisions,
    }

@router.get("/network/congestion-forecast")
def get_congestion_forecast(steps: int = 3) -> dict[str, object]:
    state = get_simulator().get_state()
    current_snapshot = [link.utilization for link in state.links]
    forecast_history.append(current_snapshot)
    del forecast_history[:-congestion_predictor.seq_len]

    predictions = []
    rolling_window = list(forecast_history)
    for step_index in range(max(1, min(steps, 10))):
        next_snapshot = congestion_predictor.predict_next(rolling_window)
        predictions.append(
            {
                "step_ahead": step_index + 1,
                "links": [
                    {
                        "source": link.source,
                        "target": link.target,
                        "predicted_utilization": next_snapshot[index],
                    }
                    for index, link in enumerate(state.links)
                ],
            }
        )
        rolling_window.append(next_snapshot)

    return {
        "step_count": state.step_count,
        "model_trained": congestion_predictor.model is not None,
        "predictions": predictions,
    }
