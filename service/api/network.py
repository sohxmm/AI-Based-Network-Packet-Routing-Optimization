"""Network state, routing, QoS, forecasting and failover endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.paths import candidate_paths, hop_breakdown
from core.qos import ALL_CLASSES, QOS_PROFILES, evaluate_path, get_profile
from routing import describe_algorithms
from routing.classical.constrained import qos_oracle
from routing.failover import measure_convergence
from service.api.dispatch import (
    current_state,
    forecast_state,
    register_flow,
    run_algorithm,
    validate_nodes,
)
from service.api.websocket import manager
from service.db.writes import build_routing_event, save_routing_events
from service.schemas.requests import (
    DEFAULT_ALGORITHMS,
    ConvergenceRequest,
    RouteCompareRequest,
    RouteRequest,
    WatchFlowRequest,
)
from service.schemas.responses import decision_to_dict, state_to_dict
from service.state import get_failover, get_forecaster, get_router, get_simulator, get_source

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/network/state")
def get_network_state() -> dict[str, object]:
    """The current network state."""
    return state_to_dict(current_state())


@router.get("/network/source")
def get_network_source() -> dict[str, object]:
    """Where the current state comes from: simulator, trace or live probe."""
    return get_source().describe()


@router.get("/network/topology")
def get_network_topology() -> dict[str, object]:
    """Static topology, without the time-varying link metrics."""
    state = current_state()
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


@router.get("/network/algorithms")
def list_algorithms() -> dict[str, object]:
    """Algorithm and traffic-class metadata for the dashboard pickers."""
    return {
        "algorithms": describe_algorithms(),
        "traffic_classes": [
            {
                "name": profile.traffic_class.value,
                "label": profile.label,
                "description": profile.description,
                "weights": {
                    "latency": profile.w_latency,
                    "loss": profile.w_loss,
                    "utilization": profile.w_utilization,
                },
                "constraints": {
                    "max_path_loss": profile.max_path_loss,
                    "max_bottleneck_utilization": profile.max_bottleneck_utilization,
                    "max_hops": profile.max_hops,
                },
            }
            for profile in (QOS_PROFILES[c] for c in ALL_CLASSES)
        ],
    }


@router.post("/network/route")
async def route_packet(request: RouteRequest) -> dict[str, object]:
    """Route one demand with one algorithm."""
    state = current_state()
    validate_nodes(state, request.source, request.destination)

    routing_state = state
    forecast_used = False
    if request.use_forecast:
        predicted = forecast_state(state)
        if predicted is not None:
            routing_state = predicted
            forecast_used = True

    decision = run_algorithm(
        request.algorithm,
        routing_state,
        request.source,
        request.destination,
        request.traffic_class,
    )

    # A failed decision is a 404 (nothing exists), not a 400 (bad request), and
    # it is no longer persisted and broadcast *before* raising - the WebSocket
    # used to show an event the HTTP caller had been told failed.
    if not decision.success:
        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    f"No path exists between {request.source} and "
                    f"{request.destination} using {request.algorithm}."
                ),
                "source": request.source,
                "destination": request.destination,
                "algorithm": request.algorithm,
            },
        )

    register_flow(decision.path)
    await save_routing_events(
        [build_routing_event(decision, state.step_count, request.traffic_class)]
    )
    payload = decision_to_dict(decision, state, get_profile(request.traffic_class))
    payload["forecast_used"] = forecast_used
    await manager.broadcast({"type": "routing_event", "payload": payload})
    return payload


@router.post("/network/route/compare")
async def compare_routes(request: RouteCompareRequest) -> dict[str, object]:
    """Route one demand with several algorithms, for the divergence view."""
    state = current_state()
    validate_nodes(state, request.source, request.destination)
    algorithms = request.algorithms or DEFAULT_ALGORITHMS

    profile = get_profile(request.traffic_class)
    forecast = forecast_state(state) if request.use_forecast else None
    decisions = []
    results = []

    for algorithm in algorithms:
        decision = run_algorithm(
            algorithm, state, request.source, request.destination, request.traffic_class
        )
        decisions.append(decision)
        results.append(decision_to_dict(decision, state, profile))

        # A predictive variant is only emitted when a forecast actually exists.
        # Emitting it unconditionally is how the benchmark ended up with two
        # columns byte-identical to their base algorithms.
        if forecast is not None and algorithm in ("gnn", "rl"):
            predicted = run_algorithm(
                algorithm, forecast, request.source, request.destination, request.traffic_class
            )
            # Scored against the *real* state, not the forecast it routed on:
            # the question is whether the predicted-optimal path is feasible in
            # the network that actually exists.
            entry = decision_to_dict(predicted, state, profile)
            entry["algorithm"] = f"{algorithm}_predictive"
            entry["forecast_used"] = True
            results.append(entry)

    if not any(d.success for d in decisions):
        raise HTTPException(
            status_code=404,
            detail={
                "message": (
                    f"No path exists between {request.source} and "
                    f"{request.destination} using any selected algorithm."
                ),
                "algorithms": algorithms,
            },
        )

    # One transaction and one broadcast, not one of each per algorithm.
    await save_routing_events(
        [
            build_routing_event(d, state.step_count, request.traffic_class)
            for d in decisions
        ]
    )
    await manager.broadcast({"type": "routing_comparison", "payload": results})

    oracle_path, oracle_score, oracle_feasible = qos_oracle(
        state, request.source, request.destination, profile
    )

    return {
        "source": request.source,
        "destination": request.destination,
        "step_count": state.step_count,
        "traffic_class": request.traffic_class,
        "use_forecast": request.use_forecast,
        "forecast_available": forecast is not None,
        "results": results,
        "oracle": {
            "path": oracle_path,
            "score": None if oracle_score == float("inf") else oracle_score,
            "feasible": oracle_feasible,
        },
    }


@router.get("/network/candidates")
def get_candidates(
    source: str, destination: str, traffic_class: str = "best_effort", k: int = 5
) -> dict[str, object]:
    """The candidate set every algorithm chooses from, scored under a QoS class.

    Exposed so the dashboard can show *why* one algorithm diverged from another:
    they are picking different rows of this same table.
    """
    state = current_state()
    validate_nodes(state, source, destination)
    profile = get_profile(traffic_class)

    paths = candidate_paths(state, source, destination, k=max(1, min(k, 10)))
    return {
        "source": source,
        "destination": destination,
        "traffic_class": traffic_class,
        "candidates": [
            {
                "path": path,
                "hops": hop_breakdown(state, path),
                "qos": evaluate_path(state, path, profile).as_dict(),
            }
            for path in paths
        ],
    }


@router.get("/network/congestion-forecast")
def get_congestion_forecast() -> dict[str, object]:
    """Next-step link utilization, or an honest statement that it is unavailable.

    This endpoint used to return persistence values (a copy of the current
    utilization) labelled ``predicted_utilization`` whenever the model was
    absent, and to mutate a module-level global from a GET handler.
    """
    state = current_state()
    forecaster = get_forecaster()

    if not forecaster.is_trained:
        return {
            "step_count": state.step_count,
            "model_trained": False,
            "predictions": [],
            "message": (
                "No trained LSTM forecaster. Run `python -m ml.training.train_lstm` "
                "to enable forecasting. Copying the current utilization forward "
                "would not be a forecast, so nothing is returned."
            ),
        }

    forecaster.observe(state)
    if len(forecaster.history) < forecaster.seq_len:
        return {
            "step_count": state.step_count,
            "model_trained": True,
            "predictions": [],
            "message": (
                f"Warming up: {len(forecaster.history)}/{forecaster.seq_len} "
                "snapshots collected."
            ),
        }

    predicted = forecaster.predict_next(forecaster.history)
    return {
        "step_count": state.step_count,
        "model_trained": True,
        "skill_score": forecaster.skill_score,
        "predictions": [
            {
                "step_ahead": 1,
                "links": [
                    {
                        "source": link.source,
                        "target": link.target,
                        "current_utilization": round(link.utilization, 4),
                        "predicted_utilization": round(predicted[index], 4),
                    }
                    for index, link in enumerate(state.links)
                    if index < len(predicted)
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Fault-tolerant rerouting
# ---------------------------------------------------------------------------
@router.get("/network/failover")
def failover_status() -> dict[str, object]:
    """Which flows are being watched, and what has been rerouted recently."""
    return get_failover().snapshot()


@router.post("/network/failover/watch")
def watch_flow(request: WatchFlowRequest) -> dict[str, object]:
    """Start monitoring a demand so it is rerouted automatically on failure."""
    state = current_state()
    validate_nodes(state, request.source, request.destination)

    monitor = get_failover()
    flow = monitor.watch(request.source, request.destination, request.traffic_class)

    decision = run_algorithm(
        "gnn", state, request.source, request.destination, request.traffic_class
    )
    flow.path = list(decision.path) if decision.success else []
    return {"watching": monitor.snapshot()["watched"]}


@router.delete("/network/failover/watch")
def unwatch_flow(
    source: str, destination: str, traffic_class: str = "best_effort"
) -> dict[str, object]:
    """Stop monitoring a demand."""
    monitor = get_failover()
    monitor.unwatch(source, destination, traffic_class)
    return {"watching": monitor.snapshot()["watched"]}


@router.post("/network/failover/convergence")
def convergence_test(request: ConvergenceRequest) -> dict[str, object]:
    """Fail a link and measure how fast each algorithm restores service.

    Runs on a private simulator seeded from the live one, so a convergence
    experiment cannot damage the running demo.
    """
    from core.simulator import NetworkSimulator

    live = get_simulator()
    if live is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Convergence testing requires the simulator source. The active "
                "source is measured or replayed and cannot have failures injected."
            ),
        )

    algorithms = request.algorithms or ["dijkstra", "aco", "gnn", "rl", "multi_agent"]
    results = []
    for algorithm in algorithms:
        sandbox = NetworkSimulator(num_nodes=live.num_nodes, seed=live.seed)
        for _ in range(live.step_count % 200):
            sandbox.step()
        results.append(
            measure_convergence(
                sandbox,
                get_router(algorithm),
                request.source,
                request.destination,
                (request.link_source, request.link_target),
                request.traffic_class,
                request.max_steps,
            )
        )

    return {
        "failed_link": [request.link_source, request.link_target],
        "traffic_class": request.traffic_class,
        "results": results,
        "note": (
            "Convergence is measured in simulator ticks until the algorithm "
            "again returns a QoS-satisfying route. A fast recovery onto a worse "
            "path is not better than a slower recovery onto a good one, so the "
            "latency before and after is reported alongside."
        ),
    }


__all__ = ["router"]
