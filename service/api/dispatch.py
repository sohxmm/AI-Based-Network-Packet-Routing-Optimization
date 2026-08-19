"""Algorithm dispatch: name in, routing decision out.

One function, one lookup, one place. The old dispatcher hand-built a table
mixing bare functions with bound methods of three different names, and every
other module that needed the algorithm list kept its own copy — which drifted.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException

from core.models import NetworkState, RoutingDecision
from core.qos import get_profile
from routing import ALGORITHM_NAMES
from routing.learned.forecaster import build_forecast_state
from service.state import get_forecaster, get_router, get_source

logger = logging.getLogger(__name__)


def run_algorithm(
    algorithm: str,
    state: NetworkState,
    source: str,
    destination: str,
    traffic_class: str = "best_effort",
) -> RoutingDecision:
    """Route ``source -> destination`` with *algorithm* under *traffic_class*."""
    if algorithm not in ALGORITHM_NAMES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown algorithm {algorithm!r}. "
                f"Supported: {', '.join(ALGORITHM_NAMES)}."
            ),
        )
    return get_router(algorithm).find_route(
        state, source, destination, get_profile(traffic_class)
    )


def validate_nodes(state: NetworkState, source: str, destination: str) -> None:
    """404 with a structured, actionable detail when a node does not exist."""
    missing = [node for node in (source, destination) if node not in state.nodes]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Unknown router node.",
                "missing_nodes": missing,
                "available_nodes": state.nodes,
            },
        )


def forecast_state(state: NetworkState) -> NetworkState | None:
    """Return a one-step-ahead state, or None when forecasting is unavailable.

    Returning ``None`` rather than the current state matters: the caller must
    know that no forecast happened. The previous code fell through to
    ``or state`` and reported present-tense data as a prediction, which is why
    the predictive benchmark columns were byte-identical to their base
    algorithms in every published result.
    """
    forecaster = get_forecaster()
    if not forecaster.is_trained:
        return None
    forecaster.observe(state)
    return build_forecast_state(state, forecaster)


def current_state() -> NetworkState:
    """Read the active network source without advancing it."""
    return get_source().get_state()


def register_flow(path: list[str]) -> None:
    """Feed a served route back into the network, when the source supports it.

    This is what makes the live dashboard demonstrate the closed loop: routing a
    packet in the UI visibly raises utilization on the links it used.
    """
    if path:
        get_source().register_flow(path, demand=1.0)


__all__ = [
    "current_state",
    "forecast_state",
    "register_flow",
    "run_algorithm",
    "validate_nodes",
]
