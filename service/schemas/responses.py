"""Response serializers.

One place where a domain object becomes JSON. ``_state_to_dict`` previously
existed in three files (``common.py``, ``websocket.py`` and the simulator
itself), which is how the frontend ended up re-deriving field names by hand and
how the benchmark output schema drifted away from the code that reads it.
"""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite

from core.models import NetworkState, RoutingDecision
from core.paths import hop_breakdown
from core.qos import QoSProfile, evaluate_path


def state_to_dict(state: NetworkState) -> dict[str, object]:
    """Serialize a network state for the REST API and the WebSocket stream."""
    return {
        "nodes": state.nodes,
        "links": [asdict(link) for link in state.links],
        "timestamp": state.timestamp,
        "step_count": state.step_count,
    }


def decision_to_dict(
    decision: RoutingDecision,
    state: NetworkState | None = None,
    profile: QoSProfile | None = None,
) -> dict[str, object]:
    """Serialize a routing decision, including a per-hop cost breakdown.

    ``total_latency`` becomes ``None`` rather than ``Infinity`` because JSON has
    no infinity literal and ``json.dumps`` emits a bare ``Infinity`` token that
    strict parsers reject.

    When *profile* is given, a ``qos`` block is computed here from the returned
    path rather than read out of ``decision.diagnostics``. The distinction
    matters: ``diagnostics`` is whatever the router chose to say about itself,
    and only the constraint-aware routers say anything, so the comparison view
    showed a QoS verdict for every algorithm except Dijkstra and Bellman-Ford —
    the two that are constraint-*blind* and therefore the two whose feasibility
    a reader most needs to see. Feasibility is a property of (path, state,
    profile); computing it here gives every row the same independently derived
    answer.
    """
    payload: dict[str, object] = {
        "source": decision.source,
        "destination": decision.destination,
        "path": decision.path,
        "algorithm": decision.algorithm,
        "total_latency": (
            decision.total_latency if isfinite(decision.total_latency) else None
        ),
        "avg_utilization": decision.avg_utilization,
        "success": decision.success,
        "is_fallback": decision.is_fallback,
        "diagnostics": decision.diagnostics or {},
    }
    if state is not None and decision.path:
        payload["hops"] = hop_breakdown(state, decision.path)
        if profile is not None:
            evaluation = evaluate_path(state, decision.path, profile)
            payload["qos"] = {
                "traffic_class": profile.traffic_class.value,
                "feasible": evaluation.feasible,
                "score": evaluation.score if isfinite(evaluation.score) else None,
                "total_loss": evaluation.total_loss,
                "bottleneck_utilization": evaluation.bottleneck_utilization,
                "hops": evaluation.hops,
                "violations": list(evaluation.violations),
            }
    return payload


__all__ = ["decision_to_dict", "state_to_dict"]
