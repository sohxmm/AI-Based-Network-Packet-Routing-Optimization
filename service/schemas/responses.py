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


def state_to_dict(state: NetworkState) -> dict[str, object]:
    """Serialize a network state for the REST API and the WebSocket stream."""
    return {
        "nodes": state.nodes,
        "links": [asdict(link) for link in state.links],
        "timestamp": state.timestamp,
        "step_count": state.step_count,
    }


def decision_to_dict(
    decision: RoutingDecision, state: NetworkState | None = None
) -> dict[str, object]:
    """Serialize a routing decision, including a per-hop cost breakdown.

    ``total_latency`` becomes ``None`` rather than ``Infinity`` because JSON has
    no infinity literal and ``json.dumps`` emits a bare ``Infinity`` token that
    strict parsers reject.
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
    return payload


__all__ = ["decision_to_dict", "state_to_dict"]
