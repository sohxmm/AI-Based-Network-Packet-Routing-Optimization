"""Request models.

These used to live in ``api/routers/common.py`` alongside the algorithm
dispatcher, the LSTM forecast builder, the response serializers, the database
helpers and a module-level mutable global — six unrelated responsibilities in
one file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.qos import ALL_CLASSES
from routing import ALGORITHM_NAMES

AlgorithmName = Literal[
    "dijkstra",
    "bellman_ford",
    "constrained",
    "aco",
    "gnn",
    "rl",
    "multi_agent",
    "random_baseline",
]

TrafficClassName = Literal[
    "emergency", "interactive", "gaming", "bulk", "best_effort"
]

DEFAULT_ALGORITHMS: list[str] = list(ALGORITHM_NAMES)
DEFAULT_TRAFFIC_CLASSES: list[str] = [c.value for c in ALL_CLASSES]


class RouteRequest(BaseModel):
    """Route one demand with one algorithm."""

    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R14"])
    algorithm: AlgorithmName = "dijkstra"
    traffic_class: TrafficClassName = "best_effort"
    use_forecast: bool = False


class RouteCompareRequest(BaseModel):
    """Route one demand with several algorithms for side-by-side comparison."""

    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R14"])
    algorithms: list[AlgorithmName] | None = None
    traffic_class: TrafficClassName = "best_effort"
    use_forecast: bool = False


class LinkRequest(BaseModel):
    """Identify one link for failure injection or restoration."""

    source: str = Field(..., min_length=1, examples=["R1"])
    target: str = Field(..., min_length=1, examples=["R2"])


class WatchFlowRequest(BaseModel):
    """Register a demand with the failover monitor."""

    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R14"])
    traffic_class: TrafficClassName = "best_effort"


class ConvergenceRequest(BaseModel):
    """Fail a link and measure how quickly each algorithm restores service."""

    source: str = Field(..., min_length=1, examples=["R1"])
    destination: str = Field(..., min_length=1, examples=["R14"])
    link_source: str = Field(..., min_length=1, examples=["R1"])
    link_target: str = Field(..., min_length=1, examples=["R2"])
    algorithms: list[AlgorithmName] | None = None
    traffic_class: TrafficClassName = "best_effort"
    max_steps: int = Field(20, ge=1, le=100)


class SourceRequest(BaseModel):
    """Switch the platform between the simulator, a trace and live probing."""

    kind: Literal["simulated", "trace", "live"] = "simulated"
    #: Path to a JSONL/CSV trace, required when kind == "trace".
    trace_path: str | None = None
    #: Hosts to probe, required when kind == "live". Explicit by design: the
    #: platform never discovers or scans targets on its own.
    targets: list[str] | None = None
    num_nodes: int = Field(25, ge=5, le=200)
    seed: int = 42


__all__ = [
    "DEFAULT_ALGORITHMS",
    "DEFAULT_TRAFFIC_CLASSES",
    "AlgorithmName",
    "ConvergenceRequest",
    "LinkRequest",
    "RouteCompareRequest",
    "RouteRequest",
    "SourceRequest",
    "TrafficClassName",
    "WatchFlowRequest",
]
