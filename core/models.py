"""Domain dataclasses shared by the simulator, the routers and the API.

No web, no ML and no I/O dependencies live here on purpose: everything else in
the project imports these types, so they must stay at the bottom of the
dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LinkState:
    """One undirected network link at one instant in time."""

    source: str
    target: str
    base_latency: float
    bandwidth: int
    utilization: float
    queue_size: int
    packet_loss_rate: float


@dataclass
class NetworkState:
    """An immutable snapshot of the whole network."""

    nodes: list[str]
    links: list[LinkState]
    timestamp: float
    step_count: int


@dataclass
class RoutingDecision:
    """The result of asking one algorithm to route one (source, destination) pair.

    ``is_fallback`` is deliberately part of the domain model rather than an API
    detail: a learned router that quietly served a heuristic answer must be
    distinguishable from one that ran its model, everywhere in the system.
    """

    source: str
    destination: str
    path: list[str]
    algorithm: str
    total_latency: float
    avg_utilization: float
    success: bool
    is_fallback: bool = False
    # Populated by learned routers so the UI can explain a decision.
    diagnostics: dict = field(default_factory=dict)
