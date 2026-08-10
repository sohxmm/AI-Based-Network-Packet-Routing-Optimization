"""Domain layer: network model, cost function, path primitives, simulator.

Nothing in this package imports the web service, the ML stack or the benchmark
harness. It is the scientific core of the project and can be used standalone.
"""

from core.cost import (
    CONGESTION_EXPONENT,
    CONGESTION_PENALTY_FACTOR,
    MAX_BASE_LATENCY,
    MAX_LATENCY_MS,
    MAX_LOSS,
    MAX_QUEUE,
    link_cost,
    raw_edge_cost,
)
from core.models import LinkState, NetworkState, RoutingDecision
from core.paths import (
    average_path_utilization,
    build_adjacency,
    build_decision,
    build_graph,
    candidate_paths,
    failed_decision,
    hop_breakdown,
    link_lookup,
    max_path_utilization,
    path_cost,
    path_links,
)
from core.simulator import NetworkSimulator

__all__ = [
    "CONGESTION_EXPONENT",
    "CONGESTION_PENALTY_FACTOR",
    "MAX_BASE_LATENCY",
    "MAX_LATENCY_MS",
    "MAX_LOSS",
    "MAX_QUEUE",
    "LinkState",
    "NetworkSimulator",
    "NetworkState",
    "RoutingDecision",
    "average_path_utilization",
    "build_adjacency",
    "build_decision",
    "build_graph",
    "candidate_paths",
    "failed_decision",
    "hop_breakdown",
    "link_cost",
    "link_lookup",
    "max_path_utilization",
    "path_cost",
    "path_links",
    "raw_edge_cost",
]
