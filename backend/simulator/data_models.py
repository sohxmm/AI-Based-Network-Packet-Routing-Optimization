from dataclasses import dataclass
from typing import List


@dataclass
class LinkState:
    source: str
    target: str
    base_latency: float
    bandwidth: int
    utilization: float
    queue_size: int
    packet_loss_rate: float


@dataclass
class NetworkState:
    nodes: List[str]
    links: List[LinkState]
    timestamp: float
    step_count: int


@dataclass
class RoutingDecision:
    source: str
    destination: str
    path: List[str]
    algorithm: str
    total_latency: float
    avg_utilization: float
    success: bool
    is_fallback: bool = False