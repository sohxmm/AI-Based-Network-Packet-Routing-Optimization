#added docstrings for easier conversion to json format and helpful for future reference
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class LinkState:
    """Represents the current state of a single network link between two routers."""

    source: str
    target: str
    base_latency: float
    bandwidth: int
    utilization: float
    queue_size: int
    packet_loss_rate: float


@dataclass
class NetworkState:
    """Snapshot of the entire network at a single point in time."""

    nodes: List[str]
    links: List[LinkState]
    timestamp: float
    step_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert NetworkState to a JSON-serializable dictionary."""
        return {
            "nodes": self.nodes,
            "links": [asdict(link) for link in self.links],
            "timestamp": self.timestamp,
            "step_count": self.step_count,
        }


@dataclass
class RoutingDecision:
    """Records the outcome of a single routing algorithm decision."""

    source: str
    destination: str
    path: List[str]
    algorithm: str
    total_latency: float
    avg_utilization: float
    success: bool