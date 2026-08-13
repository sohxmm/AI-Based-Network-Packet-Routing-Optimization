from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, JSON, DateTime, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RoutingEvent(Base):
    """Log of every routing decision made by the system."""
    __tablename__ = "routing_events"
    
    # Columns
    id = Column(String, primary_key=True)  # UUID string
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String, index=True)  # Router ID (e.g., "R1")
    destination = Column(String, index=True)  # Router ID (e.g., "R10")
    algorithm = Column(String, index=True)  # "dijkstra" | "bellman_ford" | "aco" | "rl"
    path = Column(JSON)  # List of router IDs: ["R1", "R3", "R5"]
    total_latency = Column(Float)  # milliseconds
    success = Column(Boolean)  # Did route find path?
    step_count = Column(Integer, index=True)  # Simulator step when decision made


class NetworkSnapshot(Base):
    """Complete network state snapshot at a point in time."""
    __tablename__ = "network_snapshots"
    
    id = Column(String, primary_key=True)  # UUID
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    state_json = Column(JSON)  # Entire NetworkState as JSON
    avg_utilization = Column(Float)  # Average link utilization (0-1)
    congested_links = Column(Integer)  # Count of links with util > 0.7
    step_count = Column(Integer, index=True)


class AlgorithmMetric(Base):
    """Aggregate performance metrics per algorithm over a time window."""
    __tablename__ = "algorithm_metrics"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    algorithm = Column(String, index=True)
    window_start_step = Column(Integer)
    window_end_step = Column(Integer)
    avg_latency = Column(Float)  # Average latency for all decisions in window
    success_rate = Column(Float)  # % of decisions that found a path
    num_decisions = Column(Integer)


class PacketLog(Base):
    """Log of simulated packet transmissions."""
    __tablename__ = "packet_logs"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String, index=True)
    destination = Column(String, index=True)
    path = Column(JSON)  # ["R1", "R3", "R10"]
    success = Column(Boolean)  # Did packet reach destination?
    arrival_time = Column(Float)  # Simulated arrival time
