"""The cost model for this project. Defined here and nowhere else.

Before this module existed the formula ``base_latency * (1 + 4 * u**2)`` was
copy-pasted across 14 sites in 11 files. That was a correctness hazard, not a
style problem: a partial edit would silently make the benchmark compare
algorithms that were optimising different objectives.

Every router, every training script, every benchmark and the simulator itself
now import :func:`link_cost` (or the two constants) from here.
"""

from __future__ import annotations

from core.models import LinkState

# ---------------------------------------------------------------------------
# The congestion model
#
#   cost(link) = base_latency * (1 + PENALTY * utilization ** EXPONENT)
#
# A quadratic penalty means a link at 100% utilization costs 5x its idle
# latency, while a link at 50% costs only 2x. That convexity is what makes
# spreading load across two half-full links cheaper than saturating one.
# ---------------------------------------------------------------------------
CONGESTION_EXPONENT = 2
CONGESTION_PENALTY_FACTOR = 4.0

# Normalisation bounds shared by every observation builder, so that training
# and serving cannot drift apart.
MAX_LATENCY_MS = 200.0
MAX_QUEUE = 100.0
MAX_LOSS = 0.06
MAX_BASE_LATENCY = 25.0


def link_cost(link: LinkState) -> float:
    """Congestion-adjusted latency for one link. THE cost model for this project."""
    return link.base_latency * (
        1 + CONGESTION_PENALTY_FACTOR * link.utilization**CONGESTION_EXPONENT
    )


def raw_edge_cost(base_latency: float, utilization: float) -> float:
    """Same formula for callers holding plain scalars rather than a LinkState.

    Used by :class:`core.simulator.NetworkSimulator`, which stores edge
    attributes in networkx dicts rather than dataclasses.
    """
    return base_latency * (
        1 + CONGESTION_PENALTY_FACTOR * utilization**CONGESTION_EXPONENT
    )
