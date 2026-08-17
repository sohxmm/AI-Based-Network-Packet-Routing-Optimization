"""Quality-of-Service classes and multi-constrained path scoring.

The original problem statement (section 17E) asks for QoS-aware routing that
prioritises video calls, gaming and emergency traffic. This module is what makes
that a measurable objective rather than a slogan, and it is also the regime in
which a learned policy can legitimately beat the classical baseline.

Why it is genuinely hard
------------------------
Dijkstra is provably optimal for *any single additive* edge cost, so simply
re-weighting the cost per traffic class would not create room for a learned
policy to win — Dijkstra would just solve the re-weighted problem exactly.

What this module defines instead is a **multi-constrained optimal path** (MCOP)
problem: minimise a class-weighted additive cost *subject to* per-class hard
constraints, one of which (bottleneck utilization) is **non-additive**:

    minimise   sum over links of  qos_link_cost(link, profile)
    subject to sum of packet loss along path  <=  max_path_loss
               max  utilization along path    <=  max_bottleneck_utilization
               hop count                      <=  max_hops

Multi-constrained path selection with two or more independent constraints is
NP-hard in the general case. Congestion-weighted Dijkstra optimises the
objective while ignoring the constraints entirely, so it can and does return
infeasible paths. That gap — *constraint satisfaction rate*, not mean latency —
is the headline QoS metric of this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.cost import (
    CONGESTION_EXPONENT,
    CONGESTION_PENALTY_FACTOR,
    MAX_BASE_LATENCY,
    MAX_LOSS,
)
from core.models import LinkState, NetworkState
from core.paths import path_links


class TrafficClass(str, Enum):
    """Packet classes recognised by the routing engine."""

    EMERGENCY = "emergency"
    INTERACTIVE = "interactive"  # voice and video calls
    GAMING = "gaming"
    BULK = "bulk"  # backups, downloads, replication
    BEST_EFFORT = "best_effort"  # unclassified traffic; the historical default


@dataclass(frozen=True)
class QoSProfile:
    """Objective weights and hard constraints for one traffic class.

    Weights apply to normalised per-link quantities so they are commensurate:
    latency is divided by :data:`core.cost.MAX_BASE_LATENCY`, loss by
    :data:`core.cost.MAX_LOSS`, utilization is already in [0, 1].
    """

    traffic_class: TrafficClass
    label: str
    description: str

    w_latency: float
    w_loss: float
    w_utilization: float

    # Hard constraints. None means "unconstrained on this axis".
    max_path_loss: float | None = None
    max_bottleneck_utilization: float | None = None
    max_hops: int | None = None

    # Relative scheduling priority, used when several classes contend.
    priority: int = 0


# ---------------------------------------------------------------------------
# The five shipped profiles.
#
# Constraint values are calibrated against the simulator's own dynamics: mean
# link utilization sits near 0.40 and per-link loss is max(0, u - 0.7) * 0.2,
# so a 0.70 bottleneck cap is binding but satisfiable on most pairs.
# ---------------------------------------------------------------------------
QOS_PROFILES: dict[TrafficClass, QoSProfile] = {
    TrafficClass.EMERGENCY: QoSProfile(
        traffic_class=TrafficClass.EMERGENCY,
        label="Emergency",
        description=(
            "Life-safety traffic. Packet loss dominates the objective and the "
            "path must avoid any near-saturated link."
        ),
        w_latency=0.25,
        w_loss=0.60,
        w_utilization=0.15,
        max_path_loss=0.010,
        max_bottleneck_utilization=0.70,
        max_hops=8,
        priority=4,
    ),
    TrafficClass.INTERACTIVE: QoSProfile(
        traffic_class=TrafficClass.INTERACTIVE,
        label="Voice / video",
        description=(
            "Interactive real-time media. Latency and jitter matter; queueing "
            "delay is proxied by link utilization."
        ),
        w_latency=0.50,
        w_loss=0.25,
        w_utilization=0.25,
        max_path_loss=0.020,
        max_bottleneck_utilization=0.80,
        max_hops=8,
        priority=3,
    ),
    TrafficClass.GAMING: QoSProfile(
        traffic_class=TrafficClass.GAMING,
        label="Gaming",
        description=(
            "Latency-critical but loss-tolerant. Optimises almost purely for "
            "delay, with a tight hop budget."
        ),
        w_latency=0.80,
        w_loss=0.05,
        w_utilization=0.15,
        max_path_loss=0.040,
        max_bottleneck_utilization=0.85,
        max_hops=6,
        priority=2,
    ),
    TrafficClass.BULK: QoSProfile(
        traffic_class=TrafficClass.BULK,
        label="Bulk transfer",
        description=(
            "Backups and replication. Delay-insensitive, so it is steered onto "
            "under-used links to keep capacity free for higher classes."
        ),
        w_latency=0.10,
        w_loss=0.10,
        w_utilization=0.80,
        max_path_loss=None,
        max_bottleneck_utilization=0.95,
        max_hops=None,
        priority=1,
    ),
    TrafficClass.BEST_EFFORT: QoSProfile(
        traffic_class=TrafficClass.BEST_EFFORT,
        label="Best effort",
        description=(
            "Unclassified traffic. Reproduces the project's original "
            "congestion-weighted latency objective exactly, with no constraints."
        ),
        w_latency=1.0,
        w_loss=0.0,
        w_utilization=0.0,
        max_path_loss=None,
        max_bottleneck_utilization=None,
        max_hops=None,
        priority=0,
    ),
}

DEFAULT_CLASS = TrafficClass.BEST_EFFORT


def get_profile(traffic_class: TrafficClass | str | None) -> QoSProfile:
    """Resolve *traffic_class* to a :class:`QoSProfile` (best-effort by default)."""
    if traffic_class is None:
        return QOS_PROFILES[DEFAULT_CLASS]
    if isinstance(traffic_class, str):
        try:
            traffic_class = TrafficClass(traffic_class)
        except ValueError:
            return QOS_PROFILES[DEFAULT_CLASS]
    return QOS_PROFILES[traffic_class]


def qos_link_cost(link: LinkState, profile: QoSProfile) -> float:
    """Class-weighted cost of one link, in normalised units.

    For :data:`TrafficClass.BEST_EFFORT` this reduces exactly to
    :func:`core.cost.link_cost` divided by ``MAX_BASE_LATENCY``, so best-effort
    routing is bit-for-bit the project's original objective.
    """
    congested_latency = link.base_latency * (
        1 + CONGESTION_PENALTY_FACTOR * link.utilization**CONGESTION_EXPONENT
    )
    return (
        profile.w_latency * (congested_latency / MAX_BASE_LATENCY)
        + profile.w_loss * (link.packet_loss_rate / MAX_LOSS)
        + profile.w_utilization * link.utilization
    )


@dataclass
class QoSEvaluation:
    """The result of scoring one path against one QoS profile."""

    feasible: bool
    score: float
    total_loss: float
    bottleneck_utilization: float
    hops: int
    violations: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "score": round(self.score, 4),
            "total_loss": round(self.total_loss, 5),
            "bottleneck_utilization": round(self.bottleneck_utilization, 4),
            "hops": self.hops,
            "violations": list(self.violations),
        }


_INFEASIBLE = QoSEvaluation(
    feasible=False,
    score=float("inf"),
    total_loss=float("inf"),
    bottleneck_utilization=1.0,
    hops=0,
    violations=["no_path"],
)


def evaluate_path(
    state: NetworkState, path: list[str], profile: QoSProfile
) -> QoSEvaluation:
    """Score *path* under *profile* and check every hard constraint."""
    links = path_links(state, path)
    if links is None:
        return _INFEASIBLE
    if not links:
        # A zero-hop path (source == destination) traverses nothing, so it
        # satisfies every constraint at zero cost.
        return QoSEvaluation(
            feasible=True,
            score=0.0,
            total_loss=0.0,
            bottleneck_utilization=0.0,
            hops=0,
            violations=[],
        )

    score = sum(qos_link_cost(link, profile) for link in links)
    total_loss = 1.0 - _product(1.0 - link.packet_loss_rate for link in links)
    bottleneck = max(link.utilization for link in links)
    hops = len(links)

    violations: list[str] = []
    if profile.max_path_loss is not None and total_loss > profile.max_path_loss:
        violations.append(
            f"path_loss {total_loss:.4f} > {profile.max_path_loss:.4f}"
        )
    if (
        profile.max_bottleneck_utilization is not None
        and bottleneck > profile.max_bottleneck_utilization
    ):
        violations.append(
            f"bottleneck {bottleneck:.3f} > {profile.max_bottleneck_utilization:.3f}"
        )
    if profile.max_hops is not None and hops > profile.max_hops:
        violations.append(f"hops {hops} > {profile.max_hops}")

    return QoSEvaluation(
        feasible=not violations,
        score=score,
        total_loss=total_loss,
        bottleneck_utilization=bottleneck,
        hops=hops,
        violations=violations,
    )


def select_best_path(
    state: NetworkState, paths: list[list[str]], profile: QoSProfile
) -> tuple[list[str], QoSEvaluation]:
    """Pick the best path for *profile*: feasible ones first, then by score.

    This is the QoS oracle over a candidate set. A learned policy is measured
    against it by constraint-satisfaction rate and by regret in score.
    """
    if not paths:
        return [], _INFEASIBLE

    scored = [(p, evaluate_path(state, p, profile)) for p in paths]
    feasible = [(p, e) for p, e in scored if e.feasible]
    pool = feasible or scored
    return min(pool, key=lambda item: item[1].score)


def _product(values) -> float:
    total = 1.0
    for value in values:
        total *= value
    return total


#: Width of the vector returned by :func:`profile_vector`.
PROFILE_VECTOR_DIM = 6


def profile_vector(profile: QoSProfile) -> list[float]:
    """Encode a profile as a fixed-width vector in [0, 1].

    This is what lets a *single* learned model serve all five traffic classes:
    the class is an input, not a separate model. Unconstrained axes are encoded
    as 1.0, meaning "no limit", which is the correct saturating value for a
    constraint expressed as an upper bound.
    """
    return [
        profile.w_latency,
        profile.w_loss,
        profile.w_utilization,
        1.0 if profile.max_path_loss is None else min(1.0, profile.max_path_loss / 0.05),
        1.0
        if profile.max_bottleneck_utilization is None
        else profile.max_bottleneck_utilization,
        1.0 if profile.max_hops is None else min(1.0, profile.max_hops / 10.0),
    ]


ALL_CLASSES: list[TrafficClass] = list(QOS_PROFILES.keys())
