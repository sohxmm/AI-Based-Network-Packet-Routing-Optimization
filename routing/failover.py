"""Fault-tolerant rerouting and convergence measurement.

Problem statement section 17C asks the system to "automatically reroute packets
during node failure, link failure and heavy congestion". Doing that is easy —
every router already recomputes from the current state. The part worth building
is the *measurement*: when a link on an active route dies, how quickly does each
algorithm restore service, and at what cost?

The metric this module produces is **convergence time in simulator ticks**: the
number of steps between the failure and the first tick at which the algorithm
again returns a working, QoS-satisfying route for the affected demand. A
classical shortest-path recomputes instantly but may land on an already-hot
link; an adaptive policy may take a tick longer and land somewhere that holds.
Reporting both the delay and the post-failure path quality is what makes the
comparison meaningful rather than a race everyone ties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.models import NetworkState
from core.qos import QoSProfile, evaluate_path, get_profile
from core.simulator import NetworkSimulator
from routing.base import Router

logger = logging.getLogger(__name__)


@dataclass
class ActiveFlow:
    """A demand the failover monitor is keeping alive."""

    source: str
    destination: str
    traffic_class: str = "best_effort"
    path: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.destination, self.traffic_class)


@dataclass
class RerouteEvent:
    """One automatic reroute, recorded for the dashboard and the report."""

    step: int
    source: str
    destination: str
    traffic_class: str
    reason: str
    old_path: list[str]
    new_path: list[str]
    recovered: bool
    convergence_steps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "source": self.source,
            "destination": self.destination,
            "traffic_class": self.traffic_class,
            "reason": self.reason,
            "old_path": list(self.old_path),
            "new_path": list(self.new_path),
            "recovered": self.recovered,
            "convergence_steps": self.convergence_steps,
        }


def path_is_intact(state: NetworkState, path: list[str]) -> bool:
    """True when every hop of *path* still exists in *state*."""
    if len(path) < 2:
        return False
    present = {frozenset((link.source, link.target)) for link in state.links}
    return all(
        frozenset((path[i], path[i + 1])) in present for i in range(len(path) - 1)
    )


class FailoverMonitor:
    """Watch a set of flows and reroute them when their path breaks or degrades.

    This is the live-demo half of the feature: the dashboard registers the flows
    the operator is watching, the monitor checks them on every simulator tick,
    and any reroute is pushed to the UI as an event.
    """

    def __init__(self, router: Router, max_events: int = 100) -> None:
        self.router = router
        self.flows: dict[tuple[str, str, str], ActiveFlow] = {}
        self.events: list[RerouteEvent] = []
        self.max_events = max_events

    def watch(self, source: str, destination: str, traffic_class: str = "best_effort") -> ActiveFlow:
        """Start monitoring a demand."""
        flow = ActiveFlow(source, destination, traffic_class)
        self.flows[flow.key] = flow
        return flow

    def unwatch(self, source: str, destination: str, traffic_class: str = "best_effort") -> None:
        self.flows.pop((source, destination, traffic_class), None)

    def clear(self) -> None:
        self.flows.clear()
        self.events.clear()

    def tick(self, state: NetworkState) -> list[RerouteEvent]:
        """Check every watched flow, rerouting the ones that need it."""
        triggered: list[RerouteEvent] = []

        for flow in list(self.flows.values()):
            profile = get_profile(flow.traffic_class)
            reason = self._needs_reroute(state, flow, profile)
            if reason is None:
                continue

            decision = self.router.find_route(
                state, flow.source, flow.destination, profile
            )
            event = RerouteEvent(
                step=state.step_count,
                source=flow.source,
                destination=flow.destination,
                traffic_class=flow.traffic_class,
                reason=reason,
                old_path=list(flow.path),
                new_path=list(decision.path),
                recovered=decision.success,
                convergence_steps=0 if decision.success else -1,
            )
            flow.path = list(decision.path) if decision.success else []
            triggered.append(event)
            self.events.append(event)

        del self.events[: max(0, len(self.events) - self.max_events)]
        return triggered

    def _needs_reroute(
        self, state: NetworkState, flow: ActiveFlow, profile: QoSProfile
    ) -> str | None:
        """Return why *flow* must be rerouted, or None if it is healthy."""
        if not flow.path:
            return "no_route"
        if not path_is_intact(state, flow.path):
            return "link_failure"
        if not evaluate_path(state, flow.path, profile).feasible:
            return "qos_violation"
        return None

    def snapshot(self) -> dict[str, object]:
        """Current state of the monitor, for ``GET /network/failover``."""
        return {
            "watched": [
                {
                    "source": flow.source,
                    "destination": flow.destination,
                    "traffic_class": flow.traffic_class,
                    "path": list(flow.path),
                }
                for flow in self.flows.values()
            ],
            "events": [event.as_dict() for event in self.events[-20:]],
        }


def measure_convergence(
    simulator: NetworkSimulator,
    router: Router,
    source: str,
    destination: str,
    failed_link: tuple[str, str],
    traffic_class: str = "best_effort",
    max_steps: int = 20,
) -> dict[str, object]:
    """Fail a link on an active route and measure how fast *router* recovers.

    Returns the number of ticks to restore a QoS-satisfying route, along with
    the path cost before and after, so a fast-but-worse recovery is
    distinguishable from a slow-but-better one.
    """
    profile = get_profile(traffic_class)
    state = simulator.get_state()
    before = router.find_route(state, source, destination, profile)

    if not before.success:
        return {
            "converged": False,
            "convergence_steps": None,
            "reason": "no_route_before_failure",
        }

    try:
        simulator.inject_failure(*failed_link)
    except ValueError as exc:
        return {"converged": False, "convergence_steps": None, "reason": str(exc)}

    converged_at: int | None = None
    after = None
    for step in range(1, max_steps + 1):
        state = simulator.step()
        after = router.find_route(state, source, destination, profile)
        if after.success and evaluate_path(state, after.path, profile).feasible:
            converged_at = step
            break

    return {
        "converged": converged_at is not None,
        "convergence_steps": converged_at,
        "failed_link": list(failed_link),
        "algorithm": router.name,
        "traffic_class": traffic_class,
        "latency_before": None if before.total_latency == float("inf") else before.total_latency,
        "latency_after": (
            None
            if after is None or after.total_latency == float("inf")
            else after.total_latency
        ),
        "path_before": list(before.path),
        "path_after": list(after.path) if after else [],
    }


__all__ = [
    "ActiveFlow",
    "FailoverMonitor",
    "RerouteEvent",
    "measure_convergence",
    "path_is_intact",
]
