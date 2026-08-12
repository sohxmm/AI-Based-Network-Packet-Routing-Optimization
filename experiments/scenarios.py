"""Declarative benchmark scenarios.

Each scenario is a dataclass with a ``prepare`` hook that runs once and a
``per_step`` hook that runs every tick. The old harness used one imperative
``apply_scenario()`` function with a chain of ``if`` statements, and three of its
five scenarios destroyed the signal they existed to measure:

``high_congestion`` **self-destructed.** It added +0.4 to 30% of links *every
step, cumulatively, with no reset*. Within about ten steps essentially every
link pinned at 1.0. Once all utilizations are equal the cost function degenerates
to ``5 x base_latency`` for every link and the ranking collapses to pure
base-latency order — the scenario erased the congestion differences it was built
to test. It now sets a *sustained bias* once, and the simulator's AR(1) dynamics
mean-revert toward it, so hot links stay hot without everything saturating.

``link_failures_5_10pct`` **had zero discriminative power.** It restored all
failed edges and failed a fresh random 5-10% on *every one of 1,000 steps* — a
different topology every tick — and ``success_rate`` was 1.000 for every
algorithm. Failures are now persistent, and an edge whose removal would
disconnect the graph is skipped and counted.

``large_topology_100_nodes`` **was a ring.** Degree 2, diameter 50, exactly two
simple paths between any pair, so no algorithm could differentiate and every one
scored ~910. Fixed in the topology generator; it is now degree 4, diameter ~8,
and honestly labelled a scale test.

Two scenarios are new. ``cascading_failure`` progressively removes the
highest-betweenness edge, so the optimal route changes *structurally* over the
run — the regime where adaptive routing has a real reason to beat static
shortest-path. ``qos_mixed_traffic`` drives all five traffic classes at once,
which is where a constraint-aware policy can beat a constraint-blind one.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import networkx as nx

from core.qos import ALL_CLASSES, TrafficClass
from core.simulator import NetworkSimulator

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """One reproducible experimental condition."""

    name: str
    description: str
    num_nodes: int = 25
    #: Traffic classes to draw from. Best-effort only reproduces the original study.
    traffic_classes: list[TrafficClass] = field(
        default_factory=lambda: [TrafficClass.BEST_EFFORT]
    )
    notes: str = ""

    def prepare(self, sim: NetworkSimulator, rng: random.Random) -> dict:
        """Run once before the replicate starts. Returns scenario metadata."""
        return {}

    def per_step(self, sim: NetworkSimulator, rng: random.Random, step: int) -> None:
        """Run at every tick."""
        return None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "num_nodes": self.num_nodes,
            "traffic_classes": [c.value for c in self.traffic_classes],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
@dataclass
class NormalTraffic(Scenario):
    name: str = "normal_traffic"
    description: str = "Baseline. Simulator dynamics only, no injected stress."


# ---------------------------------------------------------------------------
@dataclass
class HighCongestion(Scenario):
    name: str = "high_congestion"
    description: str = (
        "30% of links carry a sustained elevated load. Set once, not accumulated, "
        "so hot links stay hot without the whole network saturating."
    )
    hot_fraction: float = 0.30
    bias: float = 0.35

    def prepare(self, sim: NetworkSimulator, rng: random.Random) -> dict:
        edges = list(sim.graph.edges())
        count = max(1, int(self.hot_fraction * len(edges)))
        hot = rng.sample(edges, k=min(count, len(edges)))
        for u, v in hot:
            data = sim.graph[u][v]
            # congestion_bias raises the *offered load* this link mean-reverts
            # toward, rather than adding to utilization every tick forever.
            data["congestion_bias"] = self.bias
            data["utilization"] = min(0.95, data["utilization"] + self.bias)
        return {"hot_links": len(hot), "bias": self.bias}


# ---------------------------------------------------------------------------
@dataclass
class PersistentLinkFailures(Scenario):
    name: str = "link_failures_persistent"
    description: str = (
        "8% of links fail once and stay failed, skipping any edge whose removal "
        "would disconnect the graph."
    )
    failure_fraction: float = 0.08

    def prepare(self, sim: NetworkSimulator, rng: random.Random) -> dict:
        edges = list(sim.graph.edges())
        rng.shuffle(edges)
        target = max(1, int(self.failure_fraction * len(edges)))

        failed = skipped = 0
        for u, v in edges:
            if failed >= target:
                break
            if not sim.can_fail_safely(u, v):
                skipped += 1
                continue
            sim.inject_failure(u, v)
            failed += 1

        logger.debug("Failed %d links, skipped %d that would disconnect", failed, skipped)
        return {"links_failed": failed, "links_skipped_to_stay_connected": skipped}


# ---------------------------------------------------------------------------
@dataclass
class CascadingFailure(Scenario):
    name: str = "cascading_failure"
    description: str = (
        "Every 25 steps the highest-betweenness edge fails permanently. The "
        "network degrades progressively and the optimal route changes structure, "
        "which is the regime where adaptivity should pay off."
    )
    interval: int = 25
    max_failures: int = 6

    def prepare(self, sim: NetworkSimulator, rng: random.Random) -> dict:
        self._failures = 0
        return {"interval": self.interval, "max_failures": self.max_failures}

    def per_step(self, sim: NetworkSimulator, rng: random.Random, step: int) -> None:
        if step == 0 or step % self.interval != 0:
            return
        if getattr(self, "_failures", 0) >= self.max_failures:
            return

        try:
            centrality = nx.edge_betweenness_centrality(sim.graph, k=min(12, sim.graph.number_of_nodes()), seed=1)
        except Exception:  # noqa: BLE001 - small graphs can raise here
            return

        for (u, v), _ in sorted(centrality.items(), key=lambda kv: -kv[1]):
            if sim.can_fail_safely(u, v):
                sim.inject_failure(u, v)
                self._failures = getattr(self, "_failures", 0) + 1
                logger.debug("Cascading failure %d: %s-%s at step %d", self._failures, u, v, step)
                return


# ---------------------------------------------------------------------------
@dataclass
class CongestionBursts(Scenario):
    name: str = "congestion_bursts"
    description: str = (
        "Three links burst simultaneously for 3-10 steps with p=0.1 per step. "
        "The old version congested one link at a time, which is trivially "
        "routable around."
    )
    burst_probability: float = 0.10
    burst_links: int = 3

    def prepare(self, sim: NetworkSimulator, rng: random.Random) -> dict:
        self._remaining = 0
        self._active: list[tuple[str, str]] = []
        return {"burst_links": self.burst_links}

    def per_step(self, sim: NetworkSimulator, rng: random.Random, step: int) -> None:
        if getattr(self, "_remaining", 0) > 0:
            self._remaining -= 1
            if self._remaining == 0:
                for u, v in getattr(self, "_active", []):
                    if sim.graph.has_edge(u, v):
                        sim.graph[u][v]["congestion_bias"] = 0.0
                self._active = []
            return

        if rng.random() >= self.burst_probability:
            return

        edges = list(sim.graph.edges())
        if not edges:
            return
        chosen = rng.sample(edges, k=min(self.burst_links, len(edges)))
        for u, v in chosen:
            sim.graph[u][v]["congestion_bias"] = 0.55
        self._active = chosen
        self._remaining = rng.randint(3, 10)


# ---------------------------------------------------------------------------
@dataclass
class LargeTopology(Scenario):
    name: str = "large_topology_100_nodes"
    num_nodes: int = 100
    description: str = (
        "Scale test: 100 nodes, 200 links, average degree 4. Previously a "
        "degree-2 ring, where no algorithm could differentiate."
    )


# ---------------------------------------------------------------------------
@dataclass
class QoSMixedTraffic(Scenario):
    name: str = "qos_mixed_traffic"
    description: str = (
        "All five traffic classes at once under moderate congestion. The "
        "headline metric here is constraint satisfaction rate, not latency: "
        "Dijkstra optimises an additive cost and cannot express a constraint."
    )
    traffic_classes: list[TrafficClass] = field(default_factory=lambda: list(ALL_CLASSES))
    hot_fraction: float = 0.25
    bias: float = 0.30

    def prepare(self, sim: NetworkSimulator, rng: random.Random) -> dict:
        edges = list(sim.graph.edges())
        count = max(1, int(self.hot_fraction * len(edges)))
        for u, v in rng.sample(edges, k=min(count, len(edges))):
            sim.graph[u][v]["congestion_bias"] = self.bias
        return {"hot_links": count, "classes": [c.value for c in self.traffic_classes]}


# ---------------------------------------------------------------------------
SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        NormalTraffic(name="normal_traffic", description=NormalTraffic.description),
        HighCongestion(name="high_congestion", description=HighCongestion.description),
        PersistentLinkFailures(
            name="link_failures_persistent", description=PersistentLinkFailures.description
        ),
        CascadingFailure(name="cascading_failure", description=CascadingFailure.description),
        CongestionBursts(name="congestion_bursts", description=CongestionBursts.description),
        LargeTopology(
            name="large_topology_100_nodes", description=LargeTopology.description
        ),
        QoSMixedTraffic(name="qos_mixed_traffic", description=QoSMixedTraffic.description),
    )
}

SCENARIO_NAMES: list[str] = list(SCENARIOS)


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name, with a helpful error listing the options."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}. Available: {SCENARIO_NAMES}")
    return SCENARIOS[name]


__all__ = [
    "SCENARIOS",
    "SCENARIO_NAMES",
    "CascadingFailure",
    "CongestionBursts",
    "HighCongestion",
    "LargeTopology",
    "NormalTraffic",
    "PersistentLinkFailures",
    "QoSMixedTraffic",
    "Scenario",
    "get_scenario",
]
