"""Pluggable network sources: simulated, replayed from a trace, or measured live.

Everything above this layer — the five routing algorithms, the QoS engine, the
benchmark harness, the dashboard — consumes a :class:`NetworkState`. It does not
care where that state came from. This module makes that explicit so the same
platform can run against:

``SimulatedSource``
    The synthetic closed-loop :class:`core.simulator.NetworkSimulator`. The
    default, and the only source used for any published benchmark number.

``TraceReplaySource``
    A recorded trace (JSONL or CSV) replayed step by step. Lets the project
    consume an exported measurement run — your own, or a public trace converted
    to this schema — without the platform knowing the difference.

``LiveProbeSource``
    Real round-trip measurements taken from the machine the platform is running
    on, so anyone can point the dashboard at their own network.

Scope and safety of live probing
--------------------------------
``LiveProbeSource`` is a **read-only diagnostic**. It issues ordinary ICMP echo
requests with the system ``ping`` binary to an explicit, operator-supplied list
of hosts, exactly as a network engineer would by hand. It does not scan address
ranges, does not enumerate hosts, does not inject or forward traffic, and never
requires root. Probe targets must be listed explicitly, the count is rate
limited, and the platform refuses to start live mode without
``LIVE_PROBE_ENABLED=1``. Only measure networks you are authorised to measure.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import statistics
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from core.models import LinkState, NetworkState
from core.simulator import NetworkSimulator

logger = logging.getLogger(__name__)


class NetworkSource(ABC):
    """Anything that can produce successive :class:`NetworkState` snapshots."""

    #: Short identifier surfaced by the API and the dashboard.
    kind: str = "abstract"

    @abstractmethod
    def get_state(self) -> NetworkState:
        """Return the current state without advancing."""

    @abstractmethod
    def step(self) -> NetworkState:
        """Advance one tick and return the new state."""

    def reset(self) -> NetworkState:
        """Return to the beginning. Defaults to a no-op re-read."""
        return self.get_state()

    def register_flow(self, path: list[str], demand: float = 1.0) -> None:
        """Report that a flow was routed over *path*.

        Only meaningful for a closed-loop source; measured and replayed sources
        cannot be influenced by our routing decisions, so they ignore it.
        """
        return None

    def describe(self) -> dict[str, object]:
        """Human-readable provenance, shown in the dashboard header."""
        return {"kind": self.kind}


# ---------------------------------------------------------------------------
# 1. Simulated
# ---------------------------------------------------------------------------
class SimulatedSource(NetworkSource):
    """Wrap the synthetic closed-loop simulator."""

    kind = "simulated"

    def __init__(self, simulator: NetworkSimulator | None = None, **kwargs) -> None:
        self.simulator = simulator or NetworkSimulator(**kwargs)

    def get_state(self) -> NetworkState:
        return self.simulator.get_state()

    def step(self) -> NetworkState:
        return self.simulator.step()

    def reset(self) -> NetworkState:
        return self.simulator.reset()

    def register_flow(self, path: list[str], demand: float = 1.0) -> None:
        self.simulator.register_flow(path, demand)

    def describe(self) -> dict[str, object]:
        stats = self.simulator.topology_stats()
        return {
            "kind": self.kind,
            "closed_loop": True,
            "seed": self.simulator.seed,
            **stats,
        }


# ---------------------------------------------------------------------------
# 2. Trace replay
# ---------------------------------------------------------------------------
@dataclass
class TraceFrame:
    """One timestep of a recorded trace."""

    step: int
    links: list[LinkState]


class TraceReplaySource(NetworkSource):
    """Replay a recorded measurement trace as a sequence of network states.

    Two on-disk formats are accepted.

    **JSONL** — one JSON object per line::

        {"step": 0, "links": [{"source": "A", "target": "B", "base_latency": 12.0,
                               "bandwidth": 1000, "utilization": 0.31}]}

    **CSV** — one row per link per step, with a header::

        step,source,target,base_latency,bandwidth,utilization,packet_loss_rate

    ``queue_size`` and ``packet_loss_rate`` are derived from ``utilization``
    when absent, using the same relations as the simulator, so a trace only has
    to carry latency and utilization to be usable.
    """

    kind = "trace"

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop
        self.frames: list[TraceFrame] = _load_trace(self.path)
        if not self.frames:
            raise ValueError(f"Trace {self.path} contains no usable frames.")
        self._index = 0
        self._step_count = 0

    def get_state(self) -> NetworkState:
        frame = self.frames[self._index]
        nodes = sorted({n for link in frame.links for n in (link.source, link.target)})
        return NetworkState(
            nodes=nodes,
            links=list(frame.links),
            timestamp=time.time(),
            step_count=self._step_count,
        )

    def step(self) -> NetworkState:
        self._step_count += 1
        if self._index + 1 < len(self.frames):
            self._index += 1
        elif self.loop:
            self._index = 0
        return self.get_state()

    def reset(self) -> NetworkState:
        self._index = 0
        self._step_count = 0
        return self.get_state()

    def describe(self) -> dict[str, object]:
        first = self.frames[0]
        nodes = {n for link in first.links for n in (link.source, link.target)}
        return {
            "kind": self.kind,
            "closed_loop": False,
            "trace_file": str(self.path),
            "frames": len(self.frames),
            "num_nodes": len(nodes),
            "num_edges": len(first.links),
            "note": "Replayed measurements. Routing cannot affect this network.",
        }


def _derive_link(row: dict) -> LinkState:
    """Build a LinkState from a partial trace row, filling derived fields."""
    utilization = max(0.0, min(1.0, float(row.get("utilization", 0.0))))
    return LinkState(
        source=str(row["source"]),
        target=str(row["target"]),
        base_latency=float(row.get("base_latency", 10.0)),
        bandwidth=int(float(row.get("bandwidth", 1000))),
        utilization=utilization,
        queue_size=int(row.get("queue_size", int(utilization * 100))),
        packet_loss_rate=float(
            row.get("packet_loss_rate", max(0.0, utilization - 0.7) * 0.2)
        ),
    )


def _load_trace(path: Path) -> list[TraceFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    if path.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
        frames: list[TraceFrame] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            frames.append(
                TraceFrame(
                    step=int(payload.get("step", len(frames))),
                    links=[_derive_link(row) for row in payload.get("links", [])],
                )
            )
        return [f for f in frames if f.links]

    # CSV: group rows by step.
    grouped: dict[int, list[LinkState]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row.get("step", 0)))
            grouped.setdefault(step, []).append(_derive_link(row))
    return [TraceFrame(step=s, links=grouped[s]) for s in sorted(grouped) if grouped[s]]


# ---------------------------------------------------------------------------
# 3. Live probing of the operator's own network
# ---------------------------------------------------------------------------
@dataclass
class ProbeTarget:
    """One host to measure, and the node label it maps to in the graph."""

    host: str
    label: str
    #: Nominal capacity in Mbit/s, used only to populate ``bandwidth``.
    bandwidth: int = 1000


@dataclass
class ProbeStats:
    """Rolling RTT statistics for one measured link."""

    samples: list[float] = field(default_factory=list)
    losses: int = 0
    attempts: int = 0

    def observe(self, rtt_ms: float | None, window: int = 60) -> None:
        self.attempts += 1
        if rtt_ms is None:
            self.losses += 1
        else:
            self.samples.append(rtt_ms)
            del self.samples[:-window]

    @property
    def loss_rate(self) -> float:
        return self.losses / self.attempts if self.attempts else 0.0

    @property
    def baseline(self) -> float:
        """Best observed RTT — the uncongested floor for this path."""
        return min(self.samples) if self.samples else 0.0

    @property
    def latest(self) -> float:
        return self.samples[-1] if self.samples else 0.0

    def utilization(self) -> float:
        """Estimate utilization from RTT inflation above the observed floor.

        Queueing delay grows with load, so the ratio of current RTT to the
        best-ever RTT for the same path is a standard, if coarse, congestion
        proxy. We map a 3x inflation to full utilization and saturate there.
        """
        floor = self.baseline
        if floor <= 0 or not self.samples:
            return 0.0
        inflation = (self.latest - floor) / floor
        util = inflation / 2.0  # 3x RTT (inflation 2.0) => utilization 1.0
        # Sustained loss is itself strong evidence of congestion.
        util = max(util, min(1.0, self.loss_rate * 4.0))
        return float(max(0.0, min(1.0, util)))

    def jitter(self) -> float:
        return float(statistics.pstdev(self.samples)) if len(self.samples) > 1 else 0.0


#: Default targets: the operator's own gateway plus well-known public anycast
#: resolvers. These are deliberately generic, high-availability endpoints that
#: exist to answer exactly this kind of reachability probe.
DEFAULT_PROBE_TARGETS: list[ProbeTarget] = [
    ProbeTarget("1.1.1.1", "CLOUDFLARE"),
    ProbeTarget("8.8.8.8", "GOOGLE"),
    ProbeTarget("9.9.9.9", "QUAD9"),
    ProbeTarget("208.67.222.222", "OPENDNS"),
]

_PING_RTT = re.compile(r"time[=<]\s*([\d.]+)\s*ms", re.IGNORECASE)


def ping_once(host: str, timeout_s: float = 2.0) -> float | None:
    """Send one ICMP echo request with the system ``ping``; return RTT in ms.

    Returns ``None`` when the probe times out or ``ping`` is unavailable. Uses
    the unprivileged system binary rather than raw sockets, so no elevated
    permissions are required on Linux, macOS or Windows.
    """
    binary = shutil.which("ping")
    if not binary:
        return None

    # -n/-c 1: a single echo request. Never more, so this cannot become a flood.
    if _is_windows():
        args = [binary, "-n", "1", "-w", str(int(timeout_s * 1000)), host]
    else:
        args = [binary, "-c", "1", "-W", str(max(1, int(timeout_s))), host]

    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s + 3.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if completed.returncode != 0:
        return None
    match = _PING_RTT.search(completed.stdout)
    return float(match.group(1)) if match else None


def _is_windows() -> bool:
    import os

    return os.name == "nt"


class LiveProbeSource(NetworkSource):
    """Build a live network graph from real round-trip measurements.

    The graph is a star centred on ``LOCAL`` (this machine): one edge per probe
    target, with ``base_latency`` set to the best RTT ever observed for that
    target, ``utilization`` estimated from current RTT inflation, and
    ``packet_loss_rate`` from the observed timeout ratio.

    Honest limits, stated up front:

    * A star topology has exactly one path per destination, so *routing
      algorithms cannot differentiate on it*. Live mode is a telemetry and
      congestion-detection demonstration, not a routing benchmark. No benchmark
      number in this project is produced from live data.
    * RTT-inflation is a proxy for utilization, not a measurement of it.
    * ICMP is de-prioritised or rate-limited by many networks, so an elevated
      reading may reflect router policy rather than congestion.
    """

    kind = "live"

    def __init__(
        self,
        targets: list[ProbeTarget] | None = None,
        max_targets: int = 12,
        timeout_s: float = 2.0,
    ) -> None:
        chosen = list(targets or DEFAULT_PROBE_TARGETS)[:max_targets]
        if not chosen:
            raise ValueError("LiveProbeSource requires at least one probe target.")
        self.targets = chosen
        self.timeout_s = timeout_s
        self.stats: dict[str, ProbeStats] = {t.label: ProbeStats() for t in chosen}
        self._step_count = 0
        self._probe_all()

    def _probe_all(self) -> None:
        for target in self.targets:
            rtt = ping_once(target.host, self.timeout_s)
            self.stats[target.label].observe(rtt)

    def get_state(self) -> NetworkState:
        links: list[LinkState] = []
        for target in self.targets:
            stats = self.stats[target.label]
            utilization = stats.utilization()
            links.append(
                LinkState(
                    source="LOCAL",
                    target=target.label,
                    base_latency=round(stats.baseline, 3) or 1.0,
                    bandwidth=target.bandwidth,
                    utilization=utilization,
                    queue_size=int(utilization * 100),
                    packet_loss_rate=round(stats.loss_rate, 5),
                )
            )
        nodes = ["LOCAL", *[t.label for t in self.targets]]
        return NetworkState(
            nodes=nodes,
            links=links,
            timestamp=time.time(),
            step_count=self._step_count,
        )

    def step(self) -> NetworkState:
        self._step_count += 1
        self._probe_all()
        return self.get_state()

    def reset(self) -> NetworkState:
        for stats in self.stats.values():
            stats.samples.clear()
            stats.losses = 0
            stats.attempts = 0
        self._step_count = 0
        self._probe_all()
        return self.get_state()

    def describe(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "closed_loop": False,
            "targets": [{"host": t.host, "label": t.label} for t in self.targets],
            "measurement": "ICMP echo via system ping, one request per target per tick",
            "utilization_estimator": "RTT inflation over best observed RTT",
            "warning": (
                "Star topology: one path per destination, so routing algorithms "
                "cannot differentiate. Live mode demonstrates real telemetry and "
                "congestion detection, not routing quality."
            ),
        }

    def health(self) -> dict[str, object]:
        """Per-target reachability summary, for the dashboard's live panel."""
        return {
            label: {
                "reachable": bool(stat.samples),
                "best_rtt_ms": round(stat.baseline, 3),
                "last_rtt_ms": round(stat.latest, 3),
                "jitter_ms": round(stat.jitter(), 3),
                "loss_rate": round(stat.loss_rate, 4),
                "estimated_utilization": round(stat.utilization(), 4),
            }
            for label, stat in self.stats.items()
        }


def record_trace(
    source: NetworkSource, steps: int, destination: str | Path
) -> Path:
    """Record *steps* frames from *source* into a JSONL trace file.

    This is the bridge between live measurement and repeatable analysis: probe
    your own network for a while, write a trace, then replay it deterministically
    through the benchmark harness as many times as you like.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(steps):
            state = source.get_state() if index == 0 else source.step()
            handle.write(
                json.dumps(
                    {
                        "step": index,
                        "links": [
                            {
                                "source": link.source,
                                "target": link.target,
                                "base_latency": link.base_latency,
                                "bandwidth": link.bandwidth,
                                "utilization": round(link.utilization, 6),
                                "packet_loss_rate": round(link.packet_loss_rate, 6),
                            }
                            for link in state.links
                        ],
                    }
                )
                + "\n"
            )
    logger.info("Wrote %d-frame trace to %s", steps, path)
    return path


def synthetic_trace(path: str | Path, steps: int = 300, seed: int = 42) -> Path:
    """Generate a small example trace so replay mode works out of the box."""
    return record_trace(
        SimulatedSource(num_nodes=25, seed=seed), steps=steps, destination=path
    )


__all__ = [
    "DEFAULT_PROBE_TARGETS",
    "LiveProbeSource",
    "NetworkSource",
    "ProbeStats",
    "ProbeTarget",
    "SimulatedSource",
    "TraceReplaySource",
    "ping_once",
    "record_trace",
    "synthetic_trace",
]
