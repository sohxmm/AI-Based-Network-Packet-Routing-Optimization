"""Experiment Sandbox API — user-configured benchmark runs with hard caps.

This was already the best-engineered file in the backend: Pydantic
``field_validator`` and ``model_validator``, hard caps that *reject rather than
silently clamp*, and an error message telling the user exactly which parameter
to reduce. That is all kept.

Two things are fixed:

* **The run no longer blocks the event loop.** It used to ``await`` CPU-bound
  benchmark work directly, yielding only every ten steps, so running an
  experiment froze the live WebSocket stream. It is now dispatched to a thread.
* **The job store is bounded.** It was an unbounded module-level dict, so a
  long-lived process leaked one entry per experiment forever.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from core.qos import ALL_CLASSES
from routing import ALGORITHM_NAMES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])

# -- hard caps (non-negotiable) ---------------------------------------------
MAX_STEPS = 300
MAX_PAIRS_PER_STEP = 10
MAX_RUNS = 10
MAX_TOTAL_DECISIONS = 6000

VALID_TOPOLOGY_SIZES = [25, 50, 100]
VALID_ALGORITHMS = list(ALGORITHM_NAMES)
VALID_TRAFFIC_CLASSES = [c.value for c in ALL_CLASSES]

# -- bounded in-memory job store --------------------------------------------
MAX_JOBS = 50
_jobs: dict[str, dict[str, Any]] = {}


def _evict_old_jobs() -> None:
    """Keep the job store bounded, dropping the oldest finished jobs first."""
    if len(_jobs) <= MAX_JOBS:
        return
    finished = sorted(
        (jid for jid, job in _jobs.items() if job["state"] in ("done", "failed")),
        key=lambda jid: _jobs[jid]["created_at"],
    )
    for job_id in finished[: len(_jobs) - MAX_JOBS]:
        _jobs.pop(job_id, None)


class ExperimentConfig(BaseModel):
    """A validated experiment configuration."""

    topology_size: int = Field(..., description="Number of nodes: 25, 50 or 100")
    congestion_profile: Literal["normal", "high", "bursty"] = "normal"
    failure_rate: float = Field(0.0, ge=0.0, le=30.0, description="Link failure rate, %")
    failure_pattern: Literal["none", "random", "targeted"] = "none"
    steps: int = Field(..., gt=0, le=MAX_STEPS)
    pairs_per_step: int = Field(..., gt=0, le=MAX_PAIRS_PER_STEP)
    runs: int = Field(3, gt=0, le=MAX_RUNS, description="Independent seeded replications")
    algorithms: list[str] = Field(default_factory=lambda: list(VALID_ALGORITHMS))
    traffic_classes: list[str] = Field(default_factory=lambda: ["best_effort"])

    @field_validator("topology_size")
    @classmethod
    def validate_topology_size(cls, value: int) -> int:
        if value not in VALID_TOPOLOGY_SIZES:
            raise ValueError(
                f"topology_size must be one of {VALID_TOPOLOGY_SIZES}, got {value}"
            )
        return value

    @field_validator("algorithms")
    @classmethod
    def validate_algorithms(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one algorithm must be selected")
        invalid = [a for a in value if a not in VALID_ALGORITHMS]
        if invalid:
            raise ValueError(f"Invalid algorithms: {invalid}. Valid: {VALID_ALGORITHMS}")
        return value

    @field_validator("traffic_classes")
    @classmethod
    def validate_traffic_classes(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one traffic class must be selected")
        invalid = [c for c in value if c not in VALID_TRAFFIC_CLASSES]
        if invalid:
            raise ValueError(
                f"Invalid traffic classes: {invalid}. Valid: {VALID_TRAFFIC_CLASSES}"
            )
        return value

    @model_validator(mode="after")
    def validate_total_decisions(self) -> ExperimentConfig:
        total = self.steps * self.pairs_per_step * self.runs * len(self.algorithms)
        if total > MAX_TOTAL_DECISIONS * len(self.algorithms):
            budget = MAX_TOTAL_DECISIONS
            raise ValueError(
                f"steps x pairs_per_step x runs = "
                f"{self.steps} x {self.pairs_per_step} x {self.runs} = "
                f"{self.steps * self.pairs_per_step * self.runs} decisions per "
                f"algorithm, which exceeds the cap of {budget}. Reduce steps, "
                f"pairs_per_step or runs so their product stays at or below {budget}."
            )
        return self


async def _run_experiment(job_id: str, config: ExperimentConfig) -> None:
    """Execute an experiment off the event loop."""
    job = _jobs[job_id]
    job["state"] = "running"

    try:
        from experiments.runner import run_parameterized_scenario

        def on_progress(completed: int, total: int) -> None:
            job["progress"] = {"runs_completed": completed, "total": total}

        loop = asyncio.get_running_loop()
        work = functools.partial(
            run_parameterized_scenario,
            topology_size=config.topology_size,
            congestion_profile=config.congestion_profile,
            failure_rate=config.failure_rate / 100.0,
            failure_pattern=config.failure_pattern,
            steps=config.steps,
            pairs_per_step=config.pairs_per_step,
            algorithms=config.algorithms,
            traffic_classes=config.traffic_classes,
            n_runs=config.runs,
            on_progress=on_progress,
        )
        # A thread, not the event loop: the dashboard must keep streaming while
        # an experiment runs.
        job["result"] = await loop.run_in_executor(None, work)
        job["state"] = "done"

    except Exception as exc:  # noqa: BLE001 - report, never crash the server
        logger.exception("Experiment %s failed", job_id)
        job["state"] = "failed"
        job["error"] = str(exc)


@router.post("")
async def create_experiment(
    config: ExperimentConfig, background_tasks: BackgroundTasks
) -> dict:
    """Submit an experiment. Returns a job id to poll."""
    _evict_old_jobs()
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "state": "queued",
        "progress": {"runs_completed": 0, "total": config.runs},
        "result": None,
        "error": None,
        "config": config.model_dump(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    background_tasks.add_task(_run_experiment, job_id, config)
    return {"job_id": job_id}


@router.get("/{job_id}/status")
def get_experiment_status(job_id: str) -> dict:
    """Poll an experiment's progress."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")
    return {"state": job["state"], "progress": job["progress"], "error": job["error"]}


@router.get("/{job_id}/results")
def get_experiment_results(job_id: str) -> dict:
    """Fetch results once the experiment has finished."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")
    if job["state"] == "failed":
        raise HTTPException(status_code=500, detail=f"Experiment failed: {job['error']}")
    if job["state"] != "done":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Experiment is still {job['state']}. "
                f"Poll /experiments/{job_id}/status for progress."
            ),
        )
    return job["result"]


__all__ = ["ExperimentConfig", "router"]
