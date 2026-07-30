"""Experiment Sandbox API — user-configured benchmark runs with hard caps."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

router = APIRouter(prefix="/experiments", tags=["experiments"])
logger = logging.getLogger("experiments")

# ── Hard caps (non-negotiable) ──────────────────────────────────────────────
MAX_STEPS = 300
MAX_PAIRS_PER_STEP = 10
MAX_TOTAL_DECISIONS = 3000

VALID_TOPOLOGY_SIZES = [25, 50, 100]
VALID_CONGESTION_PROFILES = ["normal", "high", "bursty"]
VALID_FAILURE_PATTERNS = ["none", "random", "targeted"]
VALID_ALGORITHMS = [
    "dijkstra", "bellman_ford", "aco", "gnn", "gnn_predictive",
    "rl", "rl_predictive", "multi_agent",
]

# ── In-memory job store ─────────────────────────────────────────────────────
# Maps job_id → {state, progress, result, error, config, created_at}
_jobs: dict[str, dict[str, Any]] = {}


class ExperimentConfig(BaseModel):
    """Validated experiment configuration from the user."""

    topology_size: int = Field(..., description="Number of nodes: 25, 50, or 100")
    congestion_profile: Literal["normal", "high", "bursty"] = "normal"
    failure_rate: float = Field(0.0, ge=0.0, le=30.0, description="Link failure rate 0-30%")
    failure_pattern: Literal["none", "random", "targeted"] = "none"
    steps: int = Field(..., gt=0, le=MAX_STEPS, description=f"Simulation steps, max {MAX_STEPS}")
    pairs_per_step: int = Field(..., gt=0, le=MAX_PAIRS_PER_STEP, description=f"Src/dst pairs per step, max {MAX_PAIRS_PER_STEP}")
    algorithms: list[str] = Field(default_factory=lambda: list(VALID_ALGORITHMS))

    @field_validator("topology_size")
    @classmethod
    def validate_topology_size(cls, v: int) -> int:
        if v not in VALID_TOPOLOGY_SIZES:
            raise ValueError(f"topology_size must be one of {VALID_TOPOLOGY_SIZES}, got {v}")
        return v

    @field_validator("algorithms")
    @classmethod
    def validate_algorithms(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one algorithm must be selected")
        invalid = [a for a in v if a not in VALID_ALGORITHMS]
        if invalid:
            raise ValueError(f"Invalid algorithms: {invalid}. Valid: {VALID_ALGORITHMS}")
        return v

    @model_validator(mode="after")
    def validate_total_decisions(self) -> "ExperimentConfig":
        total = self.steps * self.pairs_per_step
        if total > MAX_TOTAL_DECISIONS:
            raise ValueError(
                f"Total decisions (steps × pairs_per_step = {self.steps} × {self.pairs_per_step} = {total}) "
                f"exceeds hard cap of {MAX_TOTAL_DECISIONS}. "
                f"Reduce steps or pairs_per_step so their product does not exceed {MAX_TOTAL_DECISIONS}."
            )
        return self


async def _run_experiment(job_id: str, config: ExperimentConfig) -> None:
    """Execute a benchmark experiment in the background."""
    job = _jobs[job_id]
    job["state"] = "running"

    try:
        # Import here to avoid circular imports and keep module load fast
        from benchmark.run_benchmark import run_parameterized_scenario

        def on_progress(steps_completed: int, total: int) -> None:
            job["progress"] = {"steps_completed": steps_completed, "total": total}

        result = await run_parameterized_scenario(
            topology_size=config.topology_size,
            congestion_profile=config.congestion_profile,
            failure_rate=config.failure_rate / 100.0,  # Convert percentage to fraction
            failure_pattern=config.failure_pattern,
            steps=config.steps,
            pairs_per_step=config.pairs_per_step,
            algorithms=config.algorithms,
            on_progress=on_progress,
        )

        job["state"] = "done"
        job["result"] = result

    except Exception as exc:
        logger.exception(f"Experiment {job_id} failed")
        job["state"] = "failed"
        job["error"] = str(exc)


@router.post("")
async def create_experiment(config: ExperimentConfig) -> dict:
    """Submit a new experiment job.

    Hard caps enforced server-side:
    - steps ≤ 300
    - pairs_per_step ≤ 10
    - steps × pairs_per_step ≤ 3000 (REJECTED, not clamped)

    Returns a job_id for polling status and retrieving results.
    """
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "state": "queued",
        "progress": {"steps_completed": 0, "total": config.steps},
        "result": None,
        "error": None,
        "config": config.model_dump(),
        "created_at": datetime.utcnow().isoformat(),
    }

    # Launch the job as a background task
    asyncio.create_task(_run_experiment(job_id, config))

    return {"job_id": job_id}


@router.get("/{job_id}/status")
def get_experiment_status(job_id: str) -> dict:
    """Check experiment job status and progress."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")

    job = _jobs[job_id]
    return {
        "state": job["state"],
        "progress": job["progress"],
        "error": job["error"],
    }


@router.get("/{job_id}/results")
def get_experiment_results(job_id: str) -> dict:
    """Retrieve experiment results once completed.

    Returns same shape as GET /benchmark/results/{scenario}.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Experiment {job_id} not found")

    job = _jobs[job_id]

    if job["state"] == "failed":
        raise HTTPException(status_code=500, detail=f"Experiment failed: {job['error']}")

    if job["state"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Experiment is still {job['state']}. Poll /experiments/{job_id}/status for progress.",
        )

    return job["result"]
