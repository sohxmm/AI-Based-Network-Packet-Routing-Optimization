"""Single source of truth for trained model artifact locations.

The worst bug in the previous version was a filename mismatch: the RL router
loaded ``ml/models/rl_router_final.zip`` while training saved — and the repo
shipped — ``ppo_routing_agent.zip``. ``try_load_model()`` caught the resulting
``FileNotFoundError`` and returned ``False`` with no logging, so the "RL router"
silently served a Dijkstra-equivalent heuristic for the life of the project.
The same class of bug hid the missing MARL and LSTM artifacts.

Three modules each hardcoding their own path is what made that possible. Paths
now live here and nowhere else, training and serving both read them from this
registry, and a missing artifact is logged loudly at startup and exposed on
``GET /health/models``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ModelSpec:
    """Everything the system needs to know about one trained artifact."""

    key: str
    filename: str
    train_command: str
    ships_in_repo: bool
    description: str


REGISTRY: dict[str, ModelSpec] = {
    "gnn": ModelSpec(
        key="gnn",
        filename="gnn_router.pt",
        train_command="python -m ml.training.train_gnn",
        ships_in_repo=True,
        description="Graph neural network path ranker",
    ),
    "rl": ModelSpec(
        key="rl",
        filename="ppo_routing_agent.zip",
        train_command="python -m ml.training.train_rl",
        ships_in_repo=True,
        description="PPO single-agent routing policy",
    ),
    "lstm": ModelSpec(
        key="lstm",
        filename="congestion_lstm.pt",
        train_command="python -m ml.training.train_lstm",
        ships_in_repo=True,
        description="LSTM link-utilization forecaster",
    ),
    "multi_agent": ModelSpec(
        key="multi_agent",
        filename="multi_agent_region_0.zip",
        train_command="python -m ml.training.train_regional",
        ships_in_repo=True,
        description="Decentralized next-hop PPO policies, one per region",
    ),
}


def path_for(key: str) -> Path:
    """Absolute path of the artifact registered under *key*."""
    return MODEL_DIR / REGISTRY[key].filename


def regional_path(region_id: int) -> Path:
    """Absolute path of the policy for region *region_id*."""
    return MODEL_DIR / f"multi_agent_region_{region_id}.zip"


def regional_paths() -> list[Path]:
    """Every regional policy currently present on disk."""
    return sorted(MODEL_DIR.glob("multi_agent_region_*.zip"))


def missing_models() -> list[ModelSpec]:
    """Registered artifacts that are not on disk."""
    return [spec for spec in REGISTRY.values() if not path_for(spec.key).exists()]


def log_model_inventory() -> dict[str, bool]:
    """Log which artifacts are present. Called once at application startup.

    A model marked ``ships_in_repo=True`` that is missing is a *bug* and logs at
    WARNING. One marked False is an expected fresh-clone state and logs at INFO.
    """
    present: dict[str, bool] = {}
    for key, spec in REGISTRY.items():
        exists = path_for(key).exists()
        present[key] = exists
        if exists:
            logger.info("Model %-12s : present (%s)", key, spec.filename)
        else:
            emit = logger.warning if spec.ships_in_repo else logger.info
            emit(
                "Model %-12s : MISSING (%s). %s will use its heuristic fallback. "
                "Train with: %s",
                key,
                spec.filename,
                spec.description,
                spec.train_command,
            )
    if not any(present.values()):
        logger.warning(
            "No trained models found. Every AI router is running a "
            "congestion-aware heuristic, so results will not reflect learned "
            "behaviour. Run `make train` to fix this."
        )
    return present


def inventory() -> dict[str, dict[str, object]]:
    """Machine-readable artifact inventory for the diagnostics endpoint."""
    return {
        key: {
            "file": spec.filename,
            "file_present": path_for(key).exists(),
            "expected_in_repo": spec.ships_in_repo,
            "train_command": spec.train_command,
            "description": spec.description,
        }
        for key, spec in REGISTRY.items()
    }


__all__ = [
    "MODEL_DIR",
    "REGISTRY",
    "RESULTS_DIR",
    "ModelSpec",
    "inventory",
    "log_model_inventory",
    "missing_models",
    "path_for",
    "regional_path",
    "regional_paths",
]
