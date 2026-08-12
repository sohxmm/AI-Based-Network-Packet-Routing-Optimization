"""Application logging, configured once at startup.

The codebase previously contained 198 ``print()`` calls and three files that
imported ``logging``. That is not a style complaint: without levels there is no
way to silence per-decision output during a 20,000-decision benchmark, and
without ``logger.exception`` a swallowed error leaves no trace. The RL model
never loading — the single worst defect in the project — stayed invisible for
months precisely because the failure path printed nothing at all.
"""

from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str | None = None) -> None:
    """Configure root logging. Idempotent; safe to call more than once."""
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-8s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Access logs are noise at 1 Hz; the simulator loop is the interesting signal.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


__all__ = ["configure_logging"]
