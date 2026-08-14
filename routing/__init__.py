"""Routing algorithms. One file per algorithm, one shared Router contract."""

from routing.base import Router
from routing.registry import (
    ALGORITHM_NAMES,
    DEGENERACY_EXEMPT,
    LEARNED_ALGORITHMS,
    build_router_set,
    describe_algorithms,
)

__all__ = [
    "ALGORITHM_NAMES",
    "DEGENERACY_EXEMPT",
    "LEARNED_ALGORITHMS",
    "Router",
    "build_router_set",
    "describe_algorithms",
]
