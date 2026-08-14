"""Constraint-aware classical routing: the honest strong baseline for QoS.

Adding QoS classes would be a strawman comparison if the only classical
reference were constraint-blind Dijkstra. It is trivial to beat an algorithm at
a job it was never asked to do.

This module implements the standard classical answer to the multi-constrained
optimal path (MCOP) problem: enumerate the *k* shortest paths under the
class-weighted additive cost, discard the ones that violate a hard constraint,
and return the cheapest survivor. It is exact over the candidate set, so within
that set it *is* the oracle.

That makes the QoS comparison honest in both directions:

* plain ``dijkstra`` is the constraint-blind floor,
* ``constrained_kshortest`` is the constraint-aware ceiling,
* every learned router is scored on how much of the gap between them it closes,
  using the same k=5 candidate set all three see.

A learned policy cannot beat the ceiling by construction. Reporting it as the
ceiling — rather than quietly omitting it — is the point.
"""

from __future__ import annotations

from core.models import NetworkState
from core.paths import build_decision, candidate_paths, failed_decision
from core.qos import QoSProfile, evaluate_path, select_best_path
from routing.base import Router


class ConstrainedRouter(Router):
    """k-shortest-paths with feasibility filtering (exact over the candidate set)."""

    name = "constrained"
    label = "Constrained k-shortest"
    description = (
        "Enumerates k candidate paths and returns the cheapest one that "
        "satisfies every QoS constraint. The classical MCOP baseline."
    )

    def __init__(self, k_paths: int = 5) -> None:
        self.k_paths = k_paths

    def find_route(self, state, src, dst, profile=None):
        profile = self.resolve_profile(profile)

        if src not in state.nodes or dst not in state.nodes:
            return failed_decision(src, dst, self.name)

        paths = candidate_paths(state, src, dst, k=self.k_paths)
        if not paths:
            return failed_decision(src, dst, self.name)

        path, evaluation = select_best_path(state, paths, profile)
        return build_decision(
            state,
            src,
            dst,
            path,
            self.name,
            is_fallback=False,
            diagnostics={
                "qos": evaluation.as_dict(),
                "candidates_considered": len(paths),
            },
        )


def qos_oracle(
    state: NetworkState,
    src: str,
    dst: str,
    profile: QoSProfile,
    k_paths: int = 5,
) -> tuple[list[str], float, bool]:
    """Best achievable (path, score, feasible) over the k-candidate set.

    Used as the normalisation ceiling when reporting learned-policy scores,
    because a raw number without a floor and a ceiling conveys nothing.
    """
    paths = candidate_paths(state, src, dst, k=k_paths)
    if not paths:
        return [], float("inf"), False
    path, evaluation = select_best_path(state, paths, profile)
    return path, evaluation.score, evaluation.feasible


def qos_floor(
    state: NetworkState,
    src: str,
    dst: str,
    profile: QoSProfile,
    k_paths: int = 5,
) -> float:
    """Worst score over the candidate set — the "random guessing" reference."""
    paths = candidate_paths(state, src, dst, k=k_paths)
    if not paths:
        return float("inf")
    scores = [evaluate_path(state, p, profile).score for p in paths]
    finite = [s for s in scores if s != float("inf")]
    return max(finite) if finite else float("inf")


__all__ = ["ConstrainedRouter", "qos_floor", "qos_oracle"]
