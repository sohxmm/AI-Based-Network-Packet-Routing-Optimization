"""Random path selection — the floor every learned policy must clear.

The point is a simple one: a reward number, a latency figure or
an accuracy percentage conveys no information without a floor and a ceiling. If
you do not know what random guessing scores, you cannot tell whether a model
learned something or merely produced plausible output.

This router picks uniformly among the same k congestion-weighted candidate paths
every other router sees. Any algorithm that does not beat it has not learned
anything, and saying so is the whole reason it exists.
"""

from __future__ import annotations

import random

from core.models import NetworkState, RoutingDecision
from core.paths import build_decision, candidate_paths, failed_decision
from core.qos import QoSProfile, evaluate_path
from routing.base import Router


class RandomBaselineRouter(Router):
    """Uniformly sample one of the k candidate paths."""

    name = "random_baseline"
    label = "Random baseline"
    description = (
        "Picks uniformly among the candidate paths. The reference floor: any "
        "learned policy that fails to beat this has learned nothing."
    )

    def __init__(self, seed: int = 42, k_paths: int = 5) -> None:
        self._random = random.Random(seed)
        self._k_paths = k_paths

    def find_route(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        profile: QoSProfile | None = None,
    ) -> RoutingDecision:
        profile = self.resolve_profile(profile)

        paths = candidate_paths(state, src, dst, k=self._k_paths)
        if not paths:
            return failed_decision(src, dst, self.name)

        path = self._random.choice(paths)
        return build_decision(
            state,
            src,
            dst,
            path,
            self.name,
            diagnostics={"qos": evaluate_path(state, path, profile).as_dict()},
        )


__all__ = ["RandomBaselineRouter"]
