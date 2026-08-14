"""The contract every routing algorithm implements.

Before this existed, the API dispatcher had to maintain a hand-written lookup
table mixing bare functions (``dijkstra_route``) with bound methods of three
different names (``find_path``, ``predict``, ``find_route``). One name, one
signature, one place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import NetworkState, RoutingDecision
from core.qos import QoSProfile, get_profile


class Router(ABC):
    """A routing algorithm: given the network and a demand, choose a path."""

    #: Stable identifier used in the API, the benchmark and the UI.
    name: str = "router"
    #: Human-readable label for the dashboard.
    label: str = "Router"
    #: One-line description surfaced in the UI.
    description: str = ""

    @abstractmethod
    def find_route(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        profile: QoSProfile | None = None,
    ) -> RoutingDecision:
        """Return a routing decision for ``src -> dst`` under *profile*."""

    # -- capability flags -------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True when this router is serving a trained model.

        Classical and heuristic routers have nothing to train, so they report
        True: there is no such thing as a fallback for them.
        """
        return True

    @property
    def requires_model(self) -> bool:
        """True when a missing artifact would silently degrade this router."""
        return False

    def status(self) -> dict[str, object]:
        """Loading/diagnostic state, surfaced by ``GET /health/models``."""
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "requires_model": self.requires_model,
        }

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def resolve_profile(profile: QoSProfile | None) -> QoSProfile:
        """Normalise an optional profile into a concrete one."""
        return profile if profile is not None else get_profile(None)
