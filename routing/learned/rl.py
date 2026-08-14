"""RLRouter — inference wrapper around the trained PPO routing policy.

This is the router the model-path bug lived in. The loader looked for
``rl_router_final.zip``; training saved, and the repository shipped,
``ppo_routing_agent.zip``. The resulting ``FileNotFoundError`` was caught by
``except (FileNotFoundError, ImportError, Exception): return False`` with no
logging, so the "RL router" quietly served a congestion-weighted heuristic —
effectively Dijkstra — for the entire life of the project.

Three things prevent a recurrence:

* the path comes from :mod:`ml.model_registry`, which training also reads, so
  the two cannot diverge;
* every failure branch logs, at a level chosen by how surprising it is;
* the observation is built by :func:`ml.features.build_observation`, the same
  function the training environment calls, so train/serve parity is structural
  rather than something two files have to agree about by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.models import NetworkState, RoutingDecision
from core.paths import build_decision, candidate_paths, failed_decision, path_cost
from core.qos import QoSProfile, evaluate_path
from ml.model_registry import path_for
from routing.base import Router

logger = logging.getLogger(__name__)

_ppo_class: Any = None
_build_observation: Any = None


def _try_import_ppo() -> bool:
    """Import Stable-Baselines3 lazily; the API must start without it."""
    global _ppo_class, _build_observation
    if _ppo_class is not None:
        return True
    try:
        from stable_baselines3 import PPO

        from ml.features import build_observation

        _ppo_class = PPO
        _build_observation = build_observation
        return True
    except ImportError:
        return False


class RLRouter(Router):
    """Route packets with a trained PPO policy, falling back to a heuristic."""

    name = "rl"
    label = "Reinforcement Learning (PPO)"
    description = (
        "PPO policy that picks among k candidate paths. Its observation "
        "encodes the link state, the source/destination pair, per-candidate "
        "features and the QoS class."
    )

    def __init__(self, seed: int = 42, k_paths: int = 5) -> None:
        self._seed = seed
        self._k_paths = k_paths
        self._model: Any = None
        self._model_path: Path | None = None
        self._n_links: int | None = None
        self._n_nodes: int | None = None
        self.last_used_model: bool = False

    # -- model management -------------------------------------------------

    def load_model(self, path: str | Path | None = None) -> None:
        """Load the PPO policy, raising on failure."""
        candidate = Path(path) if path else path_for("rl")
        zip_path = (
            candidate if candidate.suffix == ".zip" else candidate.with_suffix(".zip")
        )

        if not zip_path.exists() and not candidate.exists():
            raise FileNotFoundError(
                f"PPO model not found at {zip_path}. "
                "Train it with: python -m ml.training.train_rl"
            )
        if not _try_import_ppo():
            raise ImportError("stable-baselines3 is required to load the RL model.")

        actual = zip_path if zip_path.exists() else candidate
        self._model = _ppo_class.load(str(actual), device="cpu")
        self._model_path = actual

        # The observation layout is recorded at save time so the router never
        # has to re-derive it by dividing the observation width, which is how
        # the old n_links inference silently broke on a different topology.
        metadata = getattr(self._model, "custom_data", None) or {}
        self._n_links = metadata.get("n_links")
        self._n_nodes = metadata.get("n_nodes")
        logger.info(
            "RLRouter loaded %s (obs_dim=%d)",
            actual.name,
            self._model.observation_space.shape[0],
        )

    def try_load_model(self, path: str | Path | None = None) -> bool:
        """Load the model, reporting failure loudly instead of swallowing it."""
        try:
            self.load_model(path)
            return True
        except FileNotFoundError as exc:
            logger.warning(
                "RLRouter: no trained model (%s). Falling back to the "
                "congestion-aware heuristic. Train with: python -m ml.training.train_rl",
                exc,
            )
            return False
        except ImportError as exc:
            logger.warning("RLRouter: ML dependencies unavailable (%s).", exc)
            return False
        except Exception:
            logger.exception("RLRouter: unexpected error loading the model.")
            return False

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def requires_model(self) -> bool:
        return True

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "requires_model": True,
            "model_path": str(self._model_path) if self._model_path else None,
            "last_used_model": self.last_used_model,
        }

    # -- inference --------------------------------------------------------

    def find_route(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        profile: QoSProfile | None = None,
    ) -> RoutingDecision:
        profile = self.resolve_profile(profile)

        if src not in state.nodes or dst not in state.nodes:
            return failed_decision(src, dst, self.name)

        paths = candidate_paths(state, src, dst, k=self._k_paths)
        if not paths:
            return failed_decision(src, dst, self.name)

        if self._model is not None and self._observation_fits(state):
            path = self._policy_select(state, src, dst, paths, profile)
            self.last_used_model = True
            is_fallback = False
        else:
            path = min(paths, key=lambda p: path_cost(state, p))
            self.last_used_model = False
            is_fallback = True

        return build_decision(
            state,
            src,
            dst,
            path,
            self.name,
            is_fallback=is_fallback,
            diagnostics={
                "qos": evaluate_path(state, path, profile).as_dict(),
                "candidates_considered": len(paths),
            },
        )

    def _observation_fits(self, state: NetworkState) -> bool:
        """True when this policy can consume an observation built from *state*.

        Two ways it can fail: the live topology differs from the one the policy
        was trained on (a 25-node policy cannot read a 100-node observation), or
        the checkpoint predates the current observation layout. Both are checked
        against the policy's declared observation width rather than assumed, and
        both return False so the decision is honestly flagged as a fallback
        instead of crashing or silently reshaping into nonsense.
        """
        from ml.features import observation_dim

        expected = observation_dim(len(state.links), len(state.nodes))
        actual = int(self._model.observation_space.shape[0])
        if expected != actual:
            logger.warning(
                "RLRouter: policy expects a %d-wide observation but this "
                "topology produces %d (%d links, %d nodes). Using the heuristic "
                "fallback. Retrain with: python -m ml.training.train_rl",
                actual,
                expected,
                len(state.links),
                len(state.nodes),
            )
            return False
        return True

    def _policy_select(
        self,
        state: NetworkState,
        src: str,
        dst: str,
        paths: list[list[str]],
        profile: QoSProfile,
    ) -> list[str]:
        node_index = {node: i for i, node in enumerate(state.nodes)}
        observation = _build_observation(
            state,
            self._n_links or len(state.links),
            self._n_nodes or len(state.nodes),
            node_index.get(src, 0),
            node_index.get(dst, 0),
            paths,
            profile,
        )
        action, _ = self._model.predict(observation, deterministic=True)
        index = int(action) % len(paths)
        logger.debug("PPO action=%s -> candidate %d/%d", action, index, len(paths))
        return paths[index]


__all__ = ["RLRouter"]
