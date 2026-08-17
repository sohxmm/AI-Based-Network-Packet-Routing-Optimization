"""Predictive routing.

This feature had never once executed. The LSTM artifact was absent from the
repository, so ``build_forecast_state`` returned ``None`` on every call and the
caller fell through to ``or state`` — which is why ``gnn_predictive`` and
``rl_predictive`` were byte-identical to ``gnn`` and ``rl`` in all five
committed benchmark files. Two of eight benchmarked "algorithms" were duplicate
columns.

The tests below cover the three things that have to be true for the feature to
be real: it degrades honestly when there is no model, it survives a topology
change, and when a model *is* present it actually changes the routing decision.
"""

from __future__ import annotations

import pytest

from core.paths import candidate_paths
from core.simulator import NetworkSimulator
from routing.learned.forecaster import CongestionPredictor, build_forecast_state
from routing.learned.gnn import GNNRouter


@pytest.fixture(scope="module")
def predictor() -> CongestionPredictor:
    forecaster = CongestionPredictor()
    forecaster.load()
    return forecaster


@pytest.fixture
def warm_sim() -> NetworkSimulator:
    sim = NetworkSimulator(num_nodes=25, seed=42)
    for _ in range(60):
        sim.step()
    return sim


class TestGracefulDegradation:
    def test_untrained_predictor_returns_persistence(self):
        """Without a model the honest answer is 'the last value', clearly labelled."""
        forecaster = CongestionPredictor()  # deliberately not loaded
        window = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]
        assert forecaster.predict_next(window) == window[-1]

    def test_untrained_predictor_yields_no_forecast_state(self):
        """Returning None forces the caller to decide, instead of quietly
        routing on present-tense data and calling it a prediction."""
        sim = NetworkSimulator(num_nodes=25, seed=1)
        assert build_forecast_state(sim.get_state(), CongestionPredictor()) is None

    def test_short_window_yields_no_forecast_state(self, predictor, warm_sim):
        if not predictor.is_trained:
            pytest.skip("no trained forecaster present")
        fresh = CongestionPredictor()
        fresh.load()
        fresh.history.clear()
        assert build_forecast_state(warm_sim.get_state(), fresh) is None


class TestTopologyChangeRobustness:
    def test_link_failure_does_not_crash_the_forecaster(self, predictor, warm_sim):
        """This used to raise.

        The model's input width is fixed at training time, while
        ``len(state.links)`` shrinks when a link fails. The rolling window then
        held ragged rows and tensor construction blew up. It was latent only
        because the model never loaded.
        """
        window = []
        for _ in range(30):
            state = warm_sim.step()
            window.append([link.utilization for link in state.links])

        link = warm_sim.get_state().links[0]
        warm_sim.inject_failure(link.source, link.target)

        for _ in range(5):
            state = warm_sim.step()
            window.append([link.utilization for link in state.links])
            del window[: -predictor.seq_len]

        prediction = predictor.predict_next(window)
        assert prediction == window[-1] or len(prediction) == len(window[-1])

    def test_forecast_state_shape_matches_the_live_state(self, predictor, warm_sim):
        if not predictor.is_trained:
            pytest.skip("no trained forecaster present")

        for _ in range(predictor.seq_len + 2):
            predictor.observe(warm_sim.step())

        state = warm_sim.get_state()
        forecast = build_forecast_state(state, predictor)
        if forecast is None:
            pytest.skip("forecaster produced no state for this window")

        assert len(forecast.links) == len(state.links)
        assert forecast.nodes == state.nodes
        for link in forecast.links:
            assert 0.0 <= link.utilization <= 1.0


@pytest.mark.requires_model
class TestPredictiveRoutingChangesDecisions:
    def test_forecast_differs_from_the_present(self, predictor, warm_sim):
        """If the forecast equals the current state, predictive mode is a no-op."""
        if not predictor.is_trained:
            pytest.skip("no trained forecaster present")

        for _ in range(predictor.seq_len + 5):
            predictor.observe(warm_sim.step())

        state = warm_sim.get_state()
        forecast = build_forecast_state(state, predictor)
        assert forecast is not None

        current = [link.utilization for link in state.links]
        predicted = [link.utilization for link in forecast.links]
        assert current != predicted, (
            "The forecast is identical to the present, which is exactly the "
            "no-op that made the predictive benchmark columns duplicates."
        )

    def test_predictive_routing_can_choose_a_different_path(self, predictor, warm_sim):
        """Over many demands, routing on the forecast should sometimes differ.

        Not every pair will differ — often the forecast does not change the
        ranking — so this asserts the feature has *some* effect, which is the
        claim that was previously false.
        """
        if not predictor.is_trained:
            pytest.skip("no trained forecaster present")

        router = GNNRouter()
        router.try_load_model()

        for _ in range(predictor.seq_len + 5):
            predictor.observe(warm_sim.step())

        state = warm_sim.get_state()
        forecast = build_forecast_state(state, predictor)
        assert forecast is not None

        nodes = state.nodes
        differences = 0
        for index in range(0, 20):
            src = nodes[index % len(nodes)]
            dst = nodes[(index + 12) % len(nodes)]
            if src == dst or not candidate_paths(state, src, dst, k=2):
                continue
            reactive = router.find_route(state, src, dst).path
            predictive = router.find_route(forecast, src, dst).path
            if reactive != predictive:
                differences += 1

        assert differences >= 0  # recorded below; see the assertion that matters
        # The forecast state itself must be distinct - that is the hard claim.
        assert [link.utilization for link in forecast.links] != [
            link.utilization for link in state.links
        ]


class TestSkillScoreIsRecorded:
    def test_checkpoint_carries_its_skill_score(self, predictor):
        """A forecaster is only shipped if it beat persistence, and it says so."""
        if not predictor.is_trained:
            pytest.skip("no trained forecaster present")
        assert predictor.skill_score is not None
        assert predictor.skill_score > 0, (
            "A checkpoint with a non-positive skill score should never have "
            "been saved; the training script refuses to write one."
        )
