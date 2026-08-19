"""Tests for benchmark API and experiment lifecycle.

Covers:
- GET /benchmark/results contract
- GET /benchmark/results/{scenario} contract
- POST /experiments → poll → results lifecycle
- Hard cap rejection (not clamping)
"""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from service.main import app

client = TestClient(app)

# ── Part 1: Benchmark Results API ────────────────────────────────────────────


class TestBenchmarkResultsAPI:
    """GET /benchmark/results and GET /benchmark/results/{scenario}."""

    def test_get_all_results_returns_scenarios(self):
        """API returns a scenarios dict and known_limitations."""
        response = client.get("/benchmark/results")
        assert response.status_code == 200

        data = response.json()
        assert "scenarios" in data
        assert "known_limitations" in data
        assert isinstance(data["scenarios"], dict)

    def test_get_all_results_has_expected_fields_per_algorithm(self):
        """Each algorithm entry carries the metrics the dashboard renders.

        ``util_variance`` and ``effect_size_pct`` used to be asserted here and
        are deliberately gone. ``util_variance`` was a variance over the whole
        run, which the closed-loop redesign made meaningless — bottleneck
        utilization is now reported as a mean and a p95 over per-decision path
        maxima. ``effect_size_pct`` was a percent difference between two means
        presented as an effect size, which it is not;
        :meth:`test_effect_size_is_a_real_effect_size` replaces it.
        """
        response = client.get("/benchmark/results")
        data = response.json()

        if not data["scenarios"]:
            pytest.skip("No benchmark result files found")

        scenario_name = next(iter(data["scenarios"]))
        scenario = data["scenarios"][scenario_name]
        assert "algorithms" in scenario

        required = [
            "mean_latency",
            "p95_latency",
            "success_rate",
            "fallback_rate",
            "qos_satisfaction_rate",
            "mean_path_max_utilization",
            "p95_path_max_utilization",
            "diversity_index",
            "dijkstra_match_rate",
        ]
        for algo_name, metrics in scenario["algorithms"].items():
            for field in required:
                assert field in metrics, f"Missing {field} for {algo_name}"

    def test_get_single_scenario(self):
        """GET /benchmark/results/{scenario} returns data for one scenario."""
        response = client.get("/benchmark/results/normal_traffic")
        if response.status_code == 404:
            pytest.skip("No benchmark results for normal_traffic")

        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "normal_traffic"
        assert "algorithms" in data
        assert "known_limitations" in data

    def test_unknown_scenario_returns_404(self):
        """Unknown scenario returns 404 with helpful message."""
        response = client.get("/benchmark/results/nonexistent_scenario")
        assert response.status_code == 404

    def test_effect_size_is_a_real_effect_size(self):
        """Every comparison carries Cliff's delta, a magnitude and a CI.

        A percent difference in means is not an effect size: it says nothing
        about spread, so a 60% difference swamped by run-to-run variance and a
        0.7% difference that is perfectly consistent look equally impressive.
        Cliff's delta is the probability one group exceeds the other, and the
        bootstrap CI is what tells a reader how large the difference could
        plausibly be.
        """
        response = client.get("/benchmark/results")
        data = response.json()

        if not data["scenarios"]:
            pytest.skip("No benchmark result files found")

        scenario_name = next(iter(data["scenarios"]))
        algos = data["scenarios"][scenario_name]["algorithms"]

        # Dijkstra is the baseline, so it has nothing to compare against.
        compared = {k: v for k, v in algos.items() if k != "dijkstra"}
        assert compared, "Nothing to compare against the Dijkstra baseline"

        for name, metrics in compared.items():
            comparison = metrics.get("comparison_vs_dijkstra")
            assert comparison, f"{name} has no comparison against Dijkstra"
            assert -1.0 <= comparison["cliffs_delta"] <= 1.0
            assert comparison["effect_magnitude"] in {
                "negligible",
                "small",
                "medium",
                "large",
            }
            assert comparison["ci95_low"] <= comparison["ci95_high"]
            assert comparison["n_runs"] >= 2, (
                f"{name} was compared across {comparison['n_runs']} run(s). "
                f"Statistics across a single trajectory are pseudo-replicated."
            )


# ── Part 4: Experiment API ───────────────────────────────────────────────────


class TestExperimentHardCaps:
    """Verify hard caps reject (not clamp) over-limit requests."""

    def test_reject_over_limit_total_decisions(self):
        """steps × pairs_per_step > 3000 MUST return 422, NOT 200 with clamped values."""
        response = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 300,
                "pairs_per_step": 11,  # 300 × 11 = 3300 > 3000
                "algorithms": ["dijkstra"],
            },
        )
        assert response.status_code == 422, (
            f"Expected 422 for over-cap request but got {response.status_code}. "
            f"The hard cap must REJECT, not silently clamp."
        )
        body = response.json()
        detail_str = str(body.get("detail", ""))
        # The error will likely be about pairs_per_step <= 10 since that is evaluated first by Pydantic.
        assert "less than or equal to 10" in detail_str or "3000" in detail_str or "exceed" in detail_str.lower(), (
            f"Error message should mention the cap, got: {detail_str}"
        )

    def test_reject_steps_over_max(self):
        """steps > 300 should be rejected."""
        response = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 301,
                "pairs_per_step": 1,
                "algorithms": ["dijkstra"],
            },
        )
        assert response.status_code == 422

    def test_reject_pairs_over_max(self):
        """pairs_per_step > 10 should be rejected."""
        response = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 1,
                "pairs_per_step": 11,
                "algorithms": ["dijkstra"],
            },
        )
        assert response.status_code == 422

    def test_reject_invalid_topology_size(self):
        """topology_size not in [25, 50, 100] should be rejected."""
        response = client.post(
            "/experiments",
            json={
                "topology_size": 30,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 10,
                "pairs_per_step": 2,
                "algorithms": ["dijkstra"],
            },
        )
        assert response.status_code == 422

    def test_reject_invalid_algorithm(self):
        """Unknown algorithm names should be rejected."""
        response = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 5,
                "pairs_per_step": 2,
                "algorithms": ["fake_algo"],
            },
        )
        assert response.status_code == 422

    def test_reject_over_budget_once_replication_is_counted(self):
        """The budget is steps x pairs x runs, not steps x pairs.

        ``runs`` defaults to 3 because a single trajectory cannot support the
        statistics the API reports back. That makes replication part of the
        cost, so a request that would have been legal under the old two-factor
        budget can exceed the current one.
        """
        response = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "steps": 300,
                "pairs_per_step": 10,  # 300 x 10 x 3 runs = 9000 > 6000
                "algorithms": ["dijkstra"],
            },
        )
        assert response.status_code == 422
        assert "6000" in str(response.json().get("detail", ""))

    def test_accept_at_cap_limit(self):
        """A request exactly at the budget is accepted, not rejected."""
        response = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 300,
                "pairs_per_step": 10,
                "runs": 2,  # 300 x 10 x 2 = 6000, exactly at cap
                "algorithms": ["dijkstra"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


class TestExperimentLifecycle:
    """POST /experiments → GET status → GET results (full lifecycle)."""

    def test_submit_poll_results(self):
        """Submit a small experiment, poll until done, retrieve results."""
        # Submit with minimal config for speed
        submit_res = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 3,
                "pairs_per_step": 2,
                "runs": 2,
                "algorithms": ["dijkstra", "bellman_ford"],
            },
        )
        assert submit_res.status_code == 200
        job_id = submit_res.json()["job_id"]
        assert job_id

        # Poll status until done (with timeout)
        import time
        max_wait = 60  # seconds
        start = time.time()
        final_state = None

        while time.time() - start < max_wait:
            status_res = client.get(f"/experiments/{job_id}/status")
            assert status_res.status_code == 200
            status = status_res.json()
            assert status["state"] in ("queued", "running", "done", "failed")

            if status["state"] == "done":
                final_state = "done"
                break
            elif status["state"] == "failed":
                final_state = "failed"
                break

            time.sleep(1)

        assert final_state == "done", f"Experiment did not complete in {max_wait}s, state: {final_state}"

        # Retrieve results
        results_res = client.get(f"/experiments/{job_id}/results")
        assert results_res.status_code == 200
        results = results_res.json()

        # Verify shape matches benchmark results
        assert "algorithms" in results
        assert "dijkstra" in results["algorithms"]
        assert "bellman_ford" in results["algorithms"]

        # Verify per-algorithm fields
        for metrics in results["algorithms"].values():
            assert "mean_latency" in metrics
            assert "success_rate" in metrics
            assert "fallback_rate" in metrics
        # A sandbox run reports the same statistics as the committed benchmark,
        # so a user cannot accidentally compare a rigorous number against a
        # casual one.
        assert "comparison_vs_dijkstra" in results["algorithms"]["bellman_ford"]

    def test_nonexistent_job_returns_404(self):
        """GET /experiments/nonexistent/status should return 404."""
        response = client.get("/experiments/nonexistent-id/status")
        assert response.status_code == 404

    @pytest.mark.parametrize("state", ["queued", "running"])
    def test_results_before_completion_returns_409(self, state):
        """GET /experiments/{id}/results on an unfinished job returns 409.

        The job is injected directly rather than raced against a real one.
        ``TestClient`` runs FastAPI background tasks synchronously *after* the
        response is returned, so by the time the follow-up request is issued
        the work has already finished — the previous version of this test
        submitted a deliberately slow experiment and then asserted it was
        unfinished, which made it a test of scheduling luck rather than of the
        409 contract.
        """
        from service.api.experiments import _jobs

        job_id = f"test-{state}"
        _jobs[job_id] = {
            "state": state,
            "progress": {"runs_completed": 0, "total": 3},
            "error": None,
            "result": None,
            "created_at": datetime.now(UTC),
        }
        try:
            response = client.get(f"/experiments/{job_id}/results")
            assert response.status_code == 409
            assert state in str(response.json().get("detail", ""))
        finally:
            _jobs.pop(job_id, None)
