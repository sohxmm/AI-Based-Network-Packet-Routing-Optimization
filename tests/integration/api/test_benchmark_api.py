"""Tests for benchmark API and experiment lifecycle.

Covers:
- GET /benchmark/results contract
- GET /benchmark/results/{scenario} contract
- POST /experiments → poll → results lifecycle
- Hard cap rejection (not clamping)
"""

import sys
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        """Each algorithm entry has the required metric fields including effect_size_pct."""
        response = client.get("/benchmark/results")
        data = response.json()

        if not data["scenarios"]:
            pytest.skip("No benchmark result files found")

        # Check the first available scenario
        scenario_name = next(iter(data["scenarios"]))
        scenario = data["scenarios"][scenario_name]
        assert "algorithms" in scenario

        for algo_name, metrics in scenario["algorithms"].items():
            assert "mean_latency" in metrics, f"Missing mean_latency for {algo_name}"
            assert "p95_latency" in metrics, f"Missing p95_latency for {algo_name}"
            assert "util_variance" in metrics, f"Missing util_variance for {algo_name}"
            assert "success_rate" in metrics, f"Missing success_rate for {algo_name}"
            assert "fallback_rate" in metrics, f"Missing fallback_rate for {algo_name}"
            assert "dijkstra_match_rate" in metrics, f"Missing dijkstra_match_rate for {algo_name}"
            assert "wilcoxon_p_value" in metrics, f"Missing wilcoxon_p_value for {algo_name}"
            assert "effect_size_pct" in metrics, f"Missing effect_size_pct for {algo_name}"

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

    def test_effect_size_pct_is_computed(self):
        """effect_size_pct should be non-null for non-dijkstra algorithms."""
        response = client.get("/benchmark/results")
        data = response.json()

        if not data["scenarios"]:
            pytest.skip("No benchmark result files found")

        scenario_name = next(iter(data["scenarios"]))
        algos = data["scenarios"][scenario_name]["algorithms"]

        if "dijkstra" in algos:
            assert algos["dijkstra"]["effect_size_pct"] == 0.0

        # At least one non-dijkstra should have a computed effect size
        non_dijkstra = {k: v for k, v in algos.items() if k != "dijkstra"}
        if non_dijkstra:
            has_effect = any(
                v["effect_size_pct"] is not None for v in non_dijkstra.values()
            )
            assert has_effect, "No non-dijkstra algorithm has a computed effect_size_pct"


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

    def test_accept_at_cap_limit(self):
        """steps × pairs = 3000 exactly should be accepted (not rejected)."""
        response = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 300,
                "pairs_per_step": 10,  # 300 × 10 = 3000, exactly at cap
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
        for algo_name, metrics in results["algorithms"].items():
            assert "mean_latency" in metrics
            assert "success_rate" in metrics
            assert "effect_size_pct" in metrics

    def test_nonexistent_job_returns_404(self):
        """GET /experiments/nonexistent/status should return 404."""
        response = client.get("/experiments/nonexistent-id/status")
        assert response.status_code == 404

    def test_results_before_completion_returns_409(self):
        """GET /experiments/{id}/results while running should return 409."""
        # Submit a larger job that won't finish instantly
        submit_res = client.post(
            "/experiments",
            json={
                "topology_size": 25,
                "congestion_profile": "normal",
                "failure_rate": 0,
                "failure_pattern": "none",
                "steps": 100,
                "pairs_per_step": 5,
                "algorithms": ["dijkstra", "bellman_ford", "aco", "gnn", "rl", "multi_agent"],
            },
        )
        assert submit_res.status_code == 200
        job_id = submit_res.json()["job_id"]

        # Immediately try to get results — should be 409 (still running/queued)
        results_res = client.get(f"/experiments/{job_id}/results")
        assert results_res.status_code == 409
