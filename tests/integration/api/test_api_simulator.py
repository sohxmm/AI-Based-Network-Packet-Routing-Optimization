"""Simulator control endpoints, plus the network-source switching."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestSimulatorControl:
    def test_step_advances_the_counter(self, client):
        before = client.get("/network/state").json()["step_count"]
        after = client.post("/sim/step").json()["step_count"]
        assert after == before + 1

    def test_reset_returns_to_step_zero(self, client):
        client.post("/sim/step")
        assert client.post("/sim/reset").json()["step_count"] == 0

    def test_inject_and_restore_is_a_round_trip(self, client):
        state = client.get("/network/state").json()
        link = state["links"][0]
        source, target = link["source"], link["target"]

        after_failure = client.post(
            "/sim/inject-failure", json={"source": source, "target": target}
        )
        assert after_failure.status_code == 200
        keys = {
            frozenset((entry["source"], entry["target"]))
            for entry in after_failure.json()["links"]
        }
        assert frozenset((source, target)) not in keys

        restored = client.post(
            "/sim/restore-link", json={"source": source, "target": target}
        )
        assert restored.status_code == 200
        found = next(
            entry
            for entry in restored.json()["links"]
            if frozenset((entry["source"], entry["target"]))
            == frozenset((source, target))
        )
        assert found["base_latency"] == link["base_latency"]

    def test_failing_a_nonexistent_link_is_400(self, client):
        response = client.post(
            "/sim/inject-failure", json={"source": "R1", "target": "R999"}
        )
        assert response.status_code == 400


class TestNetworkSources:
    """The platform can also run against a recorded trace or a real network."""

    def test_switching_to_a_trace_and_back(self, client):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            frames = [
                {
                    "step": step,
                    "links": [
                        {
                            "source": "A",
                            "target": "B",
                            "base_latency": 10,
                            "bandwidth": 1000,
                            "utilization": 0.2 + 0.01 * step,
                        },
                        {
                            "source": "B",
                            "target": "C",
                            "base_latency": 12,
                            "bandwidth": 1000,
                            "utilization": 0.4,
                        },
                    ],
                }
                for step in range(5)
            ]
            trace.write_text("\n".join(json.dumps(f) for f in frames))

            response = client.post(
                "/sim/source", json={"kind": "trace", "trace_path": str(trace)}
            )
            assert response.status_code == 200
            described = response.json()["source"]
            assert described["kind"] == "trace"
            assert described["closed_loop"] is False, (
                "A recording cannot be influenced by our routing, and the API "
                "must say so."
            )
            assert set(response.json()["state"]["nodes"]) == {"A", "B", "C"}

        # Restore the simulator so later tests are unaffected.
        assert client.post("/sim/source", json={"kind": "simulated"}).status_code == 200
        assert client.get("/network/source").json()["kind"] == "simulated"

    def test_missing_trace_file_is_404(self, client):
        response = client.post(
            "/sim/source", json={"kind": "trace", "trace_path": "/nonexistent.jsonl"}
        )
        assert response.status_code == 404

    def test_trace_kind_requires_a_path(self, client):
        assert client.post("/sim/source", json={"kind": "trace"}).status_code == 422

    def test_live_probing_is_disabled_by_default(self, client):
        """Measuring real hosts must never start by accident."""
        response = client.post(
            "/sim/source", json={"kind": "live", "targets": ["127.0.0.1"]}
        )
        assert response.status_code == 403
        assert "LIVE_PROBE_ENABLED" in response.json()["detail"]

    def test_source_health_reports_probe_state(self, client):
        payload = client.get("/sim/source/health").json()
        assert payload["kind"] == "simulated"
        assert payload["live"] is False


class TestFailover:
    def test_watching_a_flow_records_its_current_route(self, client):
        response = client.post(
            "/network/failover/watch", json={"source": "R1", "destination": "R14"}
        )
        assert response.status_code == 200
        watched = response.json()["watching"]
        assert any(f["source"] == "R1" and f["destination"] == "R14" for f in watched)

        client.delete("/network/failover/watch?source=R1&destination=R14")

    @pytest.mark.slow
    def test_convergence_measurement_compares_algorithms(self, client):
        state = client.get("/network/state").json()
        link = state["links"][0]

        response = client.post(
            "/network/failover/convergence",
            json={
                "source": "R1",
                "destination": "R14",
                "link_source": link["source"],
                "link_target": link["target"],
                "algorithms": ["dijkstra", "gnn"],
                "max_steps": 5,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["results"]) == 2
        for row in payload["results"]:
            assert "converged" in row
            # Recovery speed alone is not the story; the resulting path quality
            # has to be reported next to it.
            assert "latency_after" in row
