"""API tests.

``api/routers/`` previously had zero test coverage. Every endpoint below either
had a bug we found, or is new and needs pinning.
"""

from __future__ import annotations

import pytest

from routing import ALGORITHM_NAMES


class TestNetworkState:
    def test_state_has_the_documented_shape(self, client):
        payload = client.get("/network/state").json()
        assert set(payload) == {"nodes", "links", "timestamp", "step_count"}
        assert len(payload["nodes"]) == 25

        for link in payload["links"]:
            assert set(link) == {
                "source",
                "target",
                "base_latency",
                "bandwidth",
                "utilization",
                "queue_size",
                "packet_loss_rate",
            }
            assert 0.0 <= link["utilization"] <= 1.0

    def test_source_provenance_is_exposed(self, client):
        """A reader must be able to tell simulated data from measured data."""
        payload = client.get("/network/source").json()
        assert payload["kind"] == "simulated"
        assert payload["closed_loop"] is True
        assert payload["avg_degree"] >= 3.0

    def test_algorithm_and_class_catalogue(self, client):
        payload = client.get("/network/algorithms").json()
        names = [a["name"] for a in payload["algorithms"]]
        assert names == ALGORITHM_NAMES
        assert len(payload["traffic_classes"]) == 5


class TestRouting:
    def test_routes_a_packet(self, client):
        response = client.post(
            "/network/route",
            json={"source": "R1", "destination": "R14", "algorithm": "dijkstra"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["path"][0] == "R1"
        assert payload["path"][-1] == "R14"
        assert len(set(payload["path"])) == len(payload["path"]), "loop in path"
        assert payload["hops"], "a per-hop breakdown should be returned"

    def test_unknown_node_is_404_with_actionable_detail(self, client):
        response = client.post(
            "/network/route", json={"source": "R999", "destination": "R14"}
        )
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "missing_nodes" in detail
        assert detail["missing_nodes"] == ["R999"]

    def test_invalid_algorithm_is_rejected(self, client):
        response = client.post(
            "/network/route",
            json={"source": "R1", "destination": "R14", "algorithm": "quantum"},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "traffic_class", ["emergency", "interactive", "gaming", "bulk", "best_effort"]
    )
    def test_every_traffic_class_routes(self, client, traffic_class):
        response = client.post(
            "/network/route",
            json={
                "source": "R1",
                "destination": "R14",
                "algorithm": "constrained",
                "traffic_class": traffic_class,
            },
        )
        assert response.status_code == 200
        assert response.json()["diagnostics"]["qos"]["hops"] > 0

    def test_compare_returns_every_requested_algorithm(self, client):
        response = client.post(
            "/network/route/compare",
            json={
                "source": "R1",
                "destination": "R14",
                "algorithms": ["dijkstra", "aco", "gnn"],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert [r["algorithm"] for r in payload["results"]] == ["dijkstra", "aco", "gnn"]
        assert "oracle" in payload

    def test_every_algorithm_gets_a_qos_verdict_including_the_blind_ones(self, client):
        """Feasibility is computed by the endpoint, not self-reported by routers.

        The comparison view used to read the QoS block out of
        ``decision.diagnostics``, which only the constraint-aware routers
        populate. So every row carried a feasibility verdict except `dijkstra`
        and `bellman_ford` — the two routers that are constraint-*blind*, and
        therefore precisely the two whose feasibility a reader needs to see.
        The whole argument for QoS-aware routing is that Dijkstra will return an
        infeasible path; a view that cannot show that is arguing the point badly.
        """
        payload = client.post(
            "/network/route/compare",
            json={
                "source": "R1",
                "destination": "R14",
                "traffic_class": "emergency",
                "algorithms": ["dijkstra", "bellman_ford", "constrained", "gnn"],
            },
        ).json()

        for result in payload["results"]:
            qos = result.get("qos")
            assert qos is not None, (
                f"{result['algorithm']} has no QoS verdict. Feasibility is a "
                f"property of (path, state, profile) and must not depend on "
                f"whether the router chose to volunteer it."
            )
            assert qos["traffic_class"] == "emergency"
            assert isinstance(qos["feasible"], bool)
            assert isinstance(qos["violations"], list)
            assert qos["hops"] == len(result["path"]) - 1

    def test_predictive_variants_only_appear_when_a_forecast_exists(self, client):
        """Emitting them unconditionally is what produced duplicate columns."""
        payload = client.post(
            "/network/route/compare",
            json={
                "source": "R1",
                "destination": "R14",
                "algorithms": ["gnn"],
                "use_forecast": True,
            },
        ).json()

        names = [r["algorithm"] for r in payload["results"]]
        if payload["forecast_available"]:
            assert "gnn_predictive" in names
        else:
            assert "gnn_predictive" not in names, (
                "A predictive column appeared without a forecaster, which is "
                "exactly how the benchmark ended up with duplicate columns."
            )

    def test_candidates_are_exposed_with_qos_scores(self, client):
        payload = client.get(
            "/network/candidates?source=R1&destination=R14&traffic_class=emergency"
        ).json()
        assert payload["candidates"]
        for candidate in payload["candidates"]:
            assert candidate["path"][0] == "R1"
            assert "feasible" in candidate["qos"]
            assert candidate["hops"]


class TestForecast:
    def test_forecast_is_honest_about_being_unavailable(self, client):
        """It used to return persistence values labelled 'predicted_utilization'."""
        payload = client.get("/network/congestion-forecast").json()
        if not payload["model_trained"]:
            assert payload["predictions"] == []
            assert "message" in payload

    def test_forecast_is_idempotent_for_the_caller(self, client):
        """A GET handler used to mutate a module-level global."""
        first = client.get("/network/congestion-forecast").json()
        second = client.get("/network/congestion-forecast").json()
        assert first["model_trained"] == second["model_trained"]


class TestMetrics:
    def test_history_limit_is_capped(self, client):
        """An unbounded limit was a trivial denial-of-service vector."""
        assert client.get("/metrics/history?limit=999999").status_code == 422
        assert client.get("/metrics/history?limit=50").status_code == 200

    def test_summary_reports_model_state(self, client):
        payload = client.get("/metrics/summary").json()
        assert "models" in payload
        assert "fallback_rate" in payload

    def test_summary_rejects_an_unknown_algorithm(self, client):
        assert client.get("/metrics/summary?algorithm=quantum").status_code == 422

    def test_model_health_endpoint(self, client):
        """The one-request answer to 'is the AI actually running?'."""
        payload = client.get("/health/models").json()
        assert payload["status"] in {"ok", "degraded"}
        for key in ("gnn", "rl", "lstm", "multi_agent"):
            assert key in payload["models"]
            assert "file_present" in payload["models"][key]
            assert "train_command" in payload["models"][key]


class TestHealth:
    def test_health_reports_loop_liveness(self, client):
        """A bare {"status": "ok"} hid a dead simulator loop behind a green check."""
        payload = client.get("/health").json()
        assert "simulator_last_tick_age_s" in payload
        assert "consecutive_tick_failures" in payload
