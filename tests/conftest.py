"""Shared fixtures.

The suite could not previously be run at all from a clean clone: pytest was in
no requirements file, there was no ``pytest.ini``, no ``conftest.py``, and two
of the eleven "test files" contained zero test functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.simulator import NetworkSimulator  # noqa: E402
from routing import build_router_set  # noqa: E402


@pytest.fixture
def sim() -> NetworkSimulator:
    """A deterministic 25-node simulator."""
    return NetworkSimulator(num_nodes=25, seed=42)


@pytest.fixture
def quiet_sim() -> NetworkSimulator:
    """A simulator with no background traffic, for closed-loop assertions."""
    return NetworkSimulator(num_nodes=25, seed=42, background_flows=0)


@pytest.fixture
def state(sim: NetworkSimulator):
    """A warmed-up network state."""
    for _ in range(10):
        sim.step()
    return sim.get_state()


@pytest.fixture(scope="session")
def routers() -> dict:
    """One shared router set. Loading torch models per test is slow."""
    return build_router_set(seed=42)


@pytest.fixture
def client():
    """A TestClient for the FastAPI app, with the lifespan started."""
    from fastapi.testclient import TestClient

    from service.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def results_dir() -> Path:
    return REPO_ROOT / "experiments" / "results"
