import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.network_sim import NetworkSimulator
from benchmark.run_benchmark import aco, rl, gnn, marl, predictor, route

@pytest.mark.asyncio
async def test_no_silent_fallback():

    
    sim = NetworkSimulator(num_nodes=25, seed=42)
    state = sim.step()
    
    pairs = [("R1", "R10"), ("R5", "R15"), ("R2", "R20"), ("R3", "R8"), ("R11", "R21")]
    history = []
    
    # Check algorithms that use trained models
    ai_algos = ["gnn", "gnn_predictive", "rl", "rl_predictive", "multi_agent"]
    
    for algo in ai_algos:
        fallbacks = 0
        for src, dst in pairs:
            decision = route(algo, state, src, dst, history)
            if getattr(decision, "is_fallback", False):
                fallbacks += 1
        
        fallback_rate = fallbacks / len(pairs)
        assert fallback_rate == 0.0, f"{algo} had fallback_rate {fallback_rate} > 0"
