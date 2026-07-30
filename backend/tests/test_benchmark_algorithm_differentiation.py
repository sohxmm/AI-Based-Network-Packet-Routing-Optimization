import pytest
from pathlib import Path
import sys
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.network_sim import NetworkSimulator
from benchmark.run_benchmark import aco, rl, gnn, marl, predictor, route

@pytest.mark.asyncio
async def test_algorithm_differentiation():

    
    sim = NetworkSimulator(num_nodes=25, seed=42)
    state = sim.step()
    
    # Needs at least 20 pairs to check percentages
    node_list = list(state.nodes)
    pairs = []
    for _ in range(50):
        src, dst = random.sample(node_list, 2)
        pairs.append((src, dst))
        
    history = []
    
    rl_diff_dijkstra = 0
    marl_diff_rl = 0
    
    for src, dst in pairs:
        d_res = route("dijkstra", state, src, dst, history)
        rl_res = route("rl", state, src, dst, history)
        marl_res = route("multi_agent", state, src, dst, history)
        
        if d_res.path != rl_res.path:
            rl_diff_dijkstra += 1
            
        if rl_res.path != marl_res.path:
            marl_diff_rl += 1
            
    rl_diff_rate = rl_diff_dijkstra / len(pairs)
    marl_diff_rate = marl_diff_rl / len(pairs)
    
    assert rl_diff_rate >= 0.20, f"RL diff vs Dijkstra is {rl_diff_rate*100}%, expected >= 20%"
    assert marl_diff_rate >= 0.20, f"MARL diff vs RL is {marl_diff_rate*100}%, expected >= 20%"
