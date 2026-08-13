import argparse
import asyncio
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import scipy.stats as stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.network_sim import NetworkSimulator
from simulator.data_models import RoutingDecision, NetworkState, LinkState
from router.dijkstra import find_route as dijkstra_route
from router.bellman_ford import find_route as bellman_ford_route
from api.state import get_aco_router, get_rl_router, get_gnn_router, get_multi_agent_router
from ml.congestion_lstm import CongestionPredictor
from db.database import AsyncSessionLocal
from db.models import AlgorithmMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("benchmark")

ALGORITHMS = ["dijkstra", "bellman_ford", "aco", "gnn", "gnn_predictive", "rl", "rl_predictive", "multi_agent"]
SCENARIOS = ["normal_traffic", "high_congestion", "link_failures_5_10pct", "congestion_bursts", "large_topology_100_nodes"]

# Setup
aco = get_aco_router()
rl = get_rl_router()
gnn = get_gnn_router()
marl = get_multi_agent_router()
predictor = CongestionPredictor()
lstm_path = Path(__file__).resolve().parents[2] / "ml" / "models" / "congestion_lstm.pt"
if lstm_path.exists():
    predictor.load(str(lstm_path))

def _build_forecast_state(state: NetworkState, history: list) -> NetworkState | None:
    current_snapshot = [link.utilization for link in state.links]
    history.append(current_snapshot)
    if len(history) > predictor.seq_len:
        history.pop(0)
    
    if len(history) < predictor.seq_len or predictor.model is None:
        return None
        
    predicted_utils = predictor.predict_next(history)
    new_links = []
    for i, link in enumerate(state.links):
        pred_util = predicted_utils[i] if i < len(predicted_utils) else link.utilization
        pred_util = max(0.0, min(1.0, pred_util))
        new_links.append(LinkState(
            source=link.source, target=link.target, base_latency=link.base_latency,
            bandwidth=link.bandwidth, utilization=pred_util,
            queue_size=int(pred_util * 100), packet_loss_rate=max(0.0, pred_util - 0.7) * 0.2,
        ))
    return NetworkState(nodes=list(state.nodes), links=new_links, timestamp=state.timestamp, step_count=state.step_count)

def route(algo: str, state: NetworkState, src: str, dst: str, history: list) -> RoutingDecision:
    if algo == "dijkstra": return dijkstra_route(state, src, dst)
    if algo == "bellman_ford": return bellman_ford_route(state, src, dst)
    if algo == "aco": return aco.find_path(state, src, dst)
    if algo == "gnn": return gnn.predict(state, src, dst)
    if algo == "rl": return rl.predict(state, src, dst)
    if algo == "multi_agent": return marl.find_route(state, src, dst)
    
    fs = _build_forecast_state(state, history) or state
    if algo == "gnn_predictive": return gnn.predict(fs, src, dst)
    if algo == "rl_predictive": return rl.predict(fs, src, dst)
    raise ValueError(f"Unknown algo: {algo}")

def apply_scenario(sim: NetworkSimulator, scenario: str):
    if scenario == "high_congestion":
        for _, _, data in sim.graph.edges(data=True):
            if random.random() < 0.3:
                data["utilization"] = min(1.0, data["utilization"] + 0.4)
    elif scenario == "link_failures_5_10pct":
        for u, v in list(sim.failed_edges.keys()):
            sim.restore_link(u, v)
            
        edges = list(sim.graph.edges())
        if edges:
            to_remove = random.sample(edges, k=int(len(edges) * random.uniform(0.05, 0.10)))
            for u, v in to_remove:
                sim.inject_failure(u, v)
    elif scenario == "congestion_bursts":
        if random.random() < 0.1:
            edges = list(sim.graph.edges())
            if edges:
                sim.congestion_link = sim._edge_key(*random.choice(edges))
                sim.congestion_remaining = random.randint(3, 10)


def apply_parameterized_conditions(
    sim: NetworkSimulator,
    congestion_profile: str,
    failure_rate: float,
    failure_pattern: str,
):
    """Apply user-specified network conditions per step.

    This is the parameterized equivalent of apply_scenario().

    Args:
        sim: The network simulator instance.
        congestion_profile: "normal" (no extra congestion), "high", or "bursty".
        failure_rate: Fraction of links to fail (0.0 to 0.30).
        failure_pattern: "none", "random", or "targeted" (fail highest-degree-node links first).
    """
    # ── Congestion ──────────────────────────────────────────────────────────
    if congestion_profile == "high":
        for _, _, data in sim.graph.edges(data=True):
            if random.random() < 0.3:
                data["utilization"] = min(1.0, data["utilization"] + 0.4)
    elif congestion_profile == "bursty":
        if random.random() < 0.1:
            edges = list(sim.graph.edges())
            if edges:
                sim.congestion_link = sim._edge_key(*random.choice(edges))
                sim.congestion_remaining = random.randint(3, 10)

    # ── Link failures ───────────────────────────────────────────────────────
    if failure_rate > 0 and failure_pattern != "none":
        # Restore all previously failed edges first
        for u, v in list(sim.failed_edges.keys()):
            sim.restore_link(u, v)

        edges = list(sim.graph.edges())
        if edges:
            num_to_fail = max(1, int(len(edges) * failure_rate))

            if failure_pattern == "targeted":
                # Targeted: fail links connected to the highest-degree nodes first.
                # Sort edges by the sum of degrees of their endpoints (descending).
                degrees = dict(sim.graph.degree())
                edges_by_degree = sorted(
                    edges,
                    key=lambda e: degrees.get(e[0], 0) + degrees.get(e[1], 0),
                    reverse=True,
                )
                to_remove = edges_by_degree[:num_to_fail]
            else:
                # Random failure
                to_remove = random.sample(edges, k=min(num_to_fail, len(edges)))

            for u, v in to_remove:
                try:
                    sim.inject_failure(u, v)
                except ValueError:
                    pass  # Edge already failed or missing


def compute_metrics_from_results(
    results: dict[str, list],
    algorithms: list[str],
    n_steps: int,
    m_pairs: int,
    scenario_name: str = "custom",
) -> dict:
    """Compute aggregate metrics from raw per-decision results.

    Shared by both the fixed-scenario runner and the experiment sandbox.
    Returns the same shape as the JSON files in benchmark/results/.
    """
    import math as _math

    global_metrics: dict[str, Any] = {
        "scenario": scenario_name,
        "n_steps": n_steps,
        "m_pairs": m_pairs,
        "algorithms": {},
    }

    dijkstra_res = results.get("dijkstra", [])
    dijkstra_mean = 0.0

    # First pass: compute dijkstra mean for effect_size_pct
    if dijkstra_res:
        d_lats = [r["latency"] for r in dijkstra_res if r["success"] and r["latency"] < float("inf")]
        dijkstra_mean = float(np.mean(d_lats)) if d_lats else 0.0

    for algo in algorithms:
        algo_res = results.get(algo, [])
        if not algo_res:
            continue

        latencies = [r["latency"] for r in algo_res if r["success"] and r["latency"] < float("inf")]
        utils = [r["avg_utilization"] for r in algo_res if r["success"]]
        fallbacks = sum(1 for r in algo_res if r.get("is_fallback", False))
        successes = sum(1 for r in algo_res if r["success"])

        fallback_rate = fallbacks / len(algo_res) if algo_res else 0
        success_rate = successes / len(algo_res) if algo_res else 0

        mean_lat = float(np.mean(latencies)) if latencies else 0.0

        # Degeneracy check against dijkstra
        matches_dijkstra = 0
        if dijkstra_res:
            for r, dr in zip(algo_res, dijkstra_res):
                if r["success"] and dr["success"] and r["latency"] == dr["latency"] and r["path_len"] == dr["path_len"]:
                    matches_dijkstra += 1
        dijkstra_match_rate = matches_dijkstra / len(algo_res) if algo_res else 0

        wilcoxon_p = float("nan")
        wilcoxon_error = None
        if algo != "dijkstra" and dijkstra_res:
            paired_latencies = []
            paired_d_lats = []
            for r, dr in zip(algo_res, dijkstra_res):
                if r["success"] and dr["success"] and r["latency"] < float("inf") and dr["latency"] < float("inf"):
                    paired_latencies.append(r["latency"])
                    paired_d_lats.append(dr["latency"])

            if len(paired_latencies) > 0:
                diffs = np.array(paired_latencies) - np.array(paired_d_lats)
                if np.all(diffs == 0):
                    wilcoxon_p = float("nan")
                else:
                    try:
                        _, p = stats.wilcoxon(paired_latencies, paired_d_lats)
                        wilcoxon_p = float(p)
                    except Exception as e:
                        wilcoxon_error = str(e)
                        wilcoxon_p = -1.0

        # Effect size: % difference vs Dijkstra
        if algo == "dijkstra" or dijkstra_mean == 0:
            effect_size_pct = 0.0
        else:
            effect_size_pct = round((mean_lat - dijkstra_mean) / dijkstra_mean * 100, 2)

        # Sanitize NaN/Inf for JSON
        def _safe(v):
            if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                return None
            return v

        global_metrics["algorithms"][algo] = {
            "mean_latency": mean_lat,
            "median_latency": float(np.median(latencies)) if latencies else 0.0,
            "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "util_variance": float(np.var(utils)) if utils else 0.0,
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
            "dijkstra_match_rate": dijkstra_match_rate,
            "wilcoxon_p_value": _safe(wilcoxon_p),
            "wilcoxon_error": wilcoxon_error,
            "effect_size_pct": effect_size_pct,
        }

    return global_metrics


async def run_parameterized_scenario(
    topology_size: int,
    congestion_profile: str,
    failure_rate: float,
    failure_pattern: str,
    steps: int,
    pairs_per_step: int,
    algorithms: list[str],
    on_progress=None,
) -> dict:
    """Run a user-configured benchmark scenario using the verified benchmark engine.

    This is the parameterized entry point used by the Experiment Sandbox API.
    Uses the same routing dispatch, metric computation, and guardrail logic
    as the fixed scenarios.

    Args:
        topology_size: Number of nodes (25, 50, or 100).
        congestion_profile: "normal", "high", or "bursty".
        failure_rate: Fraction of links to fail per step (0.0-0.30).
        failure_pattern: "none", "random", or "targeted".
        steps: Number of simulation steps.
        pairs_per_step: Source/destination pairs per step.
        algorithms: Subset of the 8 known algorithms.
        on_progress: Optional callback(steps_completed, total) for progress reporting.

    Returns:
        Metrics dict in the same shape as GET /benchmark/results/{scenario}.
    """
    logger.info(
        f"--- Running parameterized scenario: {topology_size} nodes, "
        f"{congestion_profile} congestion, {failure_rate*100:.0f}% {failure_pattern} failures, "
        f"{steps} steps × {pairs_per_step} pairs ---"
    )

    sim = NetworkSimulator(num_nodes=topology_size, seed=42)
    results = defaultdict(list)
    history = []

    for step in range(steps):
        state = sim.step()
        apply_parameterized_conditions(sim, congestion_profile, failure_rate, failure_pattern)
        state = sim.get_state()

        pairs = []
        node_list = list(state.nodes)
        for _ in range(pairs_per_step):
            src, dst = random.sample(node_list, 2)
            pairs.append((src, dst))

        for algo in algorithms:
            for src, dst in pairs:
                decision = route(algo, state, src, dst, history)
                res = {
                    "algorithm": algo,
                    "step": step,
                    "src": src,
                    "dst": dst,
                    "latency": decision.total_latency,
                    "path_len": len(decision.path),
                    "avg_utilization": decision.avg_utilization,
                    "success": decision.success,
                    "is_fallback": getattr(decision, "is_fallback", False),
                }

                max_util = 0.0
                if decision.path:
                    for i in range(len(decision.path) - 1):
                        u, v = decision.path[i], decision.path[i + 1]
                        for link in state.links:
                            if (link.source == u and link.target == v) or (link.source == v and link.target == u):
                                max_util = max(max_util, link.utilization)
                res["max_path_utilization"] = max_util
                results[algo].append(res)

        # Report progress
        if on_progress:
            on_progress(step + 1, steps)

        # Yield control to the event loop periodically so the server stays responsive
        if step % 10 == 0:
            await asyncio.sleep(0)

    scenario_name = f"custom_{topology_size}n_{congestion_profile}_{failure_pattern}"
    return compute_metrics_from_results(results, algorithms, steps, pairs_per_step, scenario_name)


async def run_scenario(scenario: str, n_steps: int = 1000, m_pairs: int = 20):
    logger.info(f"--- Running {scenario} ---")
    nodes = 100 if scenario == "large_topology_100_nodes" else 25
    sim = NetworkSimulator(num_nodes=nodes, seed=42)
    
    results = defaultdict(list)
    history = []
    
    for step in range(n_steps):
        state = sim.step()
        apply_scenario(sim, scenario)
        state = sim.get_state()
        
        pairs = []
        node_list = list(state.nodes)
        for _ in range(m_pairs):
            src, dst = random.sample(node_list, 2)
            pairs.append((src, dst))
            
        for algo in ALGORITHMS:
            for src, dst in pairs:
                decision = route(algo, state, src, dst, history)
                # We need dict for json
                res = {
                    "algorithm": algo,
                    "step": step,
                    "src": src,
                    "dst": dst,
                    "latency": decision.total_latency,
                    "path_len": len(decision.path),
                    "avg_utilization": decision.avg_utilization,
                    "success": decision.success,
                    "is_fallback": getattr(decision, "is_fallback", False)
                }
                
                # Compute max link utilization on path
                max_util = 0.0
                if decision.path:
                    for i in range(len(decision.path)-1):
                        u, v = decision.path[i], decision.path[i+1]
                        for link in state.links:
                            if (link.source == u and link.target == v) or (link.source == v and link.target == u):
                                max_util = max(max_util, link.utilization)
                res["max_path_utilization"] = max_util
                results[algo].append(res)
                
    # Save raw results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{scenario}_{ts}.json"
    
    global_metrics = {
        "scenario": scenario,
        "n_steps": n_steps,
        "m_pairs": m_pairs,
        "algorithms": {}
    }
    
    # DB models
    db_metrics = []
    
    for algo in ALGORITHMS:
        algo_res = results[algo]
        latencies = [r["latency"] for r in algo_res if r["success"] and r["latency"] < float('inf')]
        utils = [r["avg_utilization"] for r in algo_res if r["success"]]
        fallbacks = sum(1 for r in algo_res if r["is_fallback"])
        successes = sum(1 for r in algo_res if r["success"])
        
        fallback_rate = fallbacks / len(algo_res) if algo_res else 0
        success_rate = successes / len(algo_res) if algo_res else 0
        
        # Path diversity (unique paths / total paths)
        paths = [str(r.get("path", [])) for r in algo_res if r["success"]]
        diversity = len(set(paths)) / len(paths) if paths else 0
        
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        
        # Degeneracy check against dijkstra
        dijkstra_res = results["dijkstra"]
        matches_dijkstra = 0
        for r, dr in zip(algo_res, dijkstra_res):
            if r["success"] and dr["success"] and r["latency"] == dr["latency"] and r["path_len"] == dr["path_len"]:
                matches_dijkstra += 1
        dijkstra_match_rate = matches_dijkstra / len(algo_res) if algo_res else 0
        
        wilcoxon_p = float("nan")
        wilcoxon_error = None
        if algo != "dijkstra":
            paired_latencies = []
            paired_d_lats = []
            for r, dr in zip(algo_res, dijkstra_res):
                if r["success"] and dr["success"] and r["latency"] < float('inf') and dr["latency"] < float('inf'):
                    paired_latencies.append(r["latency"])
                    paired_d_lats.append(dr["latency"])
                    
            if len(paired_latencies) > 0:
                print(f"[DEBUG] {scenario} {algo} vs dijkstra: paired lengths = {len(paired_latencies)} (original algo={len(latencies)}, successes={successes})")
                diffs = np.array(paired_latencies) - np.array(paired_d_lats)
                if np.all(diffs == 0):
                    wilcoxon_p = float("nan") # genuine tie
                else:
                    try:
                        _, p = stats.wilcoxon(paired_latencies, paired_d_lats)
                        wilcoxon_p = float(p)
                    except Exception as e:
                        wilcoxon_error = str(e)
                        wilcoxon_p = -1.0 # use -1.0 to indicate error instead of NaN tie
                    
        global_metrics["algorithms"][algo] = {
            "mean_latency": mean_lat,
            "median_latency": float(np.median(latencies)) if latencies else 0.0,
            "p95_latency": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "util_variance": float(np.var(utils)) if utils else 0.0,
            "max_path_utilization": float(np.max([r["max_path_utilization"] for r in algo_res])) if algo_res else 0.0,
            "diversity_index": diversity,
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
            "dijkstra_match_rate": dijkstra_match_rate,
            "wilcoxon_p_value": wilcoxon_p,
            "wilcoxon_error": wilcoxon_error
        }
        
        db_metrics.append(AlgorithmMetric(
            id=str(uuid.uuid4()),
            algorithm=algo,
            window_start_step=0,
            window_end_step=n_steps,
            avg_latency=mean_lat,
            success_rate=success_rate,
            num_decisions=len(algo_res)
        ))
        
    with open(out_file, "w") as f:
        json.dump(global_metrics, f, indent=2)
        
    # async with AsyncSessionLocal() as session:
    #     session.add_all(db_metrics)
    #     await session.commit()
        
    logger.info(f"Saved {scenario} to {out_file}")

from db.models import Base
from db.database import engine

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def run_all():
    # await init_db()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="all")
    args = parser.parse_args()
    
    scenarios_to_run = SCENARIOS if args.scenario == "all" else [args.scenario]
    for sc in scenarios_to_run:
        await run_scenario(sc)

if __name__ == "__main__":
    logger.info("Starting benchmark...")
    asyncio.run(run_all())

