/**
 * BenchmarkReport — Algorithm Performance Benchmark results view.
 *
 * Shows scenario selector, guardrail warning banners, metrics table,
 * grouped bar chart (latency + utilization variance trade-off), and
 * known limitations text. Also reused by the Experiment Sandbox for
 * rendering custom experiment results.
 *
 * Renders neutrally — the data tells a scenario-dependent, algorithm-specific story.
 */

import React, { useEffect, useState, useMemo } from "react";
import { BenchmarkResultView } from "./BenchmarkResultView.jsx";
export { BenchmarkResultView };

import { API_BASE_URL as API_BASE } from "../config.js";

const SCENARIO_LABELS = {
  normal_traffic: "Normal Traffic",
  high_congestion: "High Congestion",
  link_failures_5_10pct: "Link Failures (5–10%)",
  congestion_bursts: "Congestion Bursts",
  large_topology_100_nodes: "Large Topology (100 Nodes)",
};

const ALGO_LABELS = {
  dijkstra: "Dijkstra",
  bellman_ford: "B-Ford",
  aco: "ACO",
  gnn: "GNN",
  gnn_predictive: "GNN-P",
  rl: "RL",
  rl_predictive: "RL-P",
  multi_agent: "MARL",
};

const FINDINGS_SUMMARY = `ACO is the most consistent variance-reducer across conditions. GNN/RL only clearly help under sustained high congestion. MARL's benefit is congestion-dependent and always latency-costly. All three AI approaches degenerate on the untrained large topology (100 nodes).`;


/**
 * BenchmarkReport — Full benchmark report page with scenario selector.
 */
export default function BenchmarkReport() {
  const [allData, setAllData] = useState(null);
  const [selectedScenario, setSelectedScenario] = useState("normal_traffic");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/benchmark/results`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load benchmark results");
        return res.json();
      })
      .then((data) => {
        setAllData(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const scenarioData = allData?.scenarios?.[selectedScenario] ?? null;
  const limitations = allData?.known_limitations ?? null;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex flex-col items-center gap-3 text-app-muted">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-app-border border-t-app-accent" />
          <p className="text-sm">Loading benchmark results…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-400">
        Failed to load benchmark data: {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-app-text">
            Algorithm Performance Benchmark
          </h2>
          <p className="mt-1 text-xs text-app-muted max-w-2xl leading-relaxed">
            {FINDINGS_SUMMARY}
          </p>
        </div>

        {/* Scenario selector */}
        <label className="text-xs text-app-muted">
          Scenario
          <select
            className="mt-1 block h-9 w-64 rounded border border-app-border bg-app-input-bg px-2 text-sm text-app-text outline-none focus:border-app-accent"
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value)}
          >
            {Object.entries(SCENARIO_LABELS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Shared result view */}
      <BenchmarkResultView
        scenarioData={scenarioData}
        scenarioLabel={SCENARIO_LABELS[selectedScenario] || selectedScenario}
        showLimitations={true}
        limitations={limitations}
      />
    </div>
  );
}
