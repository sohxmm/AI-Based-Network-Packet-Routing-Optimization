/**
 * BenchmarkReport.test.jsx — Tests for the benchmark report view.
 *
 * Verifies:
 *  - Renders scenario selector
 *  - Renders metrics table with expected columns
 *  - Guardrail banners render for high fallback / degeneracy cases
 */

import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { BenchmarkResultView } from "../BenchmarkReport.jsx";

// Mock Recharts
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

const mockScenarioData = {
  scenario: "high_congestion",
  n_steps: 1000,
  m_pairs: 20,
  algorithms: {
    dijkstra: {
      mean_latency: 60.0,
      p95_latency: 117.0,
      util_variance: 0.035,
      success_rate: 1.0,
      fallback_rate: 0.0,
      dijkstra_match_rate: 1.0,
      wilcoxon_p_value: null,
      effect_size_pct: 0.0,
    },
    rl: {
      mean_latency: 83.7,
      p95_latency: 183.8,
      util_variance: 0.048,
      success_rate: 1.0,
      fallback_rate: 0.12, // High fallback — should trigger red banner
      dijkstra_match_rate: 0.47,
      wilcoxon_p_value: 0.0,
      effect_size_pct: 39.5,
    },
    gnn: {
      mean_latency: 61.0,
      p95_latency: 120.0,
      util_variance: 0.036,
      success_rate: 1.0,
      fallback_rate: 0.0,
      dijkstra_match_rate: 0.95, // High Dijkstra match — should trigger yellow banner
      wilcoxon_p_value: 0.001,
      effect_size_pct: 1.7,
    },
  },
};

describe("BenchmarkResultView", () => {
  it("renders metrics table with algorithm rows", () => {
    render(
      <BenchmarkResultView
        scenarioData={mockScenarioData}
        scenarioLabel="High Congestion"
      />
    );

    // Check table headers exist
    expect(screen.getByText("Algorithm")).toBeTruthy();
    expect(screen.getByText("Mean Latency")).toBeTruthy();
    expect(screen.getByText("p95 Latency")).toBeTruthy();
    expect(screen.getByText("Util Var")).toBeTruthy();
    expect(screen.getByText("Effect vs Dijkstra")).toBeTruthy();

    // Check algorithm names appear (may appear multiple times due to summary card)
    expect(screen.getAllByText("Dijkstra").length).toBeGreaterThan(0);
    expect(screen.getAllByText("RL").length).toBeGreaterThan(0);
    expect(screen.getAllByText("GNN").length).toBeGreaterThan(0);
  });

  it("renders red fallback guardrail banner for high fallback rate", () => {
    render(
      <BenchmarkResultView
        scenarioData={mockScenarioData}
        scenarioLabel="High Congestion"
      />
    );

    // RL has 12% fallback rate — should trigger a red banner mentioning it
    const banners = document.querySelectorAll('[class*="red"]');
    expect(banners.length).toBeGreaterThan(0);

    // Check the text mentions RL and fallback
    const bannerTexts = Array.from(banners).map((b) => b.textContent);
    const hasRlFallback = bannerTexts.some(
      (t) => t.includes("RL") && t.includes("fallback")
    );
    expect(hasRlFallback).toBe(true);
  });

  it("renders yellow degeneracy banner for high Dijkstra match rate", () => {
    render(
      <BenchmarkResultView
        scenarioData={mockScenarioData}
        scenarioLabel="High Congestion"
      />
    );

    // GNN has 95% Dijkstra match — should trigger a yellow banner
    const banners = document.querySelectorAll('[class*="amber"]');
    const bannerTexts = Array.from(banners).map((b) => b.textContent);
    const hasGnnDegen = bannerTexts.some(
      (t) => t.includes("GNN") && t.includes("Dijkstra")
    );
    expect(hasGnnDegen).toBe(true);
  });

  it("renders large topology limitation banner", () => {
    const largeTopoData = {
      ...mockScenarioData,
      scenario: "large_topology_100_nodes",
    };

    render(
      <BenchmarkResultView
        scenarioData={largeTopoData}
        scenarioLabel="Large Topology (100 Nodes)"
      />
    );

    expect(screen.getByText(/Large Topology Limitation/)).toBeTruthy();
  });

  it("renders grouped bar chart", () => {
    render(
      <BenchmarkResultView
        scenarioData={mockScenarioData}
        scenarioLabel="High Congestion"
      />
    );

    expect(screen.getByText("Latency vs. Utilization Variance Trade-off")).toBeTruthy();
    expect(screen.getByTestId("bar-chart")).toBeTruthy();
  });

  it("shows 'baseline' for Dijkstra effect size column", () => {
    render(
      <BenchmarkResultView
        scenarioData={mockScenarioData}
        scenarioLabel="High Congestion"
      />
    );

    expect(screen.getByText("baseline")).toBeTruthy();
  });

  it("renders no data message when scenarioData is null", () => {
    render(
      <BenchmarkResultView scenarioData={null} scenarioLabel="Test" />
    );

    expect(screen.getByText(/No benchmark data available/)).toBeTruthy();
  });
});
