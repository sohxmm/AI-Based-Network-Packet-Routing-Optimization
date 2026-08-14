/**
 * BenchmarkReport renders whatever the API returns, including the parts the
 * project would rather not show. These tests pin the parts that matter:
 * the guardrail warnings must appear above the table, and an empty results
 * directory must say so plainly instead of rendering a blank panel.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

import BenchmarkReport from "../BenchmarkReport.jsx";

vi.mock("recharts", () => {
  const Passthrough = ({ children }) => <div>{children}</div>;
  return {
    ResponsiveContainer: Passthrough,
    BarChart: Passthrough,
    Bar: Passthrough,
    Cell: () => null,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
    LabelList: () => null,
    ErrorBar: () => null,
    ReferenceLine: () => null,
  };
});

function mockFetch(payload) {
  global.fetch = vi.fn(() =>
    Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(payload) })
  );
}

const SCENARIO = {
  scenario: "normal_traffic",
  description: "Baseline.",
  topology: { num_nodes: 25, num_edges: 50, avg_degree: 4, diameter: 6 },
  replication: { n_runs: 30, n_steps: 100, m_pairs: 20 },
  warnings: ["rl: 100% of decisions came from the heuristic fallback, not a trained model."],
  algorithms: {
    dijkstra: {
      mean_latency: 60.1,
      success_rate: 1,
      fallback_rate: 0,
      qos_satisfaction_rate: 0.98,
      diversity_index: 0.11,
      p95_path_max_utilization: 0.62,
      dijkstra_match_rate: 1,
      mean_latency_ci: { ci95_low: 59, ci95_high: 61 },
    },
    rl: {
      mean_latency: 63.4,
      success_rate: 1,
      fallback_rate: 1,
      qos_satisfaction_rate: 0.95,
      diversity_index: 0.2,
      p95_path_max_utilization: 0.66,
      dijkstra_match_rate: 0.4,
      mean_latency_ci: { ci95_low: 62, ci95_high: 65 },
      comparison_vs_dijkstra: {
        n_runs: 30,
        mean_diff: 3.3,
        pct_diff: 5.5,
        cliffs_delta: 0.31,
        effect_magnitude: "small",
        ci95_low: 2.1,
        ci95_high: 4.5,
        wilcoxon_p_value: 0.002,
      },
    },
  },
};

describe("BenchmarkReport", () => {
  it("surfaces guardrail warnings above the results", async () => {
    mockFetch({ scenarios: { normal_traffic: SCENARIO }, known_limitations: {} });
    render(<BenchmarkReport />);
    await waitFor(() =>
      expect(screen.getByText(/heuristic fallback, not a trained model/i)).toBeTruthy()
    );
  });

  it("reports Cliff's delta rather than a percent difference labelled as effect size", async () => {
    mockFetch({ scenarios: { normal_traffic: SCENARIO }, known_limitations: {} });
    render(<BenchmarkReport />);
    await waitFor(() => expect(screen.getByText("0.310")).toBeTruthy());
    expect(screen.getByText("small")).toBeTruthy();
  });

  it("states the unit of replication", async () => {
    mockFetch({ scenarios: { normal_traffic: SCENARIO }, known_limitations: {} });
    render(<BenchmarkReport />);
    await waitFor(() =>
      expect(screen.getByText(/30 independent seeded runs/i)).toBeTruthy()
    );
  });

  it("explains what to do when no results exist", async () => {
    mockFetch({ scenarios: {}, known_limitations: {} });
    render(<BenchmarkReport />);
    await waitFor(() =>
      expect(screen.getByText(/No benchmark results found/i)).toBeTruthy()
    );
  });
});
