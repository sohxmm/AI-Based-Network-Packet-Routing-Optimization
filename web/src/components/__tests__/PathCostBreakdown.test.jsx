/**
 * The honesty badges are the most important thing this UI renders, so they get
 * the most direct test: a fallback decision must *say* it is a fallback, a
 * decision identical to Dijkstra must say it added nothing, and a route that
 * breaks its traffic class's constraints must say so.
 *
 * If these ever regress silently, the dashboard goes back to presenting
 * heuristic output as AI output — the exact failure this guards against.
 */

import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

import PathCostBreakdown from "../PathCostBreakdown.jsx";

const baseline = {
  algorithm: "dijkstra",
  path: ["R1", "R2", "R3"],
  total_latency: 40,
  is_fallback: false,
  hops: [
    { from: "R1", to: "R2", base_latency: 10, utilization: 0.2, cost: 16.4 },
    { from: "R2", to: "R3", base_latency: 14, utilization: 0.3, cost: 23.6 },
  ],
  diagnostics: { qos: feasibleQos() },
};

function feasibleQos() {
  return {
    feasible: true,
    score: 1.2,
    total_loss: 0.001,
    bottleneck_utilization: 0.3,
    hops: 2,
    violations: [],
  };
}

describe("PathCostBreakdown", () => {
  it("flags a decision that came from the heuristic fallback", () => {
    render(
      <PathCostBreakdown
        result={{
          ...baseline,
          algorithm: "rl",
          path: ["R1", "R5", "R3"],
          is_fallback: true,
        }}
        baseline={baseline}
        trafficClass="best_effort"
      />
    );
    expect(screen.getByText(/Heuristic fallback/i)).toBeTruthy();
  });

  it("flags an algorithm that just reproduced Dijkstra's path", () => {
    render(
      <PathCostBreakdown
        result={{ ...baseline, algorithm: "gnn" }}
        baseline={baseline}
        trafficClass="best_effort"
      />
    );
    expect(screen.getByText(/Matches Dijkstra/i)).toBeTruthy();
  });

  it("does not flag Dijkstra itself as matching Dijkstra", () => {
    render(
      <PathCostBreakdown result={baseline} baseline={baseline} trafficClass="best_effort" />
    );
    expect(screen.queryByText(/Matches Dijkstra/i)).toBeNull();
  });

  it("reports a QoS constraint violation", () => {
    render(
      <PathCostBreakdown
        result={{
          ...baseline,
          algorithm: "gnn",
          path: ["R1", "R9", "R3"],
          diagnostics: {
            qos: {
              ...feasibleQos(),
              feasible: false,
              bottleneck_utilization: 0.92,
              violations: ["bottleneck 0.920 > 0.700"],
            },
          },
        }}
        baseline={baseline}
        trafficClass="emergency"
      />
    );
    expect(screen.getByText(/Violates emergency QoS/i)).toBeTruthy();
  });

  it("shows the signed delta against the baseline", () => {
    render(
      <PathCostBreakdown
        result={{
          ...baseline,
          algorithm: "aco",
          path: ["R1", "R7", "R3"],
          total_latency: 52,
        }}
        baseline={baseline}
        trafficClass="best_effort"
      />
    );
    expect(screen.getByText(/\+12\.0 ms/)).toBeTruthy();
  });

  it("expands to a per-hop cost breakdown on click", () => {
    render(
      <PathCostBreakdown result={baseline} baseline={baseline} trafficClass="best_effort" />
    );
    expect(screen.queryByText("R1 → R2")).toBeNull();
    fireEvent.click(screen.getByRole("button", { expanded: false }));
    expect(screen.getByText("R1 → R2")).toBeTruthy();
  });
});
