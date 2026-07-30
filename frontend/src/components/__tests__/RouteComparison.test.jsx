/**
 * RouteComparison.test.jsx — Tests for honesty badges in the live comparison view.
 *
 * Verifies:
 *  - "Heuristic fallback used" badge renders when is_fallback=true
 *  - "Matches Dijkstra — no differentiation" badge renders when an AI algo's
 *    path exactly matches Dijkstra's path
 */

import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import RouteComparison from "../RouteComparison.jsx";

// Mock Recharts to avoid SVG rendering issues in jsdom
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
}));

const mockNetworkState = {
  nodes: ["R1", "R2", "R3", "R4", "R5"],
  links: [],
  step_count: 10,
  timestamp: Date.now(),
};

describe("RouteComparison badges", () => {
  it("renders fallback badge when is_fallback is true", () => {
    const comparison = {
      source: "R1",
      destination: "R5",
      step_count: 10,
      results: [
        {
          algorithm: "dijkstra",
          path: ["R1", "R3", "R5"],
          total_latency: 12.5,
          success: true,
          is_fallback: false,
        },
        {
          algorithm: "rl",
          path: ["R1", "R2", "R4", "R5"],
          total_latency: 15.0,
          success: true,
          is_fallback: true,
        },
      ],
    };

    render(
      <RouteComparison
        networkState={mockNetworkState}
        comparison={comparison}
        isLoading={false}
        onCompare={() => {}}
      />
    );

    // The fallback badge should be present with its icon
    const badges = screen.getAllByTitle("Heuristic fallback used");
    expect(badges.length).toBeGreaterThan(0);
  });

  it("renders dijkstra-match badge when AI path matches Dijkstra path", () => {
    const sharedPath = ["R1", "R3", "R5"];
    const comparison = {
      source: "R1",
      destination: "R5",
      step_count: 10,
      results: [
        {
          algorithm: "dijkstra",
          path: sharedPath,
          total_latency: 12.5,
          success: true,
          is_fallback: false,
        },
        {
          algorithm: "gnn",
          path: sharedPath, // Exactly matches Dijkstra
          total_latency: 12.5,
          success: true,
          is_fallback: false,
        },
      ],
    };

    render(
      <RouteComparison
        networkState={mockNetworkState}
        comparison={comparison}
        isLoading={false}
        onCompare={() => {}}
      />
    );

    const matchBadges = screen.getAllByTitle("Matches Dijkstra — no differentiation");
    expect(matchBadges.length).toBeGreaterThan(0);
  });

  it("does NOT render dijkstra-match badge for bellman_ford (excluded)", () => {
    const sharedPath = ["R1", "R3", "R5"];
    const comparison = {
      source: "R1",
      destination: "R5",
      step_count: 10,
      results: [
        {
          algorithm: "dijkstra",
          path: sharedPath,
          total_latency: 12.5,
          success: true,
          is_fallback: false,
        },
        {
          algorithm: "bellman_ford",
          path: sharedPath, // Same path, but bellman_ford is excluded from check
          total_latency: 12.5,
          success: true,
          is_fallback: false,
        },
      ],
    };

    render(
      <RouteComparison
        networkState={mockNetworkState}
        comparison={comparison}
        isLoading={false}
        onCompare={() => {}}
      />
    );

    const matchBadges = screen.queryAllByTitle("Matches Dijkstra — no differentiation");
    expect(matchBadges.length).toBe(0);
  });

  it("does NOT render dijkstra-match badge when paths differ", () => {
    const comparison = {
      source: "R1",
      destination: "R5",
      step_count: 10,
      results: [
        {
          algorithm: "dijkstra",
          path: ["R1", "R3", "R5"],
          total_latency: 12.5,
          success: true,
          is_fallback: false,
        },
        {
          algorithm: "aco",
          path: ["R1", "R2", "R4", "R5"], // Different path
          total_latency: 14.0,
          success: true,
          is_fallback: false,
        },
      ],
    };

    render(
      <RouteComparison
        networkState={mockNetworkState}
        comparison={comparison}
        isLoading={false}
        onCompare={() => {}}
      />
    );

    const matchBadges = screen.queryAllByTitle("Matches Dijkstra — no differentiation");
    expect(matchBadges.length).toBe(0);
  });
});
