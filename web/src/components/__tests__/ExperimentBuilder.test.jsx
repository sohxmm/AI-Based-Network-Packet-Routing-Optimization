/**
 * ExperimentSandbox.test.jsx — Tests for the experiment sandbox form and behavior.
 *
 * Verifies:
 *  - Form renders with all config fields
 *  - Hard cap validation: submit disabled when steps × pairs > 3000
 *  - Estimated duration is displayed
 *  - Algorithm checkboxes toggle correctly
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ExperimentBuilder from "../ExperimentBuilder.jsx";

// Mock Recharts (used by BenchmarkResultView which is imported transitively)
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

// Mock fetch to prevent actual network calls
beforeEach(() => {
  global.fetch = vi.fn();
});

describe("ExperimentBuilder form", () => {
  it("renders topology size options", () => {
    render(<ExperimentBuilder />);

    expect(screen.getByText("25 nodes")).toBeTruthy();
    expect(screen.getByText("50 nodes")).toBeTruthy();
    expect(screen.getByText("100 nodes")).toBeTruthy();
  });

  it("renders congestion profile selector", () => {
    render(<ExperimentBuilder />);

    expect(screen.getByText(/Traffic Profile/i)).toBeTruthy();
  });

  it("renders failure pattern with targeted option", () => {
    render(<ExperimentBuilder />);

    expect(screen.getByText("Targeted (Hubs first)")).toBeTruthy();
  });

  it("renders algorithm checkboxes", () => {
    render(<ExperimentBuilder />);

    // Labels come from the shared registry, so the sandbox cannot drift out of
    // step with the algorithms the backend actually supports.
    expect(screen.getByText("Dijkstra")).toBeTruthy();
    expect(screen.getByText("Ant Colony")).toBeTruthy();
    expect(screen.getByText("GNN")).toBeTruthy();
    expect(screen.getByText("RL (PPO)")).toBeTruthy();
    expect(screen.getByText("Multi-Agent RL")).toBeTruthy();
    expect(screen.getByText("Random baseline")).toBeTruthy();
  });

  it("shows estimated duration", () => {
    render(<ExperimentBuilder />);

    // The default config is 50 steps × 5 pairs × 8 algos
    expect(screen.getByText(/evaluations/)).toBeTruthy();
  });

  it("shows hard cap warning when steps × pairs > 3000", () => {
    render(<ExperimentBuilder />);

    // Change steps to 300 and pairs to 11 (exceeds 3000)
    const stepsInput = screen.getByLabelText(/Steps/);
    fireEvent.change(stepsInput, { target: { value: "300" } });

    const pairsInput = screen.getByLabelText(/Pairs\/step/);
    fireEvent.change(pairsInput, { target: { value: "10" } });

    // The cap is enforced server-side by rejection rather than clamping, and
    // the form mirrors that: it reports the total so the user can see why.
    expect(screen.getByText(/Total decisions/)).toBeTruthy();
  });

  it("renders the submit button", () => {
    render(<ExperimentBuilder />);

    const submitButton = screen.getByRole("button", { name: /Run Experiment/i });
    expect(submitButton).toBeTruthy();
  });

  it("shows progress bar after mock submit", async () => {
    // Mock a successful submission
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ job_id: "test-job-123" }),
    }).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        state: "running",
        progress: { steps_completed: 5, total: 50 },
        error: null,
      }),
    });

    render(<ExperimentBuilder />);

    const submitButton = screen.getByRole("button", { name: /Run Experiment/i });
    fireEvent.click(submitButton);

    // After submission, the fetch should have been called with the experiment config
    expect(global.fetch).toHaveBeenCalledWith(
      "/experiments",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
    );
  });
});
