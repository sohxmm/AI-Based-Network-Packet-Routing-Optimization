import * as d3 from "d3";

/**
 * Chart colour, decided by the job each colour does rather than by taste.
 *
 * The categorical slots below are a validated eight-hue set: adjacent pairs
 * clear the colour-vision-deficiency separation floor and the lightness/chroma
 * bands in both light and dark mode. Three light-mode slots sit below 3:1
 * contrast on a pale surface, which is why every chart in this dashboard also
 * carries direct value labels and a table view — identity is never conveyed by
 * colour alone.
 *
 * Rules this file exists to enforce:
 *   - categorical hues are assigned in fixed order and never cycled;
 *   - colour follows the algorithm, not its rank, so filtering the comparison
 *     never repaints the survivors;
 *   - magnitude uses one hue light-to-dark, polarity uses two hues with a
 *     neutral midpoint, and status colours are reserved.
 */

/** Green -> amber -> orange -> red as a link approaches saturation. */
export const utilizationColor = d3
  .scaleLinear()
  .domain([0, 0.4, 0.7, 1])
  .range(["#22c55e", "#eab308", "#f97316", "#ef4444"]);

// --- categorical: identity ------------------------------------------------
const CATEGORICAL_LIGHT = {
  dijkstra: "#2a78d6", // blue
  bellman_ford: "#4a3aa7", // violet
  constrained: "#1baf7a", // aqua
  aco: "#eb6834", // orange
  gnn: "#e87ba4", // magenta
  rl: "#008300", // green
  multi_agent: "#eda100", // yellow
  random_baseline: "#e34948", // red
};

const CATEGORICAL_DARK = {
  dijkstra: "#3987e5",
  bellman_ford: "#9085e9",
  constrained: "#199e70",
  aco: "#d95926",
  gnn: "#d55181",
  rl: "#008300",
  multi_agent: "#c98500",
  random_baseline: "#e66767",
};

/**
 * A predictive variant is the *same algorithm* in a different mode, so it takes
 * the same hue and is separated by a dash pattern instead of a ninth colour.
 */
const VARIANT_OF = { gnn_predictive: "gnn", rl_predictive: "rl" };

export function algorithmColor(name, isDark = true) {
  const table = isDark ? CATEGORICAL_DARK : CATEGORICAL_LIGHT;
  return table[name] ?? table[VARIANT_OF[name]] ?? (isDark ? "#c3c2b7" : "#52514e");
}

export function isVariant(name) {
  return name in VARIANT_OF;
}

// --- sequential: magnitude (one hue, light -> dark) -----------------------
export const SEQUENTIAL_BLUE = [
  "#cde2fb",
  "#9ec5f4",
  "#6da7ec",
  "#3987e5",
  "#256abf",
  "#184f95",
];

// --- diverging: polarity (two hues + neutral midpoint) --------------------
// Blue for better, orange for worse. Deliberately not red/green, which is the
// single worst pair for colour-vision deficiency.
export const DIVERGING = {
  better: { light: "#2a78d6", dark: "#3987e5" },
  worse: { light: "#eb6834", dark: "#d95926" },
  neutral: { light: "#f0efec", dark: "#383835" },
};

export function deltaColor(delta, isDark = true) {
  if (delta == null || Math.abs(delta) < 1e-9) {
    return isDark ? DIVERGING.neutral.dark : DIVERGING.neutral.light;
  }
  const pole = delta < 0 ? DIVERGING.better : DIVERGING.worse;
  return isDark ? pole.dark : pole.light;
}

// --- status: reserved, never reused as a series colour --------------------
export const STATUS_COLORS = {
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
};

// --- labels ---------------------------------------------------------------
export const ALGORITHM_LABELS = {
  dijkstra: "Dijkstra",
  bellman_ford: "Bellman-Ford",
  constrained: "Constrained k-shortest",
  aco: "Ant Colony",
  gnn: "GNN",
  rl: "RL (PPO)",
  multi_agent: "Multi-Agent RL",
  random_baseline: "Random baseline",
  gnn_predictive: "GNN (predictive)",
  rl_predictive: "RL (predictive)",
};

export function algorithmLabel(name) {
  return ALGORITHM_LABELS[name] ?? name;
}

export const TRAFFIC_CLASS_LABELS = {
  emergency: "Emergency",
  interactive: "Voice / video",
  gaming: "Gaming",
  bulk: "Bulk transfer",
  best_effort: "Best effort",
};

export const SCENARIO_LABELS = {
  normal_traffic: "Normal traffic",
  high_congestion: "High congestion",
  link_failures_persistent: "Persistent link failures",
  cascading_failure: "Cascading failure",
  congestion_bursts: "Congestion bursts",
  large_topology_100_nodes: "Large topology (100 nodes)",
  qos_mixed_traffic: "Mixed QoS traffic",
};

export function scenarioLabel(name) {
  return SCENARIO_LABELS[name] ?? name.replace(/_/g, " ");
}
