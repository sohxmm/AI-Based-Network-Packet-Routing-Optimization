/**
 * GuardrailBadge — Shared honesty badge component for benchmark warnings.
 *
 * Used in both BenchmarkReport (Part 2) and RouteComparison (Part 3).
 *
 * Types:
 *  - "fallback"       → red/amber badge: "Heuristic fallback used"
 *  - "dijkstra-match"  → yellow badge: "Matches Dijkstra — no differentiation"
 */

import React from "react";

const BADGE_CONFIG = {
  fallback: {
    label: "Heuristic fallback used",
    className:
      "bg-red-500/20 text-red-400 border border-red-500/30",
    icon: "⚠",
  },
  "dijkstra-match": {
    label: "Matches Dijkstra — no differentiation",
    className:
      "bg-amber-500/20 text-amber-400 border border-amber-500/30",
    icon: "≡",
  },
};

export default function GuardrailBadge({ type, compact = false }) {
  const config = BADGE_CONFIG[type];
  if (!config) return null;

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium leading-tight whitespace-nowrap ${config.className}`}
      title={config.label}
    >
      <span aria-hidden="true">{config.icon}</span>
      {!compact && <span>{config.label}</span>}
    </span>
  );
}
