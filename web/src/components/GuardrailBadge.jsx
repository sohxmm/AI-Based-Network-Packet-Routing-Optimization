/**
 * GuardrailBadge — the component whose entire job is to advertise this
 * project's own failure modes on its own results.
 *
 * It renders "Heuristic fallback used" when an AI router did not actually run a
 * model, and "Matches Dijkstra, no differentiation" when a learned algorithm
 * chose the identical path to the baseline. Building UI that undermines your own
 * headline is uncomfortable and it is the right thing to do: a reviewer who has
 * to discover a limitation themselves assumes it was hidden.
 *
 * Two APIs, deliberately:
 *  - `type` — the original preset form ("fallback", "dijkstra-match"), kept so
 *    existing components and tests keep working.
 *  - `tone` + `label` — a general form for the QoS and status badges added
 *    alongside the multi-class routing work.
 */

const PRESETS = {
  fallback: {
    tone: "bad",
    label: "Heuristic fallback used",
    icon: "⚠",
    title:
      "No trained model was loaded, so this decision came from a heuristic. " +
      "It is not an AI result.",
  },
  "dijkstra-match": {
    tone: "warn",
    label: "Matches Dijkstra, no differentiation",
    icon: "≡",
    title:
      "This algorithm returned the same path as the classical baseline, so it " +
      "added no information here.",
  },
};

const TONES = {
  ok: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  warn: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  bad: "bg-red-500/15 text-red-400 border-red-500/30",
  neutral: "bg-slate-500/15 text-app-muted border-app-border",
};

const TONE_ICONS = { ok: "✓", warn: "⚠", bad: "⚠", neutral: "≡" };

export default function GuardrailBadge({
  type,
  tone,
  label,
  title,
  icon,
  compact = false,
}) {
  const preset = type ? PRESETS[type] : null;
  if (type && !preset) return null;

  const resolvedTone = preset?.tone ?? tone ?? "neutral";
  const resolvedLabel = preset?.label ?? label;
  if (!resolvedLabel) return null;

  const resolvedIcon = icon ?? preset?.icon ?? TONE_ICONS[resolvedTone];
  const resolvedTitle = title ?? preset?.title ?? resolvedLabel;

  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 " +
        "text-[10px] font-medium leading-tight whitespace-nowrap " +
        (TONES[resolvedTone] ?? TONES.neutral)
      }
      title={resolvedTitle}
    >
      <span aria-hidden="true">{resolvedIcon}</span>
      {!compact && <span>{resolvedLabel}</span>}
    </span>
  );
}
