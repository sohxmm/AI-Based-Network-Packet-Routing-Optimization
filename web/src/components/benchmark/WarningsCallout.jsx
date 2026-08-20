/**
 * The guardrail warnings, rendered above the results table rather than hidden
 * in a tooltip.
 *
 * The benchmark emits a warning whenever an algorithm ran mostly on its
 * heuristic fallback, or chose the same path as Dijkstra almost every time. A
 * reader should meet those caveats *before* the numbers, not after.
 */
function WarningsCallout({ warnings }) {
  if (!warnings?.length) return null;

  return (
    <div className="rounded border border-amber-500/40 bg-amber-500/10 px-3 py-2">
      <p className="text-xs font-semibold text-amber-400">
        <span aria-hidden="true">⚠ </span>
        Read these before the table
      </p>
      <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] text-amber-300/90">
        {warnings.map((warning) => (
          <li key={warning}>{warning}</li>
        ))}
      </ul>
    </div>
  );
}

export default WarningsCallout;
