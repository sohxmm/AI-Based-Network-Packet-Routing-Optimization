import { scenarioLabel } from "../../utils/colorScales.js";

/** Filter row, above the charts, one row, never a sidebar of checkboxes. */
function ScenarioSelector({ scenarios, active, onSelect }) {
  if (!scenarios?.length) return null;

  return (
    <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Benchmark scenario">
      {scenarios.map((name) => (
        <button
          key={name}
          type="button"
          role="tab"
          aria-selected={name === active}
          onClick={() => onSelect(name)}
          className={`rounded border px-2.5 py-1 text-xs transition-colors ${
            name === active
              ? "border-app-accent bg-app-accent/15 text-app-text"
              : "border-app-border text-app-muted hover:text-app-text"
          }`}
        >
          {scenarioLabel(name)}
        </button>
      ))}
    </div>
  );
}

export default ScenarioSelector;
