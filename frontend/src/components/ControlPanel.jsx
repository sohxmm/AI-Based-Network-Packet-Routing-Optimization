import { useMemo, useState } from "react";

import { RotateCcw, Send, StepForward, Workflow } from "lucide-react";

function ControlPanel({ networkState, isLoading, onSimulatorAction }) {
  const links = networkState?.links ?? [];
  const [selectedLink, setSelectedLink] = useState("");
  const [lastAction, setLastAction] = useState("");

  const linkOptions = useMemo(
    () => links.map((link) => `${link.source}|${link.target}`),
    [links]
  );
  const activeLink = selectedLink || linkOptions[0] || "";

  async function stepMany(count) {
    let result = null;
    for (let index = 0; index < count; index += 1) {
      result = await onSimulatorAction("/sim/step");
      if (!result) {
        return;
      }
    }
    setLastAction(`Advanced ${count} step${count === 1 ? "" : "s"}.`);
  }

  async function sendLinkAction(path, label) {
    const [source, target] = activeLink.split("|");
    const result = await onSimulatorAction(path, { source, target });
    if (result) {
      setLastAction(label);
    }
  }

  return (
    <section className="flex flex-wrap items-center gap-2 rounded border border-app-border bg-app-panel p-4">
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-app-border bg-transparent px-3 text-sm text-app-text hover:bg-app-input-bg disabled:opacity-50"
        disabled={isLoading}
        onClick={() => stepMany(1)}
        type="button"
      >
        <StepForward className="h-4 w-4" aria-hidden="true" />
        Step +1
      </button>
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-app-border bg-transparent px-3 text-sm text-app-text hover:bg-app-input-bg disabled:opacity-50"
        disabled={isLoading}
        onClick={() => stepMany(10)}
        type="button"
      >
        <Send className="h-4 w-4" aria-hidden="true" />
        Step +10
      </button>
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-red-500/50 text-red-500 bg-transparent px-3 text-sm hover:bg-red-500/10 disabled:opacity-50"
        disabled={isLoading}
        onClick={async () => {
          const result = await onSimulatorAction("/sim/reset");
          if (result) setLastAction("Reset simulation.");
        }}
        type="button"
      >
        <RotateCcw className="h-4 w-4" aria-hidden="true" />
        Reset
      </button>
      <select
        className="h-9 min-w-36 rounded border border-app-border bg-app-input-bg px-2 text-sm text-app-text"
        value={activeLink}
        onChange={(event) => setSelectedLink(event.target.value)}
      >
        {linkOptions.map((link) => {
          const [source, target] = link.split("|");
          return (
            <option key={link} value={link}>
              {source}-{target}
            </option>
          );
        })}
      </select>
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-app-border bg-transparent px-3 text-sm text-app-text hover:bg-app-input-bg disabled:opacity-50"
        disabled={isLoading || !activeLink}
        onClick={() => sendLinkAction("/sim/inject-failure", "Injected link failure.")}
        type="button"
      >
        <Workflow className="h-4 w-4" aria-hidden="true" />
        Inject Failure
      </button>
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-app-border bg-transparent px-3 text-sm text-app-text hover:bg-app-input-bg disabled:opacity-50"
        disabled={isLoading || !activeLink}
        onClick={() => sendLinkAction("/sim/restore-link", "Restored selected link.")}
        type="button"
      >
        <RotateCcw className="h-4 w-4" aria-hidden="true" />
        Restore Link
      </button>
      {lastAction && <span className="text-sm text-app-muted">{lastAction}</span>}
    </section>
  );
}

export default ControlPanel;
