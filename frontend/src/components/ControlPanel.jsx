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
    <section className="flex flex-wrap items-center gap-2 rounded border border-slate-800 bg-slate-900/90 p-4">
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm disabled:opacity-50"
        disabled={isLoading}
        onClick={() => stepMany(1)}
        type="button"
      >
        <StepForward className="h-4 w-4" aria-hidden="true" />
        Step +1
      </button>
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm disabled:opacity-50"
        disabled={isLoading}
        onClick={() => stepMany(10)}
        type="button"
      >
        <Send className="h-4 w-4" aria-hidden="true" />
        Step +10
      </button>
      <select
        className="h-9 min-w-36 rounded border border-slate-700 bg-slate-950 px-2 text-sm text-slate-100"
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
        className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm disabled:opacity-50"
        disabled={isLoading || !activeLink}
        onClick={() => sendLinkAction("/sim/inject-failure", "Injected link failure.")}
        type="button"
      >
        <Workflow className="h-4 w-4" aria-hidden="true" />
        Inject Failure
      </button>
      <button
        className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm disabled:opacity-50"
        disabled={isLoading || !activeLink}
        onClick={() => sendLinkAction("/sim/restore-link", "Restored selected link.")}
        type="button"
      >
        <RotateCcw className="h-4 w-4" aria-hidden="true" />
        Restore Link
      </button>
      {lastAction && <span className="text-sm text-slate-400">{lastAction}</span>}
    </section>
  );
}

export default ControlPanel;
