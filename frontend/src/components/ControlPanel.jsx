// TODO: implement
import { RotateCcw, Send, StepForward, Workflow } from "lucide-react";

function ControlPanel() {
  return (
    <section className="flex flex-wrap items-center gap-2 rounded border border-slate-800 bg-slate-900 p-4">
      <button className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm">
        <StepForward className="h-4 w-4" aria-hidden="true" />
        Step +1
      </button>
      <button className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm">
        <Send className="h-4 w-4" aria-hidden="true" />
        Step +10
      </button>
      <button className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm">
        <Workflow className="h-4 w-4" aria-hidden="true" />
        Inject Failure
      </button>
      <button className="inline-flex h-9 items-center gap-2 rounded border border-slate-700 px-3 text-sm">
        <RotateCcw className="h-4 w-4" aria-hidden="true" />
        Restore Link
      </button>
    </section>
  );
}

export default ControlPanel;
