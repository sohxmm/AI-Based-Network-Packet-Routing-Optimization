import { Layers } from "lucide-react";

function LeftPanel({ networkState }) {
  const data = (networkState?.links ?? [])
    .map((link) => ({
      name: `${link.source}-${link.target}`,
      queue: link.queue_size ?? 0
    }))
    .sort((a, b) => b.queue - a.queue)
    .slice(0, 5);

  return (
    <section className="flex flex-col rounded border border-app-border bg-app-panel p-4">
      <div className="flex items-center justify-between gap-3 pb-3 border-b border-app-border">
        <h2 className="text-sm font-semibold text-app-text">Top Queue Sizes</h2>
        <Layers className="h-4 w-4 text-app-accent" aria-hidden="true" />
      </div>
      <div className="mt-3 flex flex-col gap-2">
        {data.length ? (
          data.map((link) => (
            <div key={link.name} className="flex items-center justify-between">
              <span className="text-sm font-medium text-app-muted">{link.name}</span>
              <span className="text-sm font-semibold text-app-text">{link.queue} pkts</span>
            </div>
          ))
        ) : (
          <div className="py-4 text-center text-sm text-app-muted">No data</div>
        )}
      </div>
    </section>
  );
}

export default LeftPanel;
