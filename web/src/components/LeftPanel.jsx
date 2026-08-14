import { memo } from "react";
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


/**
 * Memoized on step_count and theme.
 *
 * The backend broadcasts a full network state once per second and every
 * consumer re-renders. TopologyGraph handles that deliberately; these panels
 * did not, and at 100 nodes the cascade is visible. The payload object is
 * replaced every tick, so a default shallow compare never helps - the
 * comparator has to key on the tick counter.
 */
const MemoizedLeftPanel = memo(LeftPanel, (prev, next) =>
    prev.networkState?.step_count === next.networkState?.step_count &&
    prev.comparison === next.comparison);

export default MemoizedLeftPanel;
