import { Network } from "lucide-react";

function RightPanel({ networkState }) {
  const nodesCount = networkState?.nodes?.length ?? 0;
  const linksCount = networkState?.links?.length ?? 0;
  
  const avgCapacity = linksCount > 0
    ? (networkState.links.reduce((acc, link) => acc + link.bandwidth, 0) / linksCount).toFixed(0)
    : 0;

  const congestedLinksCount = networkState?.links?.filter((l) => l.utilization >= 0.7).length ?? 0;

  return (
    <section className="flex flex-col rounded border border-app-border bg-app-panel p-4">
      <div className="flex items-center justify-between gap-3 pb-3 border-b border-app-border">
        <h2 className="text-sm font-semibold text-app-text">Global Topology Stats</h2>
        <Network className="h-4 w-4 text-app-accent" aria-hidden="true" />
      </div>
      <div className="mt-3 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-app-muted">Total Routers</span>
          <span className="text-sm font-semibold text-app-text">{nodesCount}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-app-muted">Active Links</span>
          <span className="text-sm font-semibold text-app-text">{linksCount}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-app-muted">Avg Link Capacity</span>
          <span className="text-sm font-semibold text-app-text">{avgCapacity} pkts</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-app-muted">Congested Links</span>
          <span className="text-sm font-semibold text-app-text">{congestedLinksCount}</span>
        </div>
      </div>
    </section>
  );
}

export default RightPanel;
