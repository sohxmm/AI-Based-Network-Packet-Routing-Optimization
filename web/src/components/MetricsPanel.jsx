import { memo } from "react";
import { Activity, Gauge, RadioTower, Route } from "lucide-react";

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function getMetrics(networkState, comparison) {
  const links = networkState?.links ?? [];
  const successfulResults = comparison?.results?.filter((result) => result.success) ?? [];
  const bestResult = successfulResults.reduce((best, result) => {
    if (!best) {
      return result;
    }
    return result.total_latency < best.total_latency ? result : best;
  }, null);

  const avgLatency = bestResult?.total_latency
    ? `${bestResult.total_latency.toFixed(1)} ms`
    : "-- ms";
  const avgLoss = links.length
    ? links.reduce((total, link) => total + link.packet_loss_rate, 0) / links.length
    : 0;
  const congestionEvents = links.filter((link) => link.utilization >= 0.7).length;

  return [
    { label: "Best Latency", value: avgLatency, Icon: Gauge },
    { label: "Packet Delivery", value: formatPercent(Math.max(0, 1 - avgLoss)), Icon: RadioTower },
    { label: "Congested Links", value: String(congestionEvents), Icon: Activity },
    { label: "Best Algorithm", value: bestResult?.algorithm ?? "--", Icon: Route }
  ];
}

function MetricsPanel({ networkState, comparison }) {
  const metrics = getMetrics(networkState, comparison);

  return (
    <section className="flex flex-col gap-3">
      {metrics.map(({ label, value, Icon }) => (
        <article key={label} className="rounded border border-app-border bg-app-panel p-4">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs uppercase text-app-muted">{label}</span>
            <Icon className="h-4 w-4 text-app-accent" aria-hidden="true" />
          </div>
          <p className="mt-3 truncate text-2xl font-semibold text-app-text">{value}</p>
        </article>
      ))}
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
const MemoizedMetricsPanel = memo(MetricsPanel, (prev, next) =>
    prev.networkState?.step_count === next.networkState?.step_count &&
    prev.comparison === next.comparison);

export default MemoizedMetricsPanel;
