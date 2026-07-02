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
        <article key={label} className="rounded border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900/90">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs uppercase text-slate-500">{label}</span>
            <Icon className="h-4 w-4 text-cyan-600 dark:text-cyan-300" aria-hidden="true" />
          </div>
          <p className="mt-3 truncate text-2xl font-semibold text-slate-900 dark:text-slate-100">{value}</p>
        </article>
      ))}
    </section>
  );
}

export default MetricsPanel;
