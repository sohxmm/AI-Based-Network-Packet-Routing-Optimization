// TODO: implement
import { Activity, Gauge, RadioTower, Route } from "lucide-react";

const metrics = [
  { label: "Avg Latency", value: "-- ms", Icon: Gauge },
  { label: "Packet Delivery", value: "--%", Icon: RadioTower },
  { label: "Congestion Events", value: "--", Icon: Activity },
  { label: "Algorithm", value: "Dijkstra", Icon: Route }
];

function MetricsPanel() {
  return (
    <section className="grid gap-3 sm:grid-cols-2">
      {metrics.map(({ label, value, Icon }) => (
        <article key={label} className="rounded border border-slate-800 bg-slate-900 p-4">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs uppercase tracking-wide text-slate-500">{label}</span>
            <Icon className="h-4 w-4 text-cyan-300" aria-hidden="true" />
          </div>
          <p className="mt-3 text-2xl font-semibold">{value}</p>
        </article>
      ))}
    </section>
  );
}

export default MetricsPanel;
