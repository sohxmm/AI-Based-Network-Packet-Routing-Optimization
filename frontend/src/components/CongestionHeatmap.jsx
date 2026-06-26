import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

function CongestionHeatmap({ networkState }) {
  const data = (networkState?.links ?? [])
    .map((link) => ({
      name: `${link.source}-${link.target}`,
      utilization: Number((link.utilization * 100).toFixed(1)),
      queue: link.queue_size
    }))
    .sort((left, right) => right.utilization - left.utilization)
    .slice(0, 12);

  return (
    <section className="rounded border border-slate-800 bg-slate-900/90 p-4">
      <h2 className="text-sm font-semibold text-slate-200">Congestion Heatmap</h2>
      <div className="mt-4 h-72">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 18, right: 18 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" unit="%" />
              <YAxis dataKey="name" type="category" stroke="#94a3b8" width={72} />
              <Tooltip
                cursor={{ fill: "rgba(148, 163, 184, 0.08)" }}
                contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                labelStyle={{ color: "#e2e8f0" }}
              />
              <Bar dataKey="utilization" name="Utilization" fill="#22d3ee" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded border border-dashed border-slate-700 text-sm text-slate-500">
            Waiting for network state
          </div>
        )}
      </div>
    </section>
  );
}

export default CongestionHeatmap;
