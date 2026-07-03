import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

function CongestionHeatmap({ networkState, isDark = true }) {
  const data = (networkState?.links ?? [])
    .map((link) => ({
      name: `${link.source}-${link.target}`,
      utilization: Number((link.utilization * 100).toFixed(1)),
      queue: link.queue_size
    }))
    .sort((left, right) => right.utilization - left.utilization)
    .slice(0, 12);

  // We can use generic neutral colors or inherit from css variables if available.
  const axisColor = isDark ? "#94a3b8" : "#475569";
  const gridColor = isDark ? "#1e293b" : "#e2e8f0";
  const tooltipBg = "var(--color-panel)";
  const tooltipBorder = "var(--color-border)";
  const tooltipColor = "var(--color-text-main)";

  return (
    <section className="rounded border border-app-border bg-app-panel p-4">
      <h2 className="text-sm font-semibold text-app-text">Congestion Heatmap</h2>
      <div className="mt-4 h-72">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 18, right: 18 }}>
              <CartesianGrid stroke={gridColor} strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} stroke={axisColor} unit="%" />
              <YAxis dataKey="name" type="category" stroke={axisColor} width={72} />
              <Tooltip
                cursor={{ fill: isDark ? "rgba(148, 163, 184, 0.08)" : "rgba(15, 23, 42, 0.05)" }}
                contentStyle={{ background: tooltipBg, border: `1px solid ${tooltipBorder}` }}
                labelStyle={{ color: tooltipColor }}
              />
              <Bar dataKey="utilization" name="Utilization" fill="var(--color-accent)" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded border border-dashed border-app-border text-sm text-app-muted">
            Waiting for network state
          </div>
        )}
      </div>
    </section>
  );
}

export default CongestionHeatmap;
