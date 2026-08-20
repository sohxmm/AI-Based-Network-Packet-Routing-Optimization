import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SEQUENTIAL_BLUE, algorithmLabel, deltaColor } from "../../utils/colorScales.js";

/**
 * Two charts, because there are two questions and they need different encodings.
 *
 * 1. **How slow is each algorithm?** That is a magnitude comparison across
 *    categories, so: horizontal bars sorted best-first, one sequential hue,
 *    direct value labels, and 95% confidence intervals drawn as error bars
 *    (the intervals matter — several algorithms overlap).
 *
 * 2. **Better or worse than the baseline?** That is polarity, so: a diverging
 *    scale around zero, blue for better and orange for worse, with a neutral
 *    zero line. Deliberately not red/green, which is the worst possible pair
 *    for colour-vision deficiency, and every bar is labelled with its signed
 *    value so the reading never depends on colour.
 */
function LatencyChart({ rows, isDark }) {
  if (!rows?.length) return null;

  const sorted = [...rows].sort(
    (a, b) => (a.mean_latency ?? Infinity) - (b.mean_latency ?? Infinity)
  );

  const latencyData = sorted.map((row) => ({
    name: algorithmLabel(row.algorithm),
    algorithm: row.algorithm,
    latency: row.mean_latency,
    // Recharts ErrorBar takes [below, above] offsets from the value.
    error: row.ci
      ? [
          Math.max(0, (row.mean_latency ?? 0) - (row.ci.ci95_low ?? row.mean_latency)),
          Math.max(0, (row.ci.ci95_high ?? row.mean_latency) - (row.mean_latency ?? 0)),
        ]
      : undefined,
  }));

  const deltaData = sorted
    .filter((row) => row.algorithm !== "dijkstra" && row.pct_diff != null)
    .map((row) => ({
      name: algorithmLabel(row.algorithm),
      algorithm: row.algorithm,
      delta: row.pct_diff,
    }));

  const axisColor = isDark ? "#c3c2b7" : "#52514e";
  const gridColor = isDark ? "#383835" : "#e6e5e1";
  const barFill = isDark ? SEQUENTIAL_BLUE[3] : SEQUENTIAL_BLUE[4];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <figure className="rounded border border-app-border bg-app-input-bg p-3">
        <figcaption className="text-xs font-semibold text-app-text">
          Mean path latency
          <span className="ml-1 font-normal text-app-muted">
            (ms, lower is better, bars show 95% CI across runs)
          </span>
        </figcaption>
        <ResponsiveContainer width="100%" height={Math.max(180, latencyData.length * 34)}>
          <BarChart
            data={latencyData}
            layout="vertical"
            margin={{ top: 8, right: 56, left: 8, bottom: 8 }}
          >
            <CartesianGrid stroke={gridColor} horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: axisColor, fontSize: 11 }}
              stroke={gridColor}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={130}
              tick={{ fill: axisColor, fontSize: 11 }}
              stroke={gridColor}
            />
            <Tooltip
              cursor={{ fill: gridColor, fillOpacity: 0.3 }}
              contentStyle={{
                background: "var(--color-panel)",
                border: "1px solid var(--color-border)",
                borderRadius: 4,
                fontSize: 12,
                color: "var(--color-text-main)",
              }}
              formatter={(value) => [`${Number(value).toFixed(1)} ms`, "mean latency"]}
            />
            <Bar dataKey="latency" fill={barFill} radius={[0, 4, 4, 0]} barSize={16}>
              <LabelList
                dataKey="latency"
                position="right"
                formatter={(value) => Number(value).toFixed(1)}
                style={{ fill: axisColor, fontSize: 11 }}
              />
              <ErrorBar dataKey="error" width={4} strokeWidth={1.5} stroke={axisColor} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </figure>

      {deltaData.length > 0 && (
        <figure className="rounded border border-app-border bg-app-input-bg p-3">
          <figcaption className="text-xs font-semibold text-app-text">
            Difference vs Dijkstra
            <span className="ml-1 font-normal text-app-muted">
              (%, negative is better than the baseline)
            </span>
          </figcaption>
          <ResponsiveContainer width="100%" height={Math.max(180, deltaData.length * 34)}>
            <BarChart
              data={deltaData}
              layout="vertical"
              margin={{ top: 8, right: 56, left: 8, bottom: 8 }}
            >
              <CartesianGrid stroke={gridColor} horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: axisColor, fontSize: 11 }}
                stroke={gridColor}
                tickFormatter={(value) => `${value > 0 ? "+" : ""}${value}%`}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={130}
                tick={{ fill: axisColor, fontSize: 11 }}
                stroke={gridColor}
              />
              <Tooltip
                cursor={{ fill: gridColor, fillOpacity: 0.3 }}
                contentStyle={{
                  background: "var(--color-panel)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 4,
                  fontSize: 12,
                  color: "var(--color-text-main)",
                }}
                formatter={(value) => [
                  `${value > 0 ? "+" : ""}${Number(value).toFixed(1)}%`,
                  "vs Dijkstra",
                ]}
              />
              <ReferenceLine x={0} stroke={axisColor} strokeWidth={1.5} />
              <Bar dataKey="delta" radius={4} barSize={16}>
                {deltaData.map((entry) => (
                  <Cell key={entry.algorithm} fill={deltaColor(entry.delta, isDark)} />
                ))}
                <LabelList
                  dataKey="delta"
                  position="right"
                  formatter={(value) =>
                    `${value > 0 ? "+" : ""}${Number(value).toFixed(1)}%`
                  }
                  style={{ fill: axisColor, fontSize: 11 }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </figure>
      )}
    </div>
  );
}

export default LatencyChart;
