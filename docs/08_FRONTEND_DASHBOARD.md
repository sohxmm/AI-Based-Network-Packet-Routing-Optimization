# Frontend Dashboard

React 18 + Vite + Tailwind, with D3 for the topology and Recharts for the
comparison charts. Source lives in `web/`.

```bash
cd web
npm install
npm run dev        # http://localhost:5173
npm run test       # Vitest + Testing Library
npm run lint       # ESLint, --max-warnings 0
npm run build      # production bundle
```

---

## 1. What the dashboard is for

It is not a demo surface. Its job is to make three things visible that are
otherwise buried in JSON:

1. **Whether the AI is actually running.** Every learned router can fall back to
   a heuristic, and before this revision that was invisible.
2. **Why one algorithm chose a different path from another.** A latency number
   tells you *that* they differ; the divergence view tells you *where*.
3. **What the benchmark's own guardrails are saying about its results.**

---

## 2. Layout

Four tabs (`web/src/App.jsx`):

| Tab | Contents |
|---|---|
| **Live network** | Topology graph, congestion heatmap, metrics, simulator controls, network source panel, failover panel |
| **Path divergence** | Side-by-side path comparison with per-hop cost breakdown |
| **Benchmark** | Committed scenario results, statistics, warnings |
| **Lab** | Experiment builder — configure and run a sandbox benchmark |

Four themes: GitHub Dark, Nord, Dracula, Solarized Light.

Four, not ten. The previous ten-theme engine worked and demonstrated real CSS
architecture, but ten themes cost roughly what fixing the model-loading bug cost,
and the project needed the second one more. Four keep the engine and the
light/dark handling honest without reading as padding. `isDark` drives the chart
palettes, which are **validated separately for each mode** rather than flipped
automatically — an automatic inversion produces colours that pass in neither.

---

## 3. Components

### 3.1 Honesty surfaces

These exist to make the system's self-reported problems unmissable.

| Component | What it shows |
|---|---|
| `ModelStatusBanner` | Which model artifacts are present and which actually loaded. Two separate facts, because a file that exists but failed to load is the interesting case. |
| `WarningsCallout` | The `warnings` block from a results file, rendered **above** the results table, not below it. |
| `GuardrailBadge` | Marks a decision the safety guardrails would have rejected. |

A user should never have to notice a caveat themselves. If a row was produced by
a fallback heuristic, the dashboard says so next to the number.

### 3.2 Network views

| Component | Notes |
|---|---|
| `TopologyGraph` | D3 force-directed. Two-phase render: the simulation is created once and only *updated* on state change, so nodes do not jump every tick. Node positions persist across ticks. |
| `CongestionHeatmap` | Recharts bar chart of per-link utilisation. |
| `MetricsPanel` / `LeftPanel` / `RightPanel` | Summary cards, queue sizes, topology stats. |
| `ControlPanel` | Step, reset, inject failure, restore link. |
| `NetworkSourcePanel` | Switch between simulator, trace replay and live probing. Renders the live-mode limitation inline rather than in a tooltip. |
| `FailoverPanel` | Watched flows and reroute events as they happen. |

### 3.3 Path analysis

| Component | Notes |
|---|---|
| `PathDivergenceView` | Two algorithms' paths side by side, with the divergence point highlighted. |
| `PathCostBreakdown` | Per-hop `base_latency`, `utilization` and `cost`, so the total is auditable rather than asserted. The numbers multiply out to the reported total. |

### 3.4 Benchmark

`web/src/components/benchmark/`:

| Component | Notes |
|---|---|
| `ScenarioSelector` | Pick a committed scenario. |
| `WarningsCallout` | Guardrail findings, first. |
| `AlgorithmMetricsTable` | Latency, p95, success, fallback, QoS satisfaction, diversity, match rate. |
| `LatencyChart` | Horizontal bars with error bars from the bootstrap CI. |
| `StatisticalSummary` | Cliff's delta, magnitude, CI and Wilcoxon p per algorithm. |

The statistics are shown next to the latency, never instead of it. A bar chart
without a confidence interval invites the reader to believe a 0.4 ms difference
is real.

---

## 4. Hooks

All server state comes through three hooks. No component fetches directly.

| Hook | Responsibility |
|---|---|
| `useNetworkStream` | WebSocket connection with reconnect and backoff; exposes `networkState`, `failoverEvents`, `isConnected` |
| `useRouteRequest` | REST calls for routing and comparison |
| `useModelHealth` | Polls `/health/models` for the status banner |

Errors go through `utils/apiError.js`, which turns a FastAPI `detail` into
something a user can act on instead of `[object Object]`.

---

## 5. Colour

`web/src/utils/colorScales.js` is the single source of chart colour, and it is
organised by the **job** each colour does rather than by preference.

**Categorical (identity).** Eight hues, one per algorithm, assigned in fixed
order and never cycled. Colour follows the algorithm, not its rank, so filtering
the comparison never repaints the survivors — a filter that recolours the
remaining series makes two charts of the same data unreadable side by side.

The set is validated: adjacent pairs clear the colour-vision-deficiency
separation floor and the lightness/chroma bands in both light and dark mode.
Three light-mode slots sit below 3:1 contrast on a pale surface, which is exactly
why **every chart also carries direct value labels and a table view**. Identity is
never conveyed by colour alone.

**Sequential (magnitude).** Link utilisation uses one ramp, green to red through
amber, at fixed breakpoints (0, 0.4, 0.7, 1.0). One hue family, light to dark —
never a rainbow.

**Status.** Reserved for good/warning/serious/critical and never reused as
"series 4". Status always ships with an icon and a label, not colour alone.

There are no dual-axis charts anywhere in the dashboard. Two measures of
different scale get two charts.

---

## 6. Configuration

`web/src/config.js` reads `VITE_API_URL` at build time and defaults to
same-origin, so the nginx image works with no configuration and the dev server
works via the Vite proxy.

The previous version hardcoded `http://localhost:8000` in
`useRouteRequest.js` and hardcoded port 8000 in the WebSocket URL, which meant
any non-localhost deployment required a source edit.

---

## 7. Testing

```bash
npm run test           # 18 tests
```

Vitest + Testing Library, with `web/src/test/setup.js` providing jest-dom
matchers and a `ResizeObserver` stub (jsdom has none, and Recharts needs one).

`vite.config.js` sets `esbuild: { jsx: "automatic" }`. Without it every test
failed with "React is not defined", because the components use the automatic JSX
runtime and Vitest's transform did not.

---

## 8. Linting

```bash
npm run lint           # exits 0 at --max-warnings 0
```

ESLint with a `.eslintrc.cjs` legacy config. The previous setup mixed flat-config
packages with a legacy config file, so `npm run lint` did not run at all — it
errored out before checking anything, which reads as "no lint errors" in a
terminal you are not watching closely.
