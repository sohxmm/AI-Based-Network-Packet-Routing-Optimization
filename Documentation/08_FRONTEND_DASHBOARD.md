# Frontend Dashboard

The dashboard is a React 18 single-page application built with Vite, styled with Tailwind CSS 3, and visualized using D3.js and Recharts.

---

## 1. Technology Stack

| Technology       | Version | Purpose                                 |
|-----------------|---------|------------------------------------------|
| React           | 18.3    | UI framework                             |
| Vite            | 8.x     | Build tool and dev server                |
| Tailwind CSS    | 3.4     | Utility-first CSS framework              |
| D3.js           | 7.9     | Force-directed topology graph            |
| Recharts        | 2.12    | Bar charts and data visualization        |
| Lucide React    | 0.468   | Icon library                             |

---

## 2. Dashboard Layout

The dashboard uses a responsive 3-column CSS Grid layout:

```
┌─────────────────────────────────────────────────────────────────┐
│  Header: Title, Step Count, Connection Status, Theme Selector   │
├──────────┬──────────────────────────────┬───────────────────────┤
│          │                              │                       │
│  LEFT    │         CENTER               │      RIGHT            │
│  COLUMN  │         COLUMN               │      COLUMN           │
│  (250px) │      (flexible)              │      (320px)          │
│          │                              │                       │
│ Metrics  │  TopologyGraph               │  RouteComparison      │
│  Panel   │  (D3 force graph)            │  (form + table +      │
│          │                              │   bar chart)           │
│ Left     │  CongestionHeatmap           │                       │
│  Panel   │  (Recharts bar)              │  ControlPanel         │
│ (queues) │                              │  (step/reset/failure) │
│          │                              │                       │
│          │                              │  RightPanel           │
│          │                              │  (topology stats)     │
│          │                              │                       │
└──────────┴──────────────────────────────┴───────────────────────┘
```

---

## 3. Components

### 3.1 TopologyGraph (`components/TopologyGraph.jsx`)

**D3 force-directed network graph** showing all routers and links.

Features:
- **Two-phase rendering**: Force simulation built once; only colors update per tick
- **Live color updates**: Link color reflects utilization (green → yellow → red)
- **Path highlighting**: Best route highlighted in cyan when comparison results exist
- **Drag-to-pin**: Users can drag nodes; they stay pinned after release
- **Congestion indicators**: Node circles turn red when adjacent links exceed 80% utilization
- **CSS transitions**: Smooth 0.6s color transitions for link and node updates

### 3.2 CongestionHeatmap (`components/CongestionHeatmap.jsx`)

**Horizontal bar chart** (Recharts) showing the top 12 most utilized links.

- Sorted by utilization (highest first)
- Updates in real-time as the simulator runs
- Uses theme-aware axis colors and tooltip styling
- X-axis shows percentage (0–100%)

### 3.3 RouteComparison (`components/RouteComparison.jsx`)

**Algorithm comparison panel** with three sub-sections:

1. **Route Form**: Source/destination dropdowns + "Compare All" button
2. **Results Table**: Algorithm name, latency, and path (click to expand long paths)
3. **Latency Bar Chart**: Visual comparison of algorithm latencies

The best algorithm row is highlighted with an accent background.

### 3.4 ControlPanel (`components/ControlPanel.jsx`)

**Simulator control buttons**:

| Button          | Action                                        |
|-----------------|-----------------------------------------------|
| Step +1         | Advance simulator by 1 step                   |
| Step +10        | Advance simulator by 10 steps (sequential)    |
| Reset           | Reset simulator to initial state (step 0)     |
| Link dropdown   | Select a link for failure/restore operations   |
| Inject Failure  | Remove the selected link from the topology     |
| Restore Link    | Restore a previously failed link               |

### 3.5 MetricsPanel (`components/MetricsPanel.jsx`)

**Four metric cards** showing:

| Metric           | Source                                      |
|------------------|---------------------------------------------|
| Best Latency     | Lowest latency from the latest comparison   |
| Packet Delivery  | Estimated from average packet loss rate     |
| Congested Links  | Count of links with utilization ≥ 0.7       |
| Best Algorithm   | Algorithm name with lowest latency          |

### 3.6 LeftPanel (`components/LeftPanel.jsx`)

**Queue size leaderboard** showing the top 5 links by queue size. Provides a quick view of which links are building up queued packets.

### 3.7 RightPanel (`components/RightPanel.jsx`)

**Global topology statistics**:
- Total routers (nodes)
- Active links
- Average link capacity
- Count of congested links (utilization ≥ 0.7)

---

## 4. Custom Hooks

### 4.1 `useNetworkStream` (`hooks/useNetworkStream.js`)

Manages the WebSocket connection to `ws://localhost:8000/ws/stream`.

**Returns:**

| Property          | Type          | Description                           |
|-------------------|---------------|---------------------------------------|
| `networkState`    | `object|null` | Latest network state from WebSocket   |
| `lastRoutingEvent`| `object|null` | Most recent routing event             |
| `isConnected`     | `boolean`     | Whether WebSocket is connected        |
| `error`           | `string|null` | Error message if connection failed    |

**Features:**
- Automatic reconnection with exponential backoff (1s → 30s)
- Clean teardown on component unmount
- Protocol auto-detection (`ws:` / `wss:` based on page protocol)

### 4.2 `useRouteRequest` (`hooks/useRouteRequest.js`)

Manages REST API calls to the backend.

**Returns:**

| Property             | Type       | Description                           |
|----------------------|------------|---------------------------------------|
| `compareRoutes`      | `function` | Send route comparison request         |
| `postSimulatorAction`| `function` | Send simulator control action         |
| `isLoading`          | `boolean`  | Whether a request is in-flight        |
| `error`              | `string|null` | Error message from last request    |

---

## 5. Theming Engine

The dashboard supports **10 color themes** via CSS custom properties defined in `index.css`.

### Available Themes

| Theme               | Type   | Description                            |
|---------------------|--------|----------------------------------------|
| `solarized-light`   | Light  | Classic Solarized light palette        |
| `solarized-dark`    | Dark   | Classic Solarized dark palette         |
| `dracula`           | Dark   | Popular Dracula color scheme (default) |
| `monokai`           | Dark   | Monokai Pro-inspired                   |
| `nord`              | Dark   | Arctic, blue-tinted palette            |
| `tokyo-night`       | Dark   | VSCode Tokyo Night inspired            |
| `one-dark`          | Dark   | Atom One Dark inspired                 |
| `gruvbox`           | Dark   | Gruvbox dark theme                     |
| `synthwave`         | Dark   | Retro neon synthwave                   |
| `github-dark`       | Dark   | GitHub dark mode inspired              |

### How Theming Works

1. Each theme defines CSS variables under a `.theme-<name>` class
2. `App.jsx` swaps the class on `<html>` when the user selects a theme
3. Components use Tailwind utility classes that reference CSS variables:
   - `bg-app-panel` → `var(--color-panel)`
   - `text-app-text` → `var(--color-text-main)`
   - `border-app-border` → `var(--color-border)`
   - `bg-app-accent` → `var(--color-accent)`

### CSS Variable Map

```css
--color-bg:           /* Page background */
--color-panel:        /* Card/panel background */
--color-input-bg:     /* Input and table header background */
--color-border:       /* Border color */
--color-text-main:    /* Primary text */
--color-text-muted:   /* Secondary/muted text */
--color-accent:       /* Accent buttons, highlights */
--color-accent-text:  /* Text on accent backgrounds */
```

---

## 6. Build & Development

### Development Server

```bash
cd frontend
npm run dev
# → http://localhost:5173
```

### Production Build

```bash
cd frontend
npm run build
# Output → frontend/dist/
```

### Linting

```bash
npm run lint
```

### Vite Proxy Configuration

In development, the Vite dev server proxies `/api` requests to the backend:

```javascript
// vite.config.js
server: {
  proxy: {
    "/api": {
      target: "http://localhost:8000",
      changeOrigin: true
    }
  }
}
```

> **Note**: The frontend currently connects directly to `http://localhost:8000` for REST calls and `ws://localhost:8000/ws/stream` for WebSocket. The proxy is configured but not actively used by the hooks.
