import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { algorithmColor } from "../utils/colorScales.js";
import { utilizationColor } from "../utils/colorScales.js";

/**
 * TopologyGraph
 *
 * Three-phase D3 rendering strategy, so the graph never "dances" on a tick:
 *
 * Phase 1 (structure) — fires ONCE when the node set changes.
 *   Builds the force simulation, draws every <line> and <g>, and stores live
 *   refs to the D3 selections.
 *
 * Phase 2 (appearance) — fires on EVERY networkState update.
 *   Updates only stroke, stroke-width and fill via selection.attr(). No DOM
 *   teardown, no simulation restart, CSS transitions for smoothness.
 *
 * Phase 3 (overlays) — fires when the highlighted paths change.
 *   Draws one polyline per algorithm, offset perpendicular to each segment so
 *   that where several algorithms share an edge, all of them stay visible.
 *   This is the view that makes the project's argument legible: same source,
 *   same destination, same instant, different choices.
 */
function TopologyGraph({
  networkState,
  highlightedPaths = [],
  isDark = true,
  height = 460,
}) {
  const svgRef = useRef(null);

  const linkBaseRef = useRef(null);
  const linkAnimRef = useRef(null);
  const nodeCircleRef = useRef(null);
  const overlayRef = useRef(null);
  const positionsRef = useRef(new Map());
  const topologyKeyRef = useRef("");

  // ─── Phase 1: build the simulation (topology changes only) ──────────────
  useEffect(() => {
    if (!networkState?.nodes?.length || !svgRef.current) return;

    const key = [...networkState.nodes].sort().join(",");
    if (key === topologyKeyRef.current) return;
    topologyKeyRef.current = key;

    const width = 760;
    const canvasHeight = 400;

    const nodeCount = networkState.nodes.length;
    const simNodes = networkState.nodes.map((id, i) => ({
      id,
      x: width / 2 + width * 0.38 * Math.cos((2 * Math.PI * i) / nodeCount),
      y: canvasHeight / 2 + canvasHeight * 0.38 * Math.sin((2 * Math.PI * i) / nodeCount),
    }));

    const simLinks = networkState.links.map((l) => ({
      source: l.source,
      target: l.target,
      edgeKey: [l.source, l.target].sort().join("-"),
    }));

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${canvasHeight}`);

    // An arrowhead so an overlaid route reads as directional.
    svg
      .append("defs")
      .append("marker")
      .attr("id", "route-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9)
      .attr("refY", 5)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", "context-stroke");

    // Denser graphs need weaker repulsion and shorter links, or 100 nodes
    // explode off the canvas.
    const linkDistance = nodeCount > 60 ? 42 : 92;
    const charge = nodeCount > 60 ? -90 : -260;

    const simulation = d3
      .forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).id((n) => n.id).distance(linkDistance))
      .force("charge", d3.forceManyBody().strength(charge))
      .force("center", d3.forceCenter(width / 2, canvasHeight / 2))
      .force("collision", d3.forceCollide(nodeCount > 60 ? 12 : 26))
      .alphaDecay(0.05);

    const linkBaseSel = svg
      .append("g")
      .attr("class", "links-base")
      .selectAll("line")
      .data(simLinks, (d) => d.edgeKey)
      .join("line")
      .attr("stroke-width", 2)
      .attr("stroke", "#334155")
      .attr("stroke-opacity", 0.6)
      .style("transition", "stroke 0.6s ease, stroke-width 0.4s ease");

    linkBaseSel.append("title").text((d) => d.edgeKey);

    const linkAnimSel = svg
      .append("g")
      .attr("class", "links-anim")
      .selectAll("line")
      .data(simLinks, (d) => d.edgeKey)
      .join("line")
      .attr("stroke-dasharray", "4, 8")
      .attr("class", "animate-packet-flow")
      .style("pointer-events", "none")
      .style("opacity", 0);

    // Overlays sit above the links but below the nodes.
    const overlayG = svg.append("g").attr("class", "path-overlays");
    overlayRef.current = overlayG;

    const nodeRadius = nodeCount > 60 ? 7 : 15;
    const nodeSel = svg
      .append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(simNodes, (d) => d.id)
      .join("g")
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.15).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event) => {
            // fx/fy are left set on purpose: a node the user positioned should
            // stay where they put it rather than drifting back on the next tick.
            if (!event.active) simulation.alphaTarget(0);
          })
      );

    const circlesSel = nodeSel
      .append("circle")
      .attr("r", nodeRadius)
      .attr("fill", "var(--color-accent)")
      .attr("stroke-width", 1.5)
      .style("transition", "fill 0.6s ease, stroke 0.6s ease");

    if (nodeCount <= 60) {
      nodeSel
        .append("text")
        .attr("dy", 4)
        .attr("text-anchor", "middle")
        .attr("font-size", 10)
        .attr("font-weight", 700)
        .attr("fill", "var(--color-accent-text)")
        .attr("pointer-events", "none")
        .text((d) => d.id);
    }
    nodeSel.append("title").text((d) => d.id);

    simulation.on("tick", () => {
      linkBaseSel
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      linkAnimSel
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);

      const positions = new Map();
      simNodes.forEach((n) => positions.set(n.id, { x: n.x, y: n.y }));
      positionsRef.current = positions;
    });

    // Redraw overlays once the layout has settled, so the first render of a
    // comparison is not drawn against pre-simulation coordinates.
    simulation.on("end", () => drawOverlays());

    linkBaseRef.current = linkBaseSel;
    linkAnimRef.current = linkAnimSel;
    nodeCircleRef.current = circlesSel;

    return () => simulation.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [networkState?.nodes?.join(",")]);

  // ─── Phase 2: appearance only (every tick) ──────────────────────────────
  useEffect(() => {
    if (!networkState?.links || !linkBaseRef.current || !nodeCircleRef.current) return;

    const utilMap = new Map(
      networkState.links.map((l) => [
        [l.source, l.target].sort().join("-"),
        l.utilization,
      ])
    );

    const failedColor = isDark ? "rgba(239, 68, 68, 0.45)" : "rgba(239, 68, 68, 0.65)";

    linkBaseRef.current
      .attr("stroke", (d) =>
        utilMap.has(d.edgeKey) ? utilizationColor(utilMap.get(d.edgeKey)) : failedColor
      )
      .attr("stroke-width", (d) =>
        utilMap.has(d.edgeKey) ? 1.5 + utilMap.get(d.edgeKey) * 3 : 2
      )
      .attr("stroke-dasharray", (d) => (utilMap.has(d.edgeKey) ? "none" : "4, 4"));

    linkBaseRef.current.select("title").text((d) => {
      if (!utilMap.has(d.edgeKey)) return `${d.edgeKey}: FAILED`;
      return `${d.edgeKey}: ${Math.round(utilMap.get(d.edgeKey) * 100)}% utilized`;
    });

    if (linkAnimRef.current) {
      linkAnimRef.current
        .attr("stroke", (d) => {
          if (!utilMap.has(d.edgeKey)) return "none";
          const util = utilMap.get(d.edgeKey) ?? 0;
          return util > 0.8 ? "#ef4444" : isDark ? "#e2e8f0" : "#475569";
        })
        .attr("stroke-width", (d) => 1.5 + (utilMap.get(d.edgeKey) ?? 0) * 2)
        .style("opacity", (d) => {
          if (!utilMap.has(d.edgeKey)) return 0;
          const util = utilMap.get(d.edgeKey) ?? 0;
          return util > 0 ? Math.min(0.9, util + 0.25) : 0;
        })
        .style("animation-duration", (d) => {
          const util = utilMap.get(d.edgeKey) ?? 0;
          return `${Math.max(0.5, 2 - util * 1.5)}s`;
        });
    }

    const congested = new Set();
    networkState.links.forEach((l) => {
      if (l.utilization > 0.8) {
        congested.add(l.source);
        congested.add(l.target);
      }
    });

    nodeCircleRef.current
      .attr("fill", (d) => (congested.has(d.id) ? "#ef4444" : "var(--color-accent)"))
      .attr("stroke", "var(--color-border)")
      .attr("class", (d) => (congested.has(d.id) ? "animate-congestion" : ""));
  }, [networkState, isDark]);

  // ─── Phase 3: path overlays ─────────────────────────────────────────────
  function drawOverlays() {
    const overlay = overlayRef.current;
    if (!overlay) return;

    overlay.selectAll("*").remove();
    const positions = positionsRef.current;
    if (!positions.size || !highlightedPaths.length) return;

    const OFFSET_STEP = 3.5;
    const count = highlightedPaths.length;

    highlightedPaths.forEach((entry, index) => {
      const path = entry?.path ?? [];
      if (path.length < 2) return;

      // Fan the routes apart so overlapping segments stay individually visible.
      const offset = (index - (count - 1) / 2) * OFFSET_STEP;
      const points = [];

      for (let i = 0; i < path.length - 1; i += 1) {
        const a = positions.get(path[i]);
        const b = positions.get(path[i + 1]);
        if (!a || !b) return;

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const length = Math.hypot(dx, dy) || 1;
        // Unit normal to this segment, used to translate the drawn line.
        const nx = -dy / length;
        const ny = dx / length;

        if (i === 0) points.push([a.x + nx * offset, a.y + ny * offset]);
        points.push([b.x + nx * offset, b.y + ny * offset]);
      }

      overlay
        .append("path")
        .attr("d", d3.line()(points))
        .attr("fill", "none")
        .attr("stroke", algorithmColor(entry.algorithm))
        .attr("stroke-width", 4)
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round")
        .attr("stroke-opacity", 0.9)
        // A dashed route is a fallback route: the model did not run.
        .attr("stroke-dasharray", entry.is_fallback ? "6,4" : null)
        .attr("marker-end", "url(#route-arrow)")
        .append("title")
        .text(
          `${entry.algorithm}: ${path.join(" → ")}` +
            (entry.total_latency != null
              ? ` (${entry.total_latency.toFixed(1)} ms)`
              : "")
        );
    });
  }

  useEffect(() => {
    drawOverlays();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightedPaths, networkState?.step_count]);

  return (
    <section className="flex h-full flex-col rounded border border-app-border bg-app-panel p-4">
      <div className="flex shrink-0 items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-app-text">Topology</h2>
        <span className="text-xs text-app-muted">
          Drag to reposition · thickness and colour show live utilization
        </span>
      </div>

      {networkState?.nodes?.length ? (
        <svg
          ref={svgRef}
          role="img"
          aria-label="Network topology graph"
          className="mt-3 w-full flex-1 rounded border border-app-border bg-app-input-bg"
          style={{ minHeight: height }}
        />
      ) : (
        <div
          className="mt-3 flex flex-1 items-center justify-center rounded border border-dashed border-app-border text-sm text-app-muted"
          style={{ minHeight: height }}
        >
          Waiting for network state…
        </div>
      )}

      {highlightedPaths.length > 0 && (
        <ul className="mt-2 flex shrink-0 flex-wrap gap-x-4 gap-y-1 text-xs">
          {highlightedPaths.map((entry) => (
            <li key={entry.algorithm} className="flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-5 rounded"
                style={{
                  backgroundColor: algorithmColor(entry.algorithm),
                  opacity: entry.is_fallback ? 0.6 : 1,
                }}
                aria-hidden="true"
              />
              <span className="text-app-muted">
                {entry.algorithm}
                {entry.is_fallback && " (fallback)"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default TopologyGraph;
