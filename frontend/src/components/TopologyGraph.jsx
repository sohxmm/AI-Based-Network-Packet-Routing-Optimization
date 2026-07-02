import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { utilizationColor } from "../utils/colorScales.js";

/**
 * TopologyGraph
 *
 * Two-phase D3 rendering strategy to prevent the graph from "dancing"
 * on every WebSocket tick:
 *
 * Phase 1 (structureEffect) — fires ONCE when the node list changes.
 *   Builds the force simulation, draws all <line> and <g> elements,
 *   and stores live refs to the D3 link/node selections.
 *
 * Phase 2 (colorEffect) — fires on EVERY networkState update.
 *   Only updates stroke color, stroke-width, and node fill via
 *   selection.attr() — no DOM teardown, no simulation restart.
 *   Uses a short CSS transition so color changes animate smoothly.
 */
function TopologyGraph({ networkState, highlightedPath = [], isDark = true }) {
  const svgRef = useRef(null);

  // Stable refs to D3 selections so Phase 2 can reach them without
  // triggering Phase 1.
  const linkSelRef = useRef(null);
  const nodeCircleRef = useRef(null);

  // Stable ref to the frozen topology (node ids + static link list).
  // We only rebuild the simulation when the node-id set actually changes.
  const topologyKeyRef = useRef("");

  // ─── Phase 1: build simulation (topology changes only) ───────────────
  useEffect(() => {
    if (!networkState?.nodes?.length || !svgRef.current) return;

    // Compute a stable key from sorted node ids
    const key = [...networkState.nodes].sort().join(",");
    if (key === topologyKeyRef.current) return; // topology unchanged — skip
    topologyKeyRef.current = key;

    const width = 760;
    const height = 360;

    // Build node objects with stable positions seeded in a ring so the
    // initial layout is already reasonable and settles quickly.
    const nodeCount = networkState.nodes.length;
    const simNodes = networkState.nodes.map((id, i) => ({
      id,
      // Pre-position on a circle so the sim barely moves on startup
      x: width / 2 + (width * 0.38) * Math.cos((2 * Math.PI * i) / nodeCount),
      y: height / 2 + (height * 0.38) * Math.sin((2 * Math.PI * i) / nodeCount),
    }));

    // Build link objects from the FIRST snapshot — structure won't change
    const simLinks = networkState.links.map((l) => ({
      source: l.source,
      target: l.target,
      // Store the edge key for highlight lookup
      edgeKey: [l.source, l.target].sort().join("-"),
    }));

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const simulation = d3
      .forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).id((n) => n.id).distance(92))
      .force("charge", d3.forceManyBody().strength(-260))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(26))
      // Cool quickly — we don't want it bouncing during the demo
      .alphaDecay(0.05);

    // ── Draw links ──────────────────────────────────────────────────────
    const linkG = svg.append("g").attr("class", "links");
    const linkSel = linkG
      .selectAll("line")
      .data(simLinks, (d) => d.edgeKey)
      .join("line")
      .attr("stroke-width", 2)
      .attr("stroke", "#334155")      // neutral default until Phase 2 fires
      .attr("stroke-opacity", 0.86)
      // Smooth color transitions — only color, not position
      .style("transition", "stroke 0.6s ease, stroke-width 0.4s ease");

    linkSel.append("title").text((d) => d.edgeKey);

    // ── Draw nodes ──────────────────────────────────────────────────────
    const nodeG = svg.append("g").attr("class", "nodes");
    const nodeSel = nodeG
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
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            // Pin the node in place after drag — stops it from drifting
            // back when new ticks fire.
            // Leave fx/fy set so the node stays where the user put it.
          })
      );

    const circlesSel = nodeSel
      .append("circle")
      .attr("r", 15)
      .attr("fill", "#38bdf8")
      .attr("stroke-width", 1.5)
      .style("transition", "fill 0.6s ease, stroke 0.6s ease");

    nodeSel
      .append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("font-weight", 700)
      .attr("fill", "#0f172a")
      .attr("pointer-events", "none")
      .text((d) => d.id);

    // ── Tick handler: only update geometry (x/y), not colors ───────────
    simulation.on("tick", () => {
      linkSel
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    // Store selections in refs so Phase 2 can reach them
    linkSelRef.current = linkSel;
    nodeCircleRef.current = circlesSel;

    return () => simulation.stop();
    // Only depends on the node-id list — NOT on the full networkState
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [networkState?.nodes?.join(",")]);

  // ─── Phase 2: update colors only (every WebSocket tick) ──────────────
  useEffect(() => {
    if (!networkState?.links || !linkSelRef.current || !nodeCircleRef.current) return;

    // Build a fast lookup: "R1-R2" → utilization
    const utilMap = new Map(
      networkState.links.map((l) => [
        [l.source, l.target].sort().join("-"),
        l.utilization,
      ])
    );

    // Build a fast lookup: "R1-R2" → is highlighted
    const highlightedEdges = new Set(
      highlightedPath
        .slice(0, -1)
        .map((node, i) => [node, highlightedPath[i + 1]].sort().join("-"))
    );

    // Update link stroke + width (CSS transition handles the animation)
    linkSelRef.current
      .attr("stroke", (d) => {
        if (highlightedEdges.has(d.edgeKey)) return isDark ? "#67e8f9" : "#06b6d4";
        const util = utilMap.get(d.edgeKey) ?? 0;
        return utilizationColor(util);
      })
      .attr("stroke-width", (d) => {
        if (highlightedEdges.has(d.edgeKey)) return 5;
        const util = utilMap.get(d.edgeKey) ?? 0;
        return 1.5 + util * 4;
      });

    // Update link tooltip text
    linkSelRef.current.select("title").text((d) => {
      const util = utilMap.get(d.edgeKey) ?? 0;
      return `${d.edgeKey}: ${Math.round(util * 100)}% utilized`;
    });

    // Update node fill (congested = any adjacent link > 0.8)
    nodeCircleRef.current.attr("fill", (d) => {
      const isCongested = networkState.links.some(
        (l) =>
          l.utilization > 0.8 && (l.source === d.id || l.target === d.id)
      );
      return isCongested ? "#ef4444" : "#38bdf8";
    });
  }, [networkState, highlightedPath, isDark]);

  // ─── Phase 3: Update theme colors ──────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    // Unhighlighted links might need updating but they're mostly colored by utilizationColor which we can assume handles it.
    // We just update the node strokes here.
    svg.selectAll(".nodes circle").attr("stroke", isDark ? "#e2e8f0" : "#cbd5e1");
  }, [isDark, networkState?.nodes]);

  return (
    <section className="min-h-[420px] rounded border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900/90">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">Topology Graph</h2>
        <span className="text-xs text-slate-500 dark:text-slate-400">Drag nodes to reposition • colors update live</span>
      </div>
      {networkState?.nodes?.length ? (
        <svg
          ref={svgRef}
          className="mt-4 h-[360px] w-full rounded border border-slate-300 bg-slate-50 dark:border-slate-800 dark:bg-slate-950"
        />
      ) : (
        <div className="mt-4 flex h-[360px] items-center justify-center rounded border border-dashed border-slate-300 text-sm text-slate-500 dark:border-slate-700">
          Waiting for network state
        </div>
      )}
    </section>
  );
}

export default TopologyGraph;
