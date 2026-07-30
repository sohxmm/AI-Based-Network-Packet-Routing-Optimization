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
  const linkBaseRef = useRef(null);
  const linkAnimRef = useRef(null);
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

    // ── Draw links (Base) ───────────────────────────────────────────────
    const linkBaseG = svg.append("g").attr("class", "links-base");
    const linkBaseSel = linkBaseG
      .selectAll("line")
      .data(simLinks, (d) => d.edgeKey)
      .join("line")
      .attr("stroke-width", 2)
      .attr("stroke", "#334155")
      .attr("stroke-opacity", 0.6)
      .style("transition", "stroke 0.6s ease, stroke-width 0.4s ease");

    linkBaseSel.append("title").text((d) => d.edgeKey);

    // ── Draw links (Animated Packets) ───────────────────────────────────
    const linkAnimG = svg.append("g").attr("class", "links-anim");
    const linkAnimSel = linkAnimG
      .selectAll("line")
      .data(simLinks, (d) => d.edgeKey)
      .join("line")
      .attr("stroke-dasharray", "4, 8")
      .attr("class", "animate-packet-flow")
      .style("pointer-events", "none")
      .style("opacity", 0);

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
      .attr("fill", "var(--color-accent)")
      .attr("stroke-width", 1.5)
      .style("transition", "fill 0.6s ease, stroke 0.6s ease");

    nodeSel
      .append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("font-weight", 700)
      .attr("fill", "var(--color-accent-text)")
      .attr("pointer-events", "none")
      .text((d) => d.id);

    // ── Tick handler: only update geometry (x/y), not colors ───────────
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
    });

    // Store selections in refs so Phase 2 can reach them
    linkBaseRef.current = linkBaseSel;
    linkAnimRef.current = linkAnimSel;
    nodeCircleRef.current = circlesSel;

    return () => simulation.stop();
    // Only depends on the node-id list — NOT on the full networkState
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [networkState?.nodes?.join(",")]);

  // ─── Phase 2: update colors only (every WebSocket tick) ──────────────
  useEffect(() => {
    if (!networkState?.links || !linkBaseRef.current || !nodeCircleRef.current || !linkAnimRef.current) return;

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

    // Update base link stroke + width
    linkBaseRef.current
      .attr("stroke", (d) => {
        if (!utilMap.has(d.edgeKey)) return isDark ? "rgba(239, 68, 68, 0.4)" : "rgba(239, 68, 68, 0.6)";
        if (highlightedEdges.has(d.edgeKey)) return isDark ? "rgba(103, 232, 249, 0.4)" : "rgba(6, 182, 212, 0.4)";
        const util = utilMap.get(d.edgeKey);
        return utilizationColor(util);
      })
      .attr("stroke-width", (d) => {
        if (!utilMap.has(d.edgeKey)) return 2;
        if (highlightedEdges.has(d.edgeKey)) return 5;
        const util = utilMap.get(d.edgeKey);
        return 1.5 + util * 3;
      })
      .attr("stroke-dasharray", (d) => {
        if (!utilMap.has(d.edgeKey)) return "4, 4";
        return "none";
      });

    // Update animated packet layer
    linkAnimRef.current
      .attr("stroke", (d) => {
        if (!utilMap.has(d.edgeKey)) return "none";
        if (highlightedEdges.has(d.edgeKey)) return isDark ? "#67e8f9" : "#06b6d4";
        const util = utilMap.get(d.edgeKey) ?? 0;
        return util > 0.8 ? "#ef4444" : isDark ? "#e2e8f0" : "#475569";
      })
      .attr("stroke-width", (d) => {
        if (highlightedEdges.has(d.edgeKey)) return 3;
        const util = utilMap.get(d.edgeKey) ?? 0;
        return 1.5 + util * 2;
      })
      .style("opacity", (d) => {
        if (!utilMap.has(d.edgeKey)) return 0;
        if (highlightedEdges.has(d.edgeKey)) return 1;
        const util = utilMap.get(d.edgeKey) ?? 0;
        return util > 0 ? Math.min(1, util + 0.3) : 0;
      })
      .style("animation-duration", (d) => {
        const util = utilMap.get(d.edgeKey) ?? 0;
        return `${Math.max(0.5, 2 - util * 1.5)}s`;
      });

    // Update link tooltip text
    linkBaseRef.current.select("title").text((d) => {
      if (!utilMap.has(d.edgeKey)) return `${d.edgeKey}: FAILED`;
      const util = utilMap.get(d.edgeKey);
      return `${d.edgeKey}: ${Math.round(util * 100)}% utilized`;
    });

    // Update node fill and congestion pulsing
    nodeCircleRef.current
      .attr("fill", (d) => {
        const isCongested = networkState.links.some(
          (l) => l.utilization > 0.8 && (l.source === d.id || l.target === d.id)
        );
        return isCongested ? "#ef4444" : "var(--color-accent)";
      })
      .attr("class", (d) => {
        const isCongested = networkState.links.some(
          (l) => l.utilization > 0.8 && (l.source === d.id || l.target === d.id)
        );
        return isCongested ? "animate-congestion" : "";
      });
  }, [networkState, highlightedPath, isDark]);

  // ─── Phase 3: Update theme colors ──────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    // Unhighlighted links might need updating but they're mostly colored by utilizationColor which we can assume handles it.
    // We just update the node strokes here.
    svg.selectAll(".nodes circle").attr("stroke", isDark ? "var(--color-border)" : "var(--color-border)");
  }, [isDark, networkState?.nodes]);

  return (
    <section className="h-full rounded border border-app-border bg-app-panel p-4 flex flex-col">
      <div className="flex items-center justify-between gap-3 shrink-0">
        <h2 className="text-sm font-semibold text-app-text">Topology Graph</h2>
        <span className="text-xs text-app-muted">Drag nodes to reposition • animations live</span>
      </div>
      {networkState?.nodes?.length ? (
        <svg
          ref={svgRef}
          className="mt-4 flex-1 w-full rounded border border-app-border bg-app-input-bg min-h-[460px]"
        />
      ) : (
        <div className="mt-4 flex flex-1 items-center justify-center rounded border border-dashed border-app-border text-sm text-app-muted min-h-[460px]">
          Waiting for network state
        </div>
      )}
    </section>
  );
}

export default TopologyGraph;
