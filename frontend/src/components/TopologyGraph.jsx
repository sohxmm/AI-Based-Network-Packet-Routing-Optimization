import { useEffect, useRef } from "react";

import * as d3 from "d3";

import { utilizationColor } from "../utils/colorScales.js";

function TopologyGraph({ networkState, highlightedPath = [] }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!networkState?.nodes?.length || !svgRef.current) {
      return;
    }

    const width = 760;
    const height = 360;
    const links = networkState.links.map((link) => ({ ...link }));
    const nodes = networkState.nodes.map((id) => ({
      id,
      congested: networkState.links.some(
        (link) => link.utilization > 0.8 && (link.source === id || link.target === id)
      )
    }));
    const highlightedEdges = new Set(
      highlightedPath.slice(0, -1).map((node, index) => [node, highlightedPath[index + 1]].sort().join("-"))
    );

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const simulation = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((node) => node.id).distance(92))
      .force("charge", d3.forceManyBody().strength(-260))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(24));

    const link = svg
      .append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", (item) => {
        const key = [item.source.id ?? item.source, item.target.id ?? item.target].sort().join("-");
        return highlightedEdges.has(key) ? 5 : 1.5 + item.utilization * 4;
      })
      .attr("stroke", (item) => {
        const key = [item.source.id ?? item.source, item.target.id ?? item.target].sort().join("-");
        return highlightedEdges.has(key) ? "#67e8f9" : utilizationColor(item.utilization);
      })
      .attr("stroke-opacity", 0.86);

    link.append("title").text(
      (item) =>
        `${item.source.id ?? item.source}-${item.target.id ?? item.target}: ` +
        `${Math.round(item.utilization * 100)}% utilized, ${item.base_latency} ms`
    );

    const node = svg
      .append("g")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .call(
        d3
          .drag()
          .on("start", (event, item) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            item.fx = item.x;
            item.fy = item.y;
          })
          .on("drag", (event, item) => {
            item.fx = event.x;
            item.fy = event.y;
          })
          .on("end", (event, item) => {
            if (!event.active) simulation.alphaTarget(0);
            item.fx = null;
            item.fy = null;
          })
      );

    node
      .append("circle")
      .attr("r", 15)
      .attr("fill", (item) => (item.congested ? "#ef4444" : "#38bdf8"))
      .attr("stroke", "#e2e8f0")
      .attr("stroke-width", 1.5);

    node
      .append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("font-weight", 700)
      .attr("fill", "#0f172a")
      .text((item) => item.id);

    simulation.on("tick", () => {
      link
        .attr("x1", (item) => item.source.x)
        .attr("y1", (item) => item.source.y)
        .attr("x2", (item) => item.target.x)
        .attr("y2", (item) => item.target.y);

      node.attr("transform", (item) => `translate(${item.x},${item.y})`);
    });

    return () => simulation.stop();
  }, [networkState, highlightedPath]);

  return (
    <section className="min-h-[420px] rounded border border-slate-800 bg-slate-900/90 p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-200">Topology Graph</h2>
        <span className="text-xs text-slate-500">Drag nodes to inspect routes</span>
      </div>
      {networkState?.nodes?.length ? (
        <svg ref={svgRef} className="mt-4 h-[360px] w-full rounded border border-slate-800 bg-slate-950" />
      ) : (
        <div className="mt-4 flex h-[360px] items-center justify-center rounded border border-dashed border-slate-700 text-sm text-slate-500">
          Waiting for network state
        </div>
      )}
    </section>
  );
}

export default TopologyGraph;
