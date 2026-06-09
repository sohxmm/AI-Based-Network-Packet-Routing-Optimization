// TODO: implement
import * as d3 from "d3";

export const utilizationColor = d3
  .scaleLinear()
  .domain([0, 0.4, 0.7, 1])
  .range(["#22c55e", "#eab308", "#f97316", "#ef4444"]);
