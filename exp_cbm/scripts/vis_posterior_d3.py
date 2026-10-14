"""
Generate a standalone D3 HTML visualization of BC-LLM's posterior concepts.

Loads a TrainingHistory pickle, embeds each concept's text, runs PCA, clusters
in embedding space (HDBSCAN), and writes an interactive HTML file where:
  - each unique concept is a dot anchored at its (PC1, PC2) position
  - dot size = posterior_freq (inclusion probability), single dark color
  - low-frequency 'explored but rarely kept' concepts are small gray markers
  - per-cluster weighted 2D KDE drawn as nested filled HPD contours (95/80/50%)
  - one draggable cluster label per cluster, auto-named 'Cluster A/B/...'
    with posterior mass appended. Click to rename.
  - users can also draw manual annotation ovals and download a
    `posterior_concepts_positions.json` capturing label positions + ovals.
    If that JSON sits next to the HTML at build time, those are restored.

One HTML file per invocation; see exp_cbm/sconscript for how this is
wired into the pipeline.
"""
import argparse
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401 — needed to register contour backend
import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BC_LLM_ROOT = os.path.join(REPO_ROOT, "bc-llm-head-to-head")
if BC_LLM_ROOT not in sys.path:
    sys.path.insert(0, BC_LLM_ROOT)

from src.training_history import TrainingHistory  # noqa: E402


_D3_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>BC-LLM posterior concepts</title>
<style>
  :root {
    --page-text: #1a1a1a;
    --muted-text: #4a4a4a;
    --dot-color: #1f3b60;
    --cluster-text: #142239;
    --axis-color: #808080;
  }
  body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
         margin: 0; padding: 20px 24px; background: #ffffff;
         color: var(--page-text); }
  .layout { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  .plot-col { flex: 1 1 1400px; min-width: 0; }
  .side-col { flex: 0 0 360px; max-height: 920px; overflow-y: auto;
              border: 1px solid #ddd; border-radius: 4px; padding: 12px 14px;
              background: #fcfcfc; font-size: 13px;
              position: sticky; top: 10px; }
  .side-col h3 { font-size: 14px; font-weight: 600; margin: 0 0 6px 0; }
  .side-col .side-sub { font-size: 11px; color: var(--muted-text); margin-bottom: 8px; }
  .side-col .search { width: 100%; box-sizing: border-box; padding: 4px 6px;
                      font-size: 12px; margin-bottom: 8px;
                      border: 1px solid #c0c0c0; border-radius: 3px;
                      font-family: inherit; }
  .side-col ul { list-style: none; padding: 0; margin: 0; }
  .side-col li { padding: 4px 0; font-size: 12px; line-height: 1.35;
                 display: flex; gap: 8px; align-items: flex-start; }
  .side-col li input { margin-top: 3px; flex: 0 0 auto; }
  .side-col li .dot-mark { flex: 0 0 10px; height: 10px; border-radius: 50%;
                           margin-top: 4px; }
  .side-col li label { cursor: pointer; flex: 1 1 auto; }
  .side-col li.is-visible label { font-weight: 600; }
  .side-col li.hidden-by-search { display: none; }
  .side-col .cluster-chip { display: inline-block; padding: 1px 6px;
                            margin-right: 4px; border-radius: 2px;
                            background: #e4ebf3; color: #2b3a4a;
                            font-size: 10px; font-weight: 600; }
  h1   { font-size: 16px; font-weight: 600; margin: 0 0 4px 0; }
  .sub { font-size: 12px; color: var(--muted-text); margin-bottom: 12px; }
  svg  { background: #ffffff; display: block; }
  .tooltip {
    position: absolute; pointer-events: none;
    background: rgba(0,0,0,0.85); color: #fff;
    padding: 6px 9px; border-radius: 4px;
    font-size: 13px; line-height: 1.35; max-width: 360px;
    opacity: 0; transition: opacity 0.1s;
  }
  .toolbar { margin-bottom: 10px; }
  .legend-bar { margin-top: 10px; padding: 10px 14px;
                border: 1px solid #e2e2e2; border-radius: 4px;
                background: #fafafa; font-size: 12px; color: #333;
                display: flex; flex-wrap: wrap; gap: 24px; align-items: center; }
  .legend-bar .legend-group { display: inline-flex; align-items: center; gap: 8px; }
  .legend-bar .legend-title { font-weight: 600; color: #555; }
  .legend-bar .legend-item { display: inline-flex; align-items: center; gap: 4px; }
  .legend-bar .legend-label { color: #333; }
  button { font-size: 12px; padding: 5px 10px; margin-right: 6px; cursor: pointer;
           background: #f4f4f4; border: 1px solid #c0c0c0; border-radius: 3px;
           font-family: inherit; }
  button:hover { background: #e8e8e8; }
  button.active { background: #ffe08a; border-color: #d4a017; }
  /* interactive-only chrome; stripped from PDF export clone */
  .dot      { cursor: pointer; }
  .concept-label { cursor: move; user-select: none; }
  .concept-label.hidden { opacity: 0; pointer-events: none; }
  .concept-leader.hidden { opacity: 0; }
  .cluster-label { cursor: move; user-select: none; }
  .cluster-label.hidden { opacity: 0; pointer-events: none; }
  .cluster-label.selected-for-merge,
  .merged-label.selected-for-merge {
    paint-order: stroke; stroke: #d4a017; stroke-width: 1.5px;
  }
  .merged-label { cursor: move; user-select: none; }
  .merged-label.hidden { opacity: 0; pointer-events: none; }
  .merge-leader { stroke: #8ab4d8; stroke-width: 3px; opacity: 0.75;
                  stroke-dasharray: 6 4;
                  pointer-events: none; }
  .merge-leader.hidden { opacity: 0; }
  .anno-oval { cursor: move; }
  .anno-handle { cursor: nwse-resize; }
  .anno-delete-bg { cursor: pointer; opacity: 0; transition: opacity 0.1s; }
  .anno-delete-x  { pointer-events: none; user-select: none;
                    opacity: 0; transition: opacity 0.1s; }
  .anno.show-delete .anno-delete-bg,
  .anno.show-delete .anno-delete-x { opacity: 1; }
  .anno-label { cursor: move; user-select: none; }
  #chart.anno-mode { cursor: crosshair; }
</style>
</head>
<body>
<h1 id="title">BC-LLM posterior concepts</h1>
<div class="sub" id="subtitle"></div>
<div class="toolbar">
  <button onclick="resetPositions()">reset positions</button>
  <button onclick="exportPositions()">download positions JSON</button>
  <button onclick="exportPdf()">download PDF</button>
  <button id="annoBtn" onclick="toggleAnnoMode()">draw annotation oval</button>
  <button onclick="clearOvals()">clear ovals</button>
  <button id="mergeBtn" onclick="mergeSelectedClusters()"
          style="display:none;">merge selected clusters</button>
  <span id="mergeHint" style="font-size:12px;color:#666;margin-left:4px;"></span>
  <span class="k-control">
    <label for="kSlider" style="font-size:12px;color:#333;font-weight:600;margin-left:12px;">K:</label>
    <input type="range" id="kSlider" style="vertical-align:middle;width:140px;" />
    <span id="kValue" style="font-size:12px;color:#333;min-width:1.5em;display:inline-block;">4</span>
  </span>
  <span id="annoHint" style="font-size:12px;color:#666;margin-left:8px;"></span>
</div>
<div class="layout">
  <div class="plot-col">
    <svg id="chart" width="1400" height="900" font-family="Helvetica"></svg>
    <div id="legend" class="legend-bar"></div>
  </div>
  <div class="side-col">
    <h3>Clusters</h3>
    <div class="side-sub">Uncheck to hide a cluster's label on the plot.
      Click the name to rename.</div>
    <ul id="clusterList"></ul>
    <h3 style="margin-top:14px;">Concepts</h3>
    <div class="side-sub">Check a concept to show its text label on the plot.
      Click any concept to also rename its cluster.</div>
    <input type="text" class="search" id="conceptSearch" placeholder="filter concepts..." />
    <ul id="conceptList"></ul>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.3/dist/svg2pdf.umd.min.js"></script>
<script>
const payload = __PAYLOAD__;

// ---- Saved-positions hydration -----------------------------------------
// Saved positions may be from:
//   - legacy flat array of {concept, labelX, labelY}
//   - K-less schema {concept_labels, cluster_labels: [...], ovals}
//   - K-aware schema {concept_labels, cluster_labels: {<K>: [...]}, ovals}
const savedRaw = payload.saved_positions;
const savedLabelByConcept = new Map();
let savedClusterByCidByK = null;  // {k_int: Map(cluster_id -> {label, labelX, labelY})}
let savedOvalList = [];
// {k_int: [{id, cluster_ids:[...], label, labelX?, labelY?, visible?}]}
// Scoped per-K because cluster_ids are only meaningful inside a specific
// clustering; a K=6 merge entry doesn't apply at K=5.
const mergesByK = {};
if (savedRaw) {
  const list = Array.isArray(savedRaw) ? savedRaw : (savedRaw.concept_labels || savedRaw.labels || []);
  list.forEach(s => savedLabelByConcept.set(s.concept, s));
  if (!Array.isArray(savedRaw)) {
    const cl = savedRaw.cluster_labels;
    savedClusterByCidByK = {};
    if (Array.isArray(cl)) {
      // legacy K-less: treat as per-default-K renames
      const m = new Map();
      cl.forEach(c => m.set(c.cluster_id, c));
      savedClusterByCidByK[payload.default_k] = m;
    } else if (cl && typeof cl === "object") {
      Object.keys(cl).forEach(kStr => {
        const m = new Map();
        (cl[kStr] || []).forEach(c => m.set(c.cluster_id, c));
        savedClusterByCidByK[parseInt(kStr, 10)] = m;
      });
    }
    if (savedRaw.ovals) savedOvalList = savedRaw.ovals;
    else if (savedRaw.circles) {
      savedOvalList = savedRaw.circles.map(c => ({
        cx: c.cx, cy: c.cy, rx: c.r, ry: c.r, label: c.label || "",
      }));
    }
    const cm = savedRaw.cluster_merges;
    if (cm && typeof cm === "object" && !Array.isArray(cm)) {
      Object.keys(cm).forEach(kStr => {
        mergesByK[parseInt(kStr, 10)] = (cm[kStr] || []).map(m => ({
          id: m.id || ("m_" + Math.random().toString(36).slice(2, 10)),
          cluster_ids: (m.cluster_ids || []).map(x => parseInt(x, 10)),
          label: m.label || "Merged cluster",
          labelX: m.labelX, labelY: m.labelY,
          visible: m.visible !== undefined ? !!m.visible : true,
        }));
      });
    }
  }
}

document.getElementById("title").textContent =
  `Posterior concept map — ${payload.axis}`;
document.getElementById("subtitle").textContent =
  `${payload.concepts.length} concepts · ` +
  `K ∈ [${payload.k_min}, ${payload.k_max}] (default ${payload.default_k}) · ` +
  `PC1 ${(payload.pc1_var*100).toFixed(1)}%, PC2 ${(payload.pc2_var*100).toFixed(1)}% variance explained · ` +
  `explored-only threshold = ${payload.explored_threshold.toFixed(2)}`;

const svg = d3.select("#chart");
const width = +svg.attr("width");
const height = +svg.attr("height");
const margin = {top: 24, right: 40, bottom: 56, left: 68};
const innerW = width - margin.left - margin.right;
const innerH = height - margin.top - margin.bottom;
const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

// Equal-aspect scales: keep per-unit scale consistent across axes.
const xDomSpan = payload.xlim[1] - payload.xlim[0];
const yDomSpan = payload.ylim[1] - payload.ylim[0];
const unitPx = Math.min(innerW / xDomSpan, innerH / yDomSpan);
const effW = unitPx * xDomSpan;
const effH = unitPx * yDomSpan;
// Center the equal-aspect plot inside the available inner box.
const xOffset = (innerW - effW) / 2;
const yOffset = (innerH - effH) / 2;
const xScale = d3.scaleLinear().domain(payload.xlim).range([xOffset, xOffset + effW]);
const yScale = d3.scaleLinear().domain(payload.ylim).range([yOffset + effH, yOffset]);

// ---- Axes --------------------------------------------------------------
const axisColor = "#808080";
g.append("line")
  .attr("x1", xOffset).attr("x2", xOffset + effW)
  .attr("y1", yOffset + effH).attr("y2", yOffset + effH)
  .attr("stroke", axisColor).attr("stroke-width", 0.75);
g.append("line")
  .attr("x1", xOffset).attr("x2", xOffset)
  .attr("y1", yOffset).attr("y2", yOffset + effH)
  .attr("stroke", axisColor).attr("stroke-width", 0.75);

function drawTicks(axis, scale, baseline, tickLen, orient) {
  const ticks = scale.ticks(5);
  axis.selectAll("line.tick").data(ticks).join("line")
    .attr("class", "tick")
    .attr("stroke", axisColor).attr("stroke-width", 0.6);
  if (orient === "x") {
    axis.selectAll("line.tick")
      .attr("x1", d => scale(d)).attr("x2", d => scale(d))
      .attr("y1", baseline).attr("y2", baseline + tickLen);
    axis.selectAll("text.tick-label").data(ticks).join("text")
      .attr("class", "tick-label")
      .attr("x", d => scale(d)).attr("y", baseline + tickLen + 12)
      .attr("fill", axisColor).attr("font-size", 11)
      .attr("font-family", "Helvetica")
      .attr("text-anchor", "middle")
      .text(d => d);
  } else {
    axis.selectAll("line.tick")
      .attr("x1", baseline - tickLen).attr("x2", baseline)
      .attr("y1", d => scale(d)).attr("y2", d => scale(d));
    axis.selectAll("text.tick-label").data(ticks).join("text")
      .attr("class", "tick-label")
      .attr("x", baseline - tickLen - 4).attr("y", d => scale(d) + 4)
      .attr("fill", axisColor).attr("font-size", 11)
      .attr("font-family", "Helvetica")
      .attr("text-anchor", "end")
      .text(d => d);
  }
}
const xAxisG = g.append("g");
const yAxisG = g.append("g");
drawTicks(xAxisG, xScale, yOffset + effH, 4, "x");
drawTicks(yAxisG, yScale, xOffset, 4, "y");

g.append("text")
  .attr("x", xOffset + effW / 2)
  .attr("y", yOffset + effH + 42)
  .attr("fill", "#555").attr("font-size", 14).attr("font-weight", 600)
  .attr("font-family", "Helvetica").attr("text-anchor", "middle")
  .text(`PC1 (${(payload.pc1_var*100).toFixed(1)}% var.)`);
g.append("text")
  .attr("transform", `translate(${xOffset - 48},${yOffset + effH/2}) rotate(-90)`)
  .attr("fill", "#555").attr("font-size", 14).attr("font-weight", 600)
  .attr("font-family", "Helvetica").attr("text-anchor", "middle")
  .text(`PC2 (${(payload.pc2_var*100).toFixed(1)}% var.)`);

// ---- Contours (filled, nested per cluster) -----------------------------
// Sequential blue, light at 95% HPD → dark at 50% HPD.
const HPD_FILLS = { 0.95: "#dde7f1", 0.80: "#b4ccea", 0.50: "#7fa6d4" };
const HPD_ALPHAS = { 0.95: 0.55, 0.80: 0.65, 0.50: 0.75 };
const LEVEL_ORDER = [0.95, 0.80, 0.50];

function polyToPath(poly) {
  if (!poly.length) return "";
  const parts = poly.map(([x, y], i) =>
    (i === 0 ? "M" : "L") + xScale(x) + "," + yScale(y));
  return parts.join(" ") + " Z";
}

const contourLayer = g.append("g").attr("class", "contour-layer");

function renderContours(contoursForK) {
  contourLayer.selectAll("*").remove();
  LEVEL_ORDER.forEach(lvl => {
    contoursForK.forEach(c => {
      const lvlEntry = c.levels.find(l => Math.abs(l.level - lvl) < 1e-6);
      if (!lvlEntry) return;
      lvlEntry.polygons.forEach(poly => {
        contourLayer.append("path")
          .attr("d", polyToPath(poly))
          .attr("fill", HPD_FILLS[lvl])
          .attr("fill-opacity", HPD_ALPHAS[lvl])
          .attr("stroke", "none");
      });
    });
  });
}

// ---- Dots --------------------------------------------------------------
const DOT_COLOR = "#1f3b60";
const EXPLORED_COLOR = "#9a9a9a";
// Uniform size for all dots — posterior inclusion probability is already
// encoded by the number of jittered replicates visible around each concept.
const UNIFORM_DOT_R = 4;
function dotRadius(d) {
  if (d.explored_only) return 2.5;
  return UNIFORM_DOT_R;
}

// Jittered replicates used by the GMM fit (debug/visual aid). Same size
// and color as the originating concept dot, but semi-transparent so the
// effective sample distribution EM saw is visible on the plot.
const jitterLayer = g.append("g").attr("class", "jitter-layer");
jitterLayer.selectAll("circle").data(payload.jitter_points || []).join("circle")
  .attr("class", "jitter-dot")
  .attr("cx", d => xScale(d[0]))
  .attr("cy", d => yScale(d[1]))
  .attr("r", d => {
    const origin = payload.concepts[d[2]];
    return origin ? dotRadius(origin) : 3;
  })
  .attr("fill", DOT_COLOR)
  .attr("fill-opacity", 0.5)
  .attr("stroke", "none");

const dotLayer = g.append("g").attr("class", "dot-layer");
const dots = dotLayer.selectAll("circle").data(payload.concepts).join("circle")
  .attr("class", "dot")
  .attr("cx", d => xScale(d.pc1))
  .attr("cy", d => yScale(d.pc2))
  .attr("r", dotRadius)
  .attr("fill", d => d.explored_only ? EXPLORED_COLOR : DOT_COLOR)
  .attr("fill-opacity", d => d.explored_only ? 0.55 : 1.0)
  .attr("stroke", "#ffffff")
  .attr("stroke-width", 0.6);

const tooltip = d3.select("#tooltip");
dots
  .on("mouseover", (event, d) => {
    const parts = [
      `<b>${d.concept}</b>`,
      `posterior_freq: ${d.posterior_freq.toFixed(3)}`,
      `mean coef: ${d.mean_coef.toFixed(3)}`,
      `cluster: ${d.cluster_id < 0 ? "(noise)" : clusterNameById(d.cluster_id)}`,
    ];
    tooltip.style("opacity", 1).html(parts.join("<br>"));
  })
  .on("mousemove", (event) => {
    tooltip.style("left", (event.pageX + 12) + "px")
           .style("top",  (event.pageY + 12) + "px");
  })
  .on("mouseout", () => tooltip.style("opacity", 0));

// ---- Per-concept labels (hidden by default) ----------------------------
// Each concept has an optional text label + leader line. Toggle via sidebar.
function initialLabelOffset(d) {
  // deterministic ring layout; angle derived from id so it's stable
  const theta = (d.id * 2.399963229);  // golden-angle-ish
  const r = 50;
  return [r * Math.cos(theta), -r * Math.sin(theta)];
}

const conceptItems = payload.concepts.map(c => {
  const saved = (savedLabelByConcept.get(c.concept)) || null;
  const [dx, dy] = initialLabelOffset(c);
  const ax = xScale(c.pc1), ay = yScale(c.pc2);
  return {
    ref: c,
    anchorX: ax, anchorY: ay,
    labelX: saved && saved.labelX != null ? saved.labelX : ax + dx,
    labelY: saved && saved.labelY != null ? saved.labelY : ay + dy,
    labelVisible: saved && saved.visible != null ? !!saved.visible : false,
    initialDx: dx, initialDy: dy,
  };
});

const leaderLayer = g.append("g").attr("class", "leader-layer");
const conceptLeaders = leaderLayer.selectAll("line")
  .data(conceptItems).join("line")
  .attr("class", "concept-leader")
  .attr("stroke", "#8a8a8a").attr("stroke-width", 0.75).attr("stroke-opacity", 0.5)
  .attr("x1", d => d.anchorX).attr("y1", d => d.anchorY)
  .attr("x2", d => d.labelX).attr("y2", d => d.labelY)
  .classed("hidden", d => !d.labelVisible);

const labelLayer = g.append("g").attr("class", "label-layer");
const conceptLabels = labelLayer.selectAll("text")
  .data(conceptItems).join("text")
  .attr("class", "concept-label")
  .attr("fill", "#2b2b2b")
  .attr("font-size", 15)
  .attr("font-weight", 700)
  .attr("font-family", "Helvetica")
  .attr("text-anchor", "middle")
  .attr("dominant-baseline", "middle")
  .attr("x", d => d.labelX).attr("y", d => d.labelY)
  .text(d => shortLabel(d.ref.concept))
  .classed("hidden", d => !d.labelVisible)
  .on("click", function(event, d) {
    if (event.defaultPrevented) return;
    setLabelVisible(d, false);
  })
  .call(d3.drag()
    .on("start", function(event) { event.sourceEvent.stopPropagation(); })
    .on("drag", function(event, d) {
      d.labelX = Math.max(0, Math.min(innerW, event.x));
      d.labelY = Math.max(0, Math.min(innerH, event.y));
      d3.select(this).attr("x", d.labelX).attr("y", d.labelY);
      conceptLeaders.filter(l => l === d)
        .attr("x2", d.labelX).attr("y2", d.labelY);
    }));

function setLabelVisible(item, visible) {
  item.labelVisible = visible;
  conceptLabels.filter(l => l === item).classed("hidden", !visible);
  conceptLeaders.filter(l => l === item).classed("hidden", !visible);
  const row = _rowByItem.get(item);
  if (row) {
    row.cb.checked = visible;
    row.li.classList.toggle("is-visible", visible);
  }
}

function shortLabel(s) {
  let t = s.trim();
  t = t.replace(/^Does Answer B\s+/i, "");
  t = t.replace(/^Is Answer B\s+/i, "");
  t = t.replace(/\s+than Answer A\??\s*$/i, "");
  if (t.length) t = t.charAt(0).toUpperCase() + t.slice(1);
  return t;
}

// ---- Cluster labels (bold, draggable) ----------------------------------
// Each cluster gets a label placed at its posterior-weighted centroid by
// default. User can rename via click or drag to reposition.
let clusterItems = [];
let currentK = payload.default_k;

function clusterText(ci) {
  return `${ci.label} · ${ci.ref.posterior_mass.toFixed(2)}`;
}
// Mirror the Python `_wrap_two_lines` helper so on-plot cluster labels match
// the matplotlib SVG export: break the name at the word boundary nearest the
// midpoint when it's long and multi-word; short/single-word names pass through.
function wrapTwoLines(text, maxChars = 22) {
  if (text.length <= maxChars || text.indexOf(" ") === -1) return [text];
  const words = text.split(" ");
  if (words.length === 1) return [text];
  const target = Math.floor(text.length / 2);
  let bestSplit = 1, bestDist = Infinity, acc = 0;
  for (let i = 0; i < words.length - 1; i++) {
    acc += words[i].length + 1;
    const dist = Math.abs(acc - target);
    if (dist < bestDist) { bestDist = dist; bestSplit = i + 1; }
  }
  return [words.slice(0, bestSplit).join(" "), words.slice(bestSplit).join(" ")];
}
// Lines rendered on the plot: wrapped name followed by the coefficient on its
// own last line (no `·` separator), matching the matplotlib export.
function clusterPlotLines(ci) {
  return wrapTwoLines(ci.label).concat([ci.ref.posterior_mass.toFixed(2)]);
}
// Render multi-line text by replacing children with <tspan> rows anchored at
// (x, y). Lines are stacked around the center (dominant-baseline="middle").
function setMultilineText(textSel, lines) {
  textSel.each(function(d) {
    const node = d3.select(this);
    const x = +node.attr("x");
    node.selectAll("tspan").remove();
    const n = lines.length;
    lines.forEach((line, i) => {
      node.append("tspan")
        .attr("x", x)
        .attr("dy", i === 0 ? `${-((n - 1) / 2) * 1.1}em` : "1.1em")
        .text(line);
    });
  });
}
function clusterNameById(id) {
  const it = clusterItems.find(x => x.ref.id === id);
  return it ? it.label : "?";
}

const clusterLayer = g.append("g").attr("class", "cluster-layer");
const mergeLeaderLayer = g.append("g").attr("class", "merge-leader-layer");
const mergedLabelLayer = g.append("g").attr("class", "merged-label-layer");

// ---- Merge selection + helpers -----------------------------------------
// Set of cluster_ids (within the currently displayed K) selected for the
// next merge operation. Shift-click a cluster label to toggle membership.
const selectedForMerge = new Set();

function clampToInner(x, y) {
  return [Math.max(0, Math.min(innerW, x)), Math.max(0, Math.min(innerH, y))];
}

function currentMerges() {
  // All merge entries for the active K; mutations happen on this array.
  if (!mergesByK[currentK]) mergesByK[currentK] = [];
  return mergesByK[currentK];
}

function clusterInSomeMerge(cid) {
  return currentMerges().some(m => m.cluster_ids.indexOf(cid) >= 0);
}

function conceptsForMerge(m) {
  // Union member_concepts of the clusters this merge covers, under current K.
  const clustersForK = payload.clusterings[String(currentK)].clusters;
  const byId = new Map(clustersForK.map(c => [c.id, c]));
  const out = [];
  m.cluster_ids.forEach(cid => {
    const c = byId.get(cid);
    if (c) out.push.apply(out, c.member_concepts);
  });
  return out;
}

function mergeMetrics(m) {
  // {posterior_mass, member_count, centroidX, centroidY} — mass-weighted
  // centroid in SVG pixel space.
  const clustersForK = payload.clusterings[String(currentK)].clusters;
  const byId = new Map(clustersForK.map(c => [c.id, c]));
  let mass = 0;
  let memberCount = 0;
  m.cluster_ids.forEach(cid => {
    const c = byId.get(cid);
    if (c) {
      mass += c.posterior_mass;
      memberCount += c.member_concepts.length;
    }
  });

  // Centroid: weight each concept by its posterior_freq.
  const freqByConcept = new Map(
    payload.concepts.map(c => [c.concept, {pc1: c.pc1, pc2: c.pc2, w: c.posterior_freq}]));
  let sw = 0, swx = 0, swy = 0;
  conceptsForMerge(m).forEach(name => {
    const r = freqByConcept.get(name);
    if (!r) return;
    sw += r.w; swx += r.w * r.pc1; swy += r.w * r.pc2;
  });
  const cx = sw > 0 ? xScale(swx / sw) : 0;
  const cy = sw > 0 ? yScale(swy / sw) - 20 : 0;
  return {posterior_mass: mass, member_count: memberCount,
          centroidX: cx, centroidY: cy};
}

function mergedPlotLines(m) {
  const metrics = mergeMetrics(m);
  return wrapTwoLines(m.label).concat([metrics.posterior_mass.toFixed(2)]);
}

function updateMergeControls() {
  const btn = document.getElementById("mergeBtn");
  const hint = document.getElementById("mergeHint");
  if (selectedForMerge.size >= 2) {
    btn.style.display = "";
    hint.textContent = `merge ${selectedForMerge.size} selected clusters`;
  } else if (selectedForMerge.size === 1) {
    btn.style.display = "none";
    hint.textContent = "shift-click another cluster label to merge";
  } else {
    btn.style.display = "none";
    hint.textContent = "";
  }
}

function toggleClusterSelection(cid) {
  if (selectedForMerge.has(cid)) selectedForMerge.delete(cid);
  else selectedForMerge.add(cid);
  clusterLayer.selectAll("text.cluster-label")
    .classed("selected-for-merge", d => selectedForMerge.has(d.ref.id));
  updateMergeControls();
}

function clearMergeSelection() {
  selectedForMerge.clear();
  clusterLayer.selectAll("text.cluster-label")
    .classed("selected-for-merge", false);
  updateMergeControls();
}

function mergeSelectedClusters() {
  const ids = Array.from(selectedForMerge);
  if (ids.length < 2) return;
  const name = prompt(
    `Name the merged cluster (${ids.length} clusters)`,
    "Merged cluster");
  if (name === null) return;
  const trimmed = name.trim() || "Merged cluster";
  currentMerges().push({
    id: "m_" + Math.random().toString(36).slice(2, 10),
    cluster_ids: ids.slice().sort((a, b) => a - b),
    label: trimmed,
    labelX: null, labelY: null,  // use centroid on first render
    visible: true,
  });
  // Auto-hide the component clusters so there's only one label on screen.
  ids.forEach(cid => {
    const item = clusterItems.find(ci => ci.ref.id === cid);
    if (item) setClusterVisible(item, false);
  });
  clearMergeSelection();
  renderMergedLabels();
}

function unmergeById(mergeId) {
  const arr = currentMerges();
  const idx = arr.findIndex(m => m.id === mergeId);
  if (idx < 0) return;
  const m = arr[idx];
  arr.splice(idx, 1);
  // Restore visibility of the clusters this merge was absorbing.
  m.cluster_ids.forEach(cid => {
    const item = clusterItems.find(ci => ci.ref.id === cid);
    if (item) setClusterVisible(item, true);
  });
  renderMergedLabels();
}

function buildClusterItems(k, previousItems) {
  // `savedClusterByCid` carries per-K renames from the saved JSON.
  const savedForK = (savedClusterByCidByK && savedClusterByCidByK[k]) || new Map();
  const clustersForK = payload.clusterings[String(k)].clusters;
  const items = clustersForK.map(c => {
    const saved = savedForK.get(c.id) || null;
    const defaultX = xScale(c.center_pc1);
    const defaultY = yScale(c.center_pc2) - 20;
    return {
      ref: c,
      labelX: saved && saved.labelX != null ? saved.labelX : defaultX,
      labelY: saved && saved.labelY != null ? saved.labelY : defaultY,
      label: (saved && saved.label) || c.default_label,
      visible: saved && saved.visible != null ? !!saved.visible : true,
      defaultX, defaultY,
    };
  });
  // Carry user renames forward from `previousItems` by nearest-centroid
  // match in PC space. We only overwrite labels that are still the default
  // ("Cluster X") so that per-K saved renames always win.
  if (previousItems && previousItems.length) {
    const taken = new Set();
    items.forEach(nu => {
      if (!/^Cluster [A-Z]+$/.test(nu.label)) return;  // already named
      let best = null, bestDist = Infinity;
      previousItems.forEach(prev => {
        if (taken.has(prev)) return;
        if (/^Cluster [A-Z]+$/.test(prev.label)) return;  // skip default-named
        const dx = nu.ref.center_pc1 - prev.ref.center_pc1;
        const dy = nu.ref.center_pc2 - prev.ref.center_pc2;
        const d = Math.hypot(dx, dy);
        if (d < bestDist) { bestDist = d; best = prev; }
      });
      if (best) {
        nu.label = best.label;
        nu.visible = best.visible;
        taken.add(best);
      }
    });
  }
  return items;
}

function renderClusterLabels() {
  const sel = clusterLayer.selectAll("text.cluster-label").data(clusterItems);
  sel.exit().remove();
  const enter = sel.enter().append("text")
    .attr("class", "cluster-label")
    .attr("fill", "#142239")
    .attr("font-size", 24)
    .attr("font-weight", 700)
    .attr("font-family", "Helvetica")
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle");
  enter.on("click", function(event, d) {
      if (event.defaultPrevented) return;
      if (event.shiftKey) {
        // Shift-click → toggle this cluster's membership in the pending merge.
        event.preventDefault();
        toggleClusterSelection(d.ref.id);
        return;
      }
      const name = prompt(
        `Rename cluster (members: ${d.ref.member_concepts.length} concepts):`,
        d.label);
      if (name !== null && name.trim()) {
        d.label = name.trim();
        setMultilineText(d3.select(this), clusterPlotLines(d));
        updateSidebarClusterNames();
        const row = _clusterRowByItem.get(d);
        if (row) {
          const lbl = row.li.querySelector("label");
          if (lbl) lbl.textContent = clusterText(d);
        }
      }
    })
    .call(d3.drag()
      .on("start", function(event) { event.sourceEvent.stopPropagation(); })
      .on("drag", function(event, d) {
        d.labelX = Math.max(0, Math.min(innerW, event.x));
        d.labelY = Math.max(0, Math.min(innerH, event.y));
        const node = d3.select(this)
          .attr("x", d.labelX).attr("y", d.labelY);
        node.selectAll("tspan").attr("x", d.labelX);
      }));
  const merged = enter.merge(sel);
  merged
    .attr("x", d => d.labelX).attr("y", d => d.labelY)
    .classed("hidden", d => !d.visible);
  merged.each(function(d) {
    setMultilineText(d3.select(this), clusterPlotLines(d));
  });
}

function renderMergeLeaders() {
  // One short line per (merge, child cluster), running from the merged-label
  // position to each child's PC-space centroid. Updated after any change
  // that moves the merged label or alters the merge membership.
  const clustersForK = payload.clusterings[String(currentK)].clusters;
  const centroidById = new Map(clustersForK.map(c => [c.id, c]));
  const data = [];
  currentMerges().forEach(m => {
    if (m.visible === false) return;
    m.cluster_ids.forEach(cid => {
      const c = centroidById.get(cid);
      if (!c) return;
      data.push({
        mergeId: m.id,
        cid,
        x1: m.labelX, y1: m.labelY,
        x2: xScale(c.center_pc1), y2: yScale(c.center_pc2),
      });
    });
  });
  const sel = mergeLeaderLayer.selectAll("line.merge-leader")
    .data(data, d => `${d.mergeId}:${d.cid}`);
  sel.exit().remove();
  sel.enter().append("line").attr("class", "merge-leader")
    .merge(sel)
    .attr("x1", d => d.x1).attr("y1", d => d.y1)
    .attr("x2", d => d.x2).attr("y2", d => d.y2);
}

function renderMergedLabels() {
  const data = currentMerges();
  const sel = mergedLabelLayer.selectAll("text.merged-label").data(data, d => d.id);
  sel.exit().remove();
  const enter = sel.enter().append("text")
    .attr("class", "merged-label")
    .attr("fill", "#142239")
    .attr("font-size", 24)
    .attr("font-weight", 700)
    .attr("font-family", "Helvetica")
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .on("click", function(event, d) {
      if (event.defaultPrevented) return;
      const name = prompt(
        `Rename merged cluster (${d.cluster_ids.length} clusters):`,
        d.label);
      if (name !== null && name.trim()) {
        d.label = name.trim();
        setMultilineText(d3.select(this), mergedPlotLines(d));
      }
    })
    .on("dblclick", function(event, d) {
      event.preventDefault();
      if (confirm(`Unmerge "${d.label}" back into ${d.cluster_ids.length} separate clusters?`)) {
        unmergeById(d.id);
      }
    })
    .call(d3.drag()
      .on("start", function(event) { event.sourceEvent.stopPropagation(); })
      .on("drag", function(event, d) {
        const [x, y] = clampToInner(event.x, event.y);
        d.labelX = x; d.labelY = y;
        const node = d3.select(this).attr("x", x).attr("y", y);
        node.selectAll("tspan").attr("x", x);
        // Keep leader lines tethered to the label mid-drag.
        mergeLeaderLayer.selectAll("line.merge-leader")
          .filter(ld => ld.mergeId === d.id)
          .attr("x1", x).attr("y1", y);
      }));
  const merged = enter.merge(sel);
  merged.each(function(d) {
    if (d.labelX == null || d.labelY == null) {
      const m = mergeMetrics(d);
      d.labelX = m.centroidX;
      d.labelY = m.centroidY;
    }
  });
  merged
    .attr("x", d => d.labelX).attr("y", d => d.labelY)
    .classed("hidden", d => d.visible === false);
  merged.each(function(d) {
    setMultilineText(d3.select(this), mergedPlotLines(d));
  });
  renderMergeLeaders();
}

function setClusterVisible(item, visible) {
  item.visible = visible;
  clusterLayer.selectAll("text.cluster-label")
    .filter(l => l === item)
    .classed("hidden", !visible);
  const row = _clusterRowByItem.get(item);
  if (row) {
    row.cb.checked = visible;
    row.li.classList.toggle("is-visible", visible);
  }
}

const _clusterRowByItem = new Map();
function renderClusterSidebar() {
  const host = document.getElementById("clusterList");
  if (!host) return;
  host.innerHTML = "";
  _clusterRowByItem.clear();
  clusterItems.forEach(d => {
    const li = document.createElement("li");
    if (d.visible) li.classList.add("is-visible");
    const id = "clcb_" + Math.random().toString(36).slice(2, 10);
    const dotMark = document.createElement("span");
    dotMark.className = "dot-mark";
    dotMark.style.background = "#142239";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.id = id;
    cb.checked = !!d.visible;
    cb.addEventListener("change", () => setClusterVisible(d, cb.checked));
    const lbl = document.createElement("label");
    lbl.htmlFor = id;
    lbl.textContent = clusterText(d);
    lbl.title = `${d.ref.member_concepts.length} member concept(s)`;
    lbl.addEventListener("click", (ev) => {
      // Click on text (not checkbox) → prompt rename, don't toggle visibility.
      if (ev.target === cb) return;
      ev.preventDefault();
      const name = prompt(
        `Rename cluster (members: ${d.ref.member_concepts.length} concepts):`,
        d.label);
      if (name !== null && name.trim()) {
        d.label = name.trim();
        clusterLayer.selectAll("text.cluster-label")
          .filter(l => l === d)
          .each(function(dd) {
            setMultilineText(d3.select(this), clusterPlotLines(dd));
          });
        lbl.textContent = clusterText(d);
        updateSidebarClusterNames();
      }
    });
    li.appendChild(dotMark);
    li.appendChild(cb);
    li.appendChild(lbl);
    host.appendChild(li);
    _clusterRowByItem.set(d, {li, cb});
  });
}

function updateSidebarClusterNames() {
  _rowByItem.forEach((row, item) => {
    const cid = item.ref.cluster_id;
    const name = cid < 0 ? "noise" : clusterNameById(cid);
    const chip = row.li.querySelector(".cluster-chip");
    if (chip) chip.textContent = name;
  });
}

function setK(newK) {
  const prevItems = clusterItems;
  // Save any renames made at the K we're leaving.
  if (prevItems && prevItems.length) {
    snapshotCurrentK();
    // Fold the just-saved per-K renames back into savedClusterByCidByK
    // so buildClusterItems picks them up when the user returns to this K.
    if (!savedClusterByCidByK) savedClusterByCidByK = {};
    const m = new Map();
    (clusterRenamesByK[String(currentK)] || []).forEach(c =>
      m.set(c.cluster_id, c));
    savedClusterByCidByK[currentK] = m;
  }
  currentK = newK;
  const assignment = payload.clusterings[String(newK)].concept_cluster_id;
  payload.concepts.forEach((c, i) => { c.cluster_id = assignment[i]; });
  clusterItems = buildClusterItems(newK, prevItems);
  // Component clusters covered by a merge at this K should start hidden
  // (merges auto-hide originals; losing that state on K-flip would show
  // duplicate labels under the merged one).
  clusterItems.forEach(d => {
    if (clusterInSomeMerge(d.ref.id)) d.visible = false;
  });
  renderContours(payload.clusterings[String(newK)].contours);
  renderClusterLabels();
  renderMergedLabels();
  renderClusterSidebar();
  updateSidebarClusterNames();
  clearMergeSelection();
  document.getElementById("kValue").textContent = String(newK);
  document.getElementById("kSlider").value = String(newK);
}

// ---- Saved positions (hydrate labels + clusters + ovals) ---------------
// (declaration must come before conceptItems/clusterItems above build.)

// ---- Sidebar list -------------------------------------------------------
const _rowByItem = new Map();
const listEl = document.getElementById("conceptList");
// Sort: show visible/high-freq first.
const sortedItems = conceptItems.slice().sort((a, b) =>
  b.ref.posterior_freq - a.ref.posterior_freq);
sortedItems.forEach(d => {
  const li = document.createElement("li");
  if (d.labelVisible) li.classList.add("is-visible");
  const id = "cb_" + Math.random().toString(36).slice(2, 10);
  const dotMark = document.createElement("span");
  dotMark.className = "dot-mark";
  dotMark.style.background = d.ref.explored_only ? EXPLORED_COLOR : DOT_COLOR;
  const cb = document.createElement("input");
  cb.type = "checkbox"; cb.id = id;
  cb.checked = d.labelVisible;
  cb.addEventListener("change", () => setLabelVisible(d, cb.checked));
  const lbl = document.createElement("label");
  lbl.htmlFor = id;
  const chip = document.createElement("span");
  chip.className = "cluster-chip";
  chip.textContent = d.ref.cluster_id < 0 ? "noise" : clusterNameById(d.ref.cluster_id);
  lbl.appendChild(chip);
  lbl.appendChild(document.createTextNode(shortLabel(d.ref.concept)));
  lbl.title = d.ref.concept +
    `\nposterior_freq ${d.ref.posterior_freq.toFixed(3)}` +
    `\nmean_coef ${d.ref.mean_coef.toFixed(3)}`;
  li.appendChild(dotMark);
  li.appendChild(cb);
  li.appendChild(lbl);
  listEl.appendChild(li);
  _rowByItem.set(d, {li, cb});
});

document.getElementById("conceptSearch").addEventListener("input", (ev) => {
  const q = ev.target.value.toLowerCase().trim();
  _rowByItem.forEach((row, item) => {
    const hay = (item.ref.concept + " " + shortLabel(item.ref.concept)).toLowerCase();
    row.li.classList.toggle("hidden-by-search", q && !hay.includes(q));
  });
});

// ---- Reset / export -----------------------------------------------------
function resetPositions() {
  conceptItems.forEach(d => {
    d.labelX = d.anchorX + d.initialDx;
    d.labelY = d.anchorY + d.initialDy;
  });
  conceptLabels.attr("x", d => d.labelX).attr("y", d => d.labelY);
  conceptLeaders.attr("x2", d => d.labelX).attr("y2", d => d.labelY);
  clusterItems.forEach(d => {
    d.labelX = d.defaultX; d.labelY = d.defaultY;
  });
  renderClusterLabels();
}

// Store cluster-label edits for the K the user is viewing; preserve
// edits from other K's that were loaded from disk.
const clusterRenamesByK = {};
if (savedClusterByCidByK) {
  Object.keys(savedClusterByCidByK).forEach(kStr => {
    const arr = [];
    savedClusterByCidByK[kStr].forEach((v, cid) => arr.push({
      cluster_id: cid,
      label: v.label,
      labelX: v.labelX, labelY: v.labelY,
      visible: v.visible,
    }));
    clusterRenamesByK[kStr] = arr;
  });
}

function snapshotCurrentK() {
  clusterRenamesByK[String(currentK)] = clusterItems.map(d => ({
    cluster_id: d.ref.id,
    label: d.label,
    labelX: Math.round(d.labelX),
    labelY: Math.round(d.labelY),
    visible: !!d.visible,
  }));
}

function exportPositions() {
  snapshotCurrentK();
  // Serialize merges keyed by K; keep only entries with ≥2 cluster_ids.
  const mergesOut = {};
  Object.keys(mergesByK).forEach(kStr => {
    const arr = (mergesByK[kStr] || []).filter(m => m.cluster_ids.length >= 2);
    if (!arr.length) return;
    mergesOut[kStr] = arr.map(m => ({
      id: m.id,
      cluster_ids: m.cluster_ids.slice(),
      label: m.label,
      labelX: m.labelX != null ? Math.round(m.labelX) : null,
      labelY: m.labelY != null ? Math.round(m.labelY) : null,
      visible: m.visible !== undefined ? !!m.visible : true,
    }));
  });
  const out = {
    concept_labels: conceptItems.map(d => ({
      concept: d.ref.concept,
      labelX: Math.round(d.labelX),
      labelY: Math.round(d.labelY),
      visible: !!d.labelVisible,
    })),
    cluster_labels: clusterRenamesByK,
    cluster_merges: mergesOut,
    current_k: currentK,
    ovals: ovals.map(o => ({
      cx: Math.round(o.cx), cy: Math.round(o.cy),
      rx: Math.round(o.rx), ry: Math.round(o.ry),
      label: o.label || "",
    })),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)],
                       {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "posterior_concepts_positions.json";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function exportPdf() {
  if (!window.jspdf || !window.svg2pdf) {
    alert("PDF libs failed to load — check your internet connection.");
    return;
  }
  const svgNode = document.getElementById("chart");
  const w = +svgNode.getAttribute("width");
  const h = +svgNode.getAttribute("height");
  const { jsPDF } = window.jspdf;
  const pdf = new jsPDF({orientation: w >= h ? "landscape" : "portrait",
                         unit: "pt", format: [w, h]});
  const clone = svgNode.cloneNode(true);
  clone.querySelectorAll(
    ".anno-delete-bg, .anno-delete-x, .anno-handle, .hidden"
  ).forEach(n => n.remove());
  await window.svg2pdf.svg2pdf(clone, pdf, {x: 0, y: 0, width: w, height: h});
  pdf.save("posterior_concepts.pdf");
}

// ---- Legend (HTML, rendered below the plot) ----------------------------
function renderLegend() {
  const host = document.getElementById("legend");
  if (!host) return;
  // Build via inline SVG snippets so the swatches/dots render crisply.
  const hpdSwatches = LEVEL_ORDER.map(lvl => `
    <span class="legend-item">
      <svg width="26" height="12" style="vertical-align:middle">
        <rect x="0" y="1" width="26" height="10"
              fill="${HPD_FILLS[lvl]}" fill-opacity="${HPD_ALPHAS[lvl]}"/>
      </svg>
      <span class="legend-label">${Math.round(lvl*100)}%</span>
    </span>`).join("");
  const dotSwatch = `
    <span class="legend-item">
      <svg width="26" height="26" style="vertical-align:middle">
        <circle cx="13" cy="13" r="${UNIFORM_DOT_R}" fill="${DOT_COLOR}"
                stroke="#fff" stroke-width="0.6"/>
      </svg>
      <span class="legend-label">concept</span>
    </span>
    <span class="legend-item">
      <svg width="26" height="26" style="vertical-align:middle">
        <circle cx="13" cy="13" r="${UNIFORM_DOT_R}" fill="${DOT_COLOR}"
                fill-opacity="0.5"/>
      </svg>
      <span class="legend-label">jittered replicate (density ∝ posterior)</span>
    </span>`;
  host.innerHTML = `
    <span class="legend-group"><span class="legend-title">Posterior HPD</span> ${hpdSwatches}</span>
    <span class="legend-group"><span class="legend-title">Dots</span> ${dotSwatch}</span>
    <span class="legend-group">
      <svg width="12" height="12" style="vertical-align:middle">
        <circle cx="6" cy="6" r="2.5" fill="${EXPLORED_COLOR}" fill-opacity="0.55"/>
      </svg>
      <span class="legend-label">explored (&lt; ${payload.explored_threshold.toFixed(2)})</span>
    </span>`;
}
renderLegend();

// =============================================================================
// Annotation ovals (manual overrides, independent of auto-clusters)
// =============================================================================
const ovalLayer = g.append("g").attr("class", "anno-layer");
const OVAL_FILL = "#4a6b8a";
const CLUSTER_TEXT = "#2b3a4a";
let ovals = [];
let annoMode = false;
let selectedOval = null;

function toggleAnnoMode() {
  annoMode = !annoMode;
  const btn = document.getElementById("annoBtn");
  const hint = document.getElementById("annoHint");
  btn.classList.toggle("active", annoMode);
  if (annoMode) {
    svg.node().classList.add("anno-mode");
    hint.textContent = "click-drag to draw an oval · dblclick to delete · shift-click to rename";
  } else {
    svg.node().classList.remove("anno-mode");
    hint.textContent = "";
  }
}

function clearOvals() {
  if (!ovals.length) return;
  if (!confirm(`Delete all ${ovals.length} annotation oval(s)?`)) return;
  ovals = [];
  renderOvals();
}

function renderOvals() {
  const sel = ovalLayer.selectAll("g.anno").data(ovals, d => d._id);
  const enter = sel.enter().append("g").attr("class", "anno");
  enter.append("ellipse").attr("class", "anno-oval")
    .attr("fill", OVAL_FILL).attr("fill-opacity", 0.1).attr("stroke", "none");
  enter.append("circle").attr("class", "anno-handle")
    .attr("r", 5).attr("fill", "#d4a017").attr("stroke", "#fff").attr("stroke-width", 1);
  enter.append("circle").attr("class", "anno-delete-bg")
    .attr("r", 8).attr("fill", "#c0392b").attr("stroke", "#fff").attr("stroke-width", 1);
  enter.append("text").attr("class", "anno-delete-x")
    .attr("fill", "#fff").attr("font-size", 12).attr("font-weight", 700)
    .attr("text-anchor", "middle").attr("dominant-baseline", "central").text("×");
  enter.append("text").attr("class", "anno-label")
    .attr("fill", CLUSTER_TEXT).attr("font-size", 24).attr("font-weight", 700)
    .attr("font-family", "Helvetica").attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle");

  sel.exit().remove();
  const merged = enter.merge(sel);
  merged.select("ellipse.anno-oval")
    .attr("cx", d => d.cx).attr("cy", d => d.cy)
    .attr("rx", d => d.rx).attr("ry", d => d.ry)
    .attr("fill-opacity", d => d === selectedOval ? 0.16 : 0.1)
    .on("mousedown", ovalMouseDown)
    .on("dblclick", (event, d) => {
      event.stopPropagation();
      ovals = ovals.filter(o => o !== d);
      if (selectedOval === d) selectedOval = null;
      renderOvals();
    });
  merged.select("circle.anno-handle")
    .attr("cx", d => d.cx + d.rx).attr("cy", d => d.cy + d.ry)
    .on("mousedown", handleMouseDown);
  const deleteOval = (event, d) => {
    event.stopPropagation(); event.preventDefault();
    ovals = ovals.filter(o => o !== d);
    if (selectedOval === d) selectedOval = null;
    renderOvals();
  };
  merged.select("circle.anno-delete-bg")
    .attr("cx", d => d.cx + d.rx).attr("cy", d => d.cy - d.ry)
    .on("mousedown", deleteOval)
    .on("mouseenter", function() { this.parentNode.classList.add("show-delete"); })
    .on("mouseleave", function() { this.parentNode.classList.remove("show-delete"); });
  merged.select("text.anno-delete-x")
    .attr("x", d => d.cx + d.rx).attr("y", d => d.cy - d.ry);
  merged.select("text.anno-label")
    .attr("x", d => d.cx).attr("y", d => d.cy)
    .text(d => d.label || "")
    .on("mousedown", (event, d) => {
      event.stopPropagation();
      const name = prompt("Oval label:", d.label || "");
      if (name !== null) { d.label = name; renderOvals(); }
    })
    .on("mouseenter", function() { this.parentNode.classList.add("show-delete"); })
    .on("mouseleave", function() { this.parentNode.classList.remove("show-delete"); });
}

function ovalMouseDown(event, d) {
  event.stopPropagation();
  if (event.shiftKey) {
    const name = prompt("Oval label:", d.label || "");
    if (name !== null) { d.label = name; renderOvals(); }
    return;
  }
  selectedOval = d;
  const [x0, y0] = d3.pointer(event, g.node());
  const cx0 = d.cx, cy0 = d.cy;
  function onMove(ev) {
    const [x, y] = d3.pointer(ev, g.node());
    d.cx = cx0 + (x - x0); d.cy = cy0 + (y - y0);
    renderOvals();
  }
  function onUp() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  }
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}

function handleMouseDown(event, d) {
  event.stopPropagation();
  selectedOval = d;
  function onMove(ev) {
    const [x, y] = d3.pointer(ev, g.node());
    d.rx = Math.max(6, Math.abs(x - d.cx));
    d.ry = Math.max(6, Math.abs(y - d.cy));
    renderOvals();
  }
  function onUp() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  }
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}

svg.on("mousedown", function(event) {
  if (!annoMode) return;
  if (event.target.closest(".anno")) return;
  const [x0, y0] = d3.pointer(event, g.node());
  const o = {_id: Date.now() + Math.random(),
             cx: x0, cy: y0, rx: 1, ry: 1, label: ""};
  ovals.push(o); selectedOval = o; renderOvals();
  function onMove(ev) {
    const [mx, my] = d3.pointer(ev, g.node());
    const cx = (x0 + mx) / 2, cy = (y0 + my) / 2;
    o.cx = cx; o.cy = cy;
    o.rx = Math.max(1, Math.abs(mx - cx));
    o.ry = Math.max(1, Math.abs(my - cy));
    renderOvals();
  }
  function onUp() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    if (o.rx < 8 || o.ry < 8) {
      ovals = ovals.filter(x => x !== o);
      selectedOval = null; renderOvals(); return;
    }
    const name = prompt("Oval label (optional):", "");
    if (name !== null) o.label = name;
    renderOvals();
  }
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
});

if (savedOvalList && savedOvalList.length) {
  ovals = savedOvalList.map(o => ({
    _id: Date.now() + Math.random(),
    cx: o.cx, cy: o.cy,
    rx: o.rx != null ? o.rx : o.r, ry: o.ry != null ? o.ry : o.r,
    label: o.label || "",
  }));
  renderOvals();
}

// ---- K slider bootstrap -------------------------------------------------
(function initKSlider() {
  const slider = document.getElementById("kSlider");
  slider.min = String(payload.k_min);
  slider.max = String(payload.k_max);
  slider.step = "1";
  // If saved JSON carries a current_k, prefer it. Otherwise default_k.
  const savedK = (savedRaw && savedRaw.current_k != null) ? savedRaw.current_k : payload.default_k;
  const initialK = Math.max(payload.k_min, Math.min(payload.k_max, savedK));
  slider.value = String(initialK);
  slider.addEventListener("input", (ev) => {
    const k = parseInt(ev.target.value, 10);
    if (k !== currentK) setK(k);
  });
  // Initial render.
  setK(initialK);
})();
</script>
</body>
</html>
"""


def _load_iter_rows(history: TrainingHistory,
                    provider_identity_coef_offset: int) -> pd.DataFrame:
    """Long-form (iter, concept, coef) table over the training history.

    When provider identity is one-hot encoded as force-keep columns, the
    first `provider_identity_coef_offset` coefficients belong to those
    columns rather than to the concepts; skip them when aligning each
    concept to its coef.
    """
    num_fitted = len(history._coefs)
    offset = len(history._concepts) - num_fitted

    rows = []
    for i in range(num_fitted):
        concept_dicts = history._concepts[i + offset]
        coefs = np.asarray(history._coefs[i]).reshape(-1)
        for j, cd in enumerate(concept_dicts):
            k = j + provider_identity_coef_offset
            rows.append({
                "iter": i,
                "concept": cd["concept"],
                "coef": float(coefs[k]) if k < len(coefs) else np.nan,
            })
    return pd.DataFrame(rows)


def _summarize_concepts(long_df: pd.DataFrame,
                        num_posterior_iters: int) -> pd.DataFrame:
    """Per-concept posterior summary over the trailing posterior window.

    Includes every concept proposed in the trailing window — including those
    with low frequency — so the caller can style "explored but rarely kept"
    concepts differently.
    """
    max_iter = long_df["iter"].max()
    cutoff = max_iter - num_posterior_iters + 1
    last = long_df[long_df["iter"] >= cutoff].dropna(subset=["coef"]).copy()
    denom = last["iter"].nunique()

    summary = (
        last.groupby("concept", as_index=False)
        .agg(mean_coef=("coef", "mean"), n_iters=("iter", "nunique"))
    )
    summary["posterior_freq"] = summary["n_iters"] / denom
    summary = summary.sort_values("posterior_freq", ascending=False)
    return summary.reset_index(drop=True)


def _cluster_concepts_range(embeddings: np.ndarray,
                            k_min: int, k_max: int) -> dict:
    """Agglomerative clustering (cosine + average linkage) for each K in range.

    Returns {k: labels_array}. Operates on full-dim embeddings so semantic
    grouping is faithful to meaning, not to the 2D PCA layout.
    """
    n = embeddings.shape[0]
    out = {}
    for k in range(k_min, k_max + 1):
        if k >= n:
            out[k] = np.arange(n, dtype=int)  # each point its own cluster
            continue
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="complete")
        out[k] = model.fit_predict(embeddings).astype(int)
    return out


def _fit_gmm_range(pcs: np.ndarray,
                   weights: np.ndarray,
                   k_min: int,
                   k_max: int,
                   seed: int = 0,
                   eps_jitter: float = 0.01) -> dict:
    """Fit a weighted GaussianMixture in 2D PC space for each K in range.

    GMM is fit jointly across components (EM), so cluster densities are
    estimated simultaneously rather than per-cluster. Returns
    {k: {"labels": (n,), "means": (k, 2), "covariances": (k, 2, 2),
         "weights": (k,)}}.

    Weighting via row replication: sklearn's GaussianMixture doesn't take
    sample weights, so each concept is replicated proportional to its
    `posterior_freq`. Replicates are jittered by Gaussian noise of std
    `eps_jitter` (in original PC units) so EM sees them as nearby — but not
    identical — samples. Larger `eps_jitter` → smoother density, wider HPD
    ellipses; smaller → tighter clusters that hug exact point locations.
    `eps_jitter=0` recovers "pure replication" (identical points, which
    inflates the effective sample size without adding spatial smoothing).
    """
    rng = np.random.default_rng(seed)
    n = pcs.shape[0]

    # Replicate rows proportional to posterior_freq. Multiplier is tuned so
    # even the lowest-weight concept gets at least 1 copy.
    w = np.asarray(weights, dtype=float)
    w = np.where(w > 0, w, w[w > 0].min() if (w > 0).any() else 1.0)
    reps = np.maximum(1, np.rint(w / w.min()).astype(int))
    pcs_rep = np.repeat(pcs, reps, axis=0)
    origin_idx = np.repeat(np.arange(n), reps)
    if eps_jitter > 0:
        pcs_rep = pcs_rep + rng.normal(0.0, eps_jitter, size=pcs_rep.shape)

    out = {}
    for k in range(k_min, k_max + 1):
        eff_k = min(k, pcs_rep.shape[0])
        model = GaussianMixture(
            n_components=eff_k,
            covariance_type="full",
            reg_covar=1e-4,
            random_state=seed,
            n_init=5,
            init_params='random_from_data',
        )
        model.fit(pcs_rep)
        # Hard-assign each ORIGINAL (un-jittered) point to its most-likely component.
        resp = model.predict_proba(pcs)
        labels = resp.argmax(axis=1).astype(int)
        out[k] = {
            "labels": labels,
            "means": model.means_,
            "covariances": model.covariances_,
            "weights": model.weights_,
            "origin_idx": origin_idx,
            "jittered_points": pcs_rep,  # (sum(reps), 2); same across K
        }
    return out


def _ellipse_polygon(mean: np.ndarray,
                     cov: np.ndarray,
                     level: float,
                     n_points: int = 128) -> list:
    """Return a closed polygon (list of (x, y)) tracing the HPD ellipse.

    For a 2D Gaussian, the `level`-HPD region is the ellipse
    {x : (x-mu)^T Sigma^{-1} (x-mu) <= chi2_2(level)}. Boundary points are
    mean + sqrt(c) * L @ unit_circle, where L is Cholesky-like
    (eigendecomposition here, so it's robust to tiny covariances).
    """
    c = chi2.ppf(level, df=2)
    # Eigendecomposition is more numerically stable than Cholesky for
    # near-singular covariances.
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, n)
    pts = vecs @ np.diag(np.sqrt(c * vals)) @ unit  # (2, n)
    pts = pts.T + mean  # (n, 2)
    return [(float(x), float(y)) for x, y in pts]


def _gmm_hpd_contours(mean: np.ndarray,
                      cov: np.ndarray,
                      levels: tuple = (0.95, 0.80, 0.50)) -> dict:
    """HPD ellipses for a single Gaussian component — one polygon per level."""
    out = []
    for lvl in levels:
        poly = _ellipse_polygon(mean, cov, lvl)
        out.append({"level": float(lvl), "polygons": [poly]})
    return {"contours": out}


def _gmm_joint_density_contours(means: np.ndarray,
                                covariances: np.ndarray,
                                weights: np.ndarray,
                                xlim: tuple,
                                ylim: tuple,
                                grid_n: int = 200,
                                levels: tuple = (0.95, 0.80, 0.50)) -> dict:
    """HPD contours of the JOINT GMM density p(x,y) = Σ π_k N(x,y|μ_k,Σ_k).

    Unlike per-component ellipses, this produces smooth multimodal shapes:
    nearby components fuse into one blob, well-separated ones give distinct
    lobes. All weight is in a single contour layer (no per-cluster split).
    """
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)

    # Vectorized multivariate normal pdf: sum over K components.
    dens = np.zeros(grid.shape[0], dtype=float)
    for mu, cov, pi in zip(means, covariances, weights):
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        diff = grid - mu
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        comp = np.exp(-0.5 * maha) / (2.0 * np.pi * np.sqrt(det))
        dens += pi * comp
    dens = dens.reshape(grid_n, grid_n)

    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    flat = dens.ravel()
    order = np.argsort(flat)[::-1]
    sorted_mass = flat[order] * cell_area
    cumsum = np.cumsum(sorted_mass)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0

    out = []
    for lvl in levels:
        target = lvl * total
        idx = min(np.searchsorted(cumsum, target), len(flat) - 1)
        threshold = float(flat[order][idx])
        polys = _extract_contour_polys(xx, yy, dens, threshold)
        out.append({"level": float(lvl), "polygons": polys})
    return {"contours": out}


def _compute_hpd_contours(xy: np.ndarray,
                          weights: np.ndarray,
                          xlim: tuple,
                          ylim: tuple,
                          grid_n: int = 160,
                          levels: tuple = (0.95, 0.80, 0.50)) -> dict:
    """Weighted 2D KDE + HPD contours.

    Returns {"contours": [{"level": float, "polygons": [[(x,y), ...], ...]}]}
    per level, where each polygon is a list of (x,y) points at the density
    threshold enclosing `level` fraction of total posterior mass.
    """
    if xy.shape[0] < 3:
        return {"contours": []}

    kde = gaussian_kde(xy.T, weights=weights, bw_method="scott")
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()])
    dens = kde(grid).reshape(grid_n, grid_n)

    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    flat = dens.ravel()
    order = np.argsort(flat)[::-1]
    sorted_mass = flat[order] * cell_area
    cumsum = np.cumsum(sorted_mass)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0

    thresholds = []
    for lvl in levels:
        target = lvl * total
        idx = np.searchsorted(cumsum, target)
        idx = min(idx, len(flat) - 1)
        thresholds.append(float(flat[order][idx]))

    out = []
    for lvl, t in zip(levels, thresholds):
        polys = _extract_contour_polys(xx, yy, dens, t)
        out.append({"level": float(lvl), "polygons": polys})
    return {"contours": out}


def _extract_contour_polys(xx, yy, dens, threshold):
    """Use matplotlib's contouring to turn a density grid into closed paths."""
    fig = mpl.figure.Figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(xx, yy, dens, levels=[threshold])
    polys = []
    if hasattr(cs, "allsegs") and cs.allsegs:
        for seg in cs.allsegs[0]:
            if len(seg) >= 3:
                polys.append([(float(x), float(y)) for x, y in seg])
    return polys


def _cluster_name(cid: int) -> str:
    """0 -> 'A', 1 -> 'B', ..., 25 -> 'Z', 26 -> 'AA', ..."""
    s = ""
    n = cid
    while True:
        s = chr(ord("A") + (n % 26)) + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


# ============================================================================
# Matplotlib export (static SVG/PDF/PNG panel matching the D3 figure).
#
# The D3 panel is a 1400x900 SVG with a translated inner group `g`
# (margin={top:24,right:40,bottom:56,left:68}). Every saved coordinate
# (concept labels, cluster labels, annotation ovals) lives in that inner-box
# pixel space. The matplotlib renderer draws in that same (x,y) space with
# the y-axis flipped so SVG-y "down" matches image "down".
# ============================================================================

# These mirror the constants in _D3_TEMPLATE so the static export is
# visually identical to the browser rendering.
_DOT_COLOR = "#1f3b60"
_EXPLORED_COLOR = "#9a9a9a"
_UNIFORM_DOT_R = 4  # svg px
_HPD_FILLS = {0.95: "#dde7f1", 0.80: "#b4ccea", 0.50: "#7fa6d4"}
_HPD_ALPHAS = {0.95: 0.55, 0.80: 0.65, 0.50: 0.75}
_LEVEL_ORDER = (0.95, 0.80, 0.50)
_OVAL_FILL = "#4a6b8a"
_CLUSTER_TEXT = "#142239"
_AXIS_COLOR = "#808080"

_CANVAS_W = 1400
_CANVAS_H = 900
# Top/right/bottom margins are enlarged relative to the HTML (which uses
# top=24) because the static panel draws its title and PC labels inside
# the Axes box; the HTML renders those in sibling DOM elements, outside
# the SVG canvas.
_MARGIN = {"top": 70, "right": 40, "bottom": 70, "left": 80}

# Display names for the axis titles in the aggregate grid. Keys match the
# `axis` payload field (which comes from the sconscript AXES list); values
# are the formal titles used in the figure.
_AXIS_DISPLAY = {
    "clinical_utility": "Clinical Utility",
    "accuracy": "Accuracy",
    "authoritativeness": "Source Quality",
    "verifiability": "Verifiability",
    "completeness": "Completeness",
}


def _short_label(s: str) -> str:
    """Mirror of shortLabel() in the D3 template."""
    t = s.strip()
    for prefix in ("Does Answer B ", "Is Answer B "):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):]
            break
    low = t.lower()
    if low.endswith(" than answer a?"):
        t = t[: -len(" than answer a?")]
    elif low.endswith(" than answer a"):
        t = t[: -len(" than answer a")]
    if t:
        t = t[0].upper() + t[1:]
    return t


def _wrap_two_lines(text: str, max_chars: int = 22) -> str:
    """Break `text` onto at most two lines by picking the word boundary closest
    to the midpoint. Short strings pass through unchanged so single-word
    labels never get artificially split.
    """
    if len(text) <= max_chars or " " not in text:
        return text
    words = text.split(" ")
    if len(words) == 1:
        return text
    target = len(text) // 2
    # Pick the space whose cumulative char position is nearest target.
    best_split = 1
    best_dist = float("inf")
    acc = 0
    for i, w in enumerate(words[:-1]):
        acc += len(w) + 1  # +1 for the space we would turn into a newline
        dist = abs(acc - target)
        if dist < best_dist:
            best_dist = dist
            best_split = i + 1
    return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])


def _build_payload(history: "TrainingHistory",
                   *,
                   num_posterior_iters: int,
                   provider_identity_coef_offset: int,
                   embedding_model: str,
                   explored_threshold: float,
                   k_min: int,
                   k_max: int,
                   default_k: int,
                   cluster_method: str,
                   seed: int,
                   gmm_eps_jitter: float,
                   axis_name: str,
                   saved_positions: dict | None) -> dict:
    """Build the full D3 payload dict from a TrainingHistory.

    Factored out of main() so the matplotlib exporter can reuse it without
    going through the HTML round-trip.
    """
    long_df = _load_iter_rows(history, provider_identity_coef_offset)
    if long_df.empty:
        raise RuntimeError("No rows loaded from training history")

    summary = _summarize_concepts(long_df, num_posterior_iters)
    if len(summary) < 2:
        raise RuntimeError(
            f"Need >=2 unique concepts in the posterior window; got "
            f"{len(summary)}")

    embedder = SentenceTransformer(embedding_model, local_files_only=True)
    emb = embedder.encode(summary["concept"].tolist(), show_progress_bar=False)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(emb)
    summary["pc1"] = pcs[:, 0]
    summary["pc2"] = pcs[:, 1]

    k_max_eff = min(k_max, max(k_min, len(summary) - 1))
    k_min_eff = min(k_min, k_max_eff)
    if cluster_method == "agglomerative":
        cluster_by_k = _cluster_concepts_range(np.asarray(emb),
                                               k_min_eff, k_max_eff)
        gmm_by_k = None
    else:
        pcs_2d = summary[["pc1", "pc2"]].to_numpy()
        gmm_by_k = _fit_gmm_range(
            pcs_2d, summary["posterior_freq"].to_numpy(),
            k_min_eff, k_max_eff, seed=seed,
            eps_jitter=gmm_eps_jitter)
        cluster_by_k = {k: v["labels"] for k, v in gmm_by_k.items()}

    pad_frac = 0.08
    x_lo, x_hi = summary["pc1"].min(), summary["pc1"].max()
    y_lo, y_hi = summary["pc2"].min(), summary["pc2"].max()
    x_pad = pad_frac * (x_hi - x_lo)
    y_pad = pad_frac * (y_hi - y_lo)
    xlim = (float(x_lo - x_pad), float(x_hi + x_pad))
    ylim = (float(y_lo - y_pad), float(y_hi + y_pad))

    clusterings_out = {}
    for k, labels in cluster_by_k.items():
        df_k = summary.copy()
        df_k["cluster_id"] = labels
        clusters_out = []
        contours_out = []
        concept_cluster = labels.tolist()
        for cid, group in df_k.groupby("cluster_id"):
            xy = group[["pc1", "pc2"]].to_numpy()
            w = group["posterior_freq"].to_numpy()
            posterior_mass = float(group["posterior_freq"].sum())
            if w.sum() > 0:
                center_x = float(
                    (group["pc1"] * group["posterior_freq"]).sum() / w.sum())
                center_y = float(
                    (group["pc2"] * group["posterior_freq"]).sum() / w.sum())
            else:
                center_x = float(group["pc1"].mean())
                center_y = float(group["pc2"].mean())
            clusters_out.append({
                "id": int(cid),
                "default_label": f"Cluster {_cluster_name(int(cid))}",
                "center_pc1": center_x,
                "center_pc2": center_y,
                "posterior_mass": posterior_mass,
                "member_concepts": group["concept"].tolist(),
            })
            if gmm_by_k is None and len(group) >= 3 and w.sum() > 0:
                kde_out = _compute_hpd_contours(xy, w, xlim, ylim)
                if kde_out.get("contours"):
                    contours_out.append({
                        "cluster_id": int(cid),
                        "levels": kde_out["contours"],
                    })
        if gmm_by_k is not None:
            joint = _gmm_joint_density_contours(
                gmm_by_k[k]["means"],
                gmm_by_k[k]["covariances"],
                gmm_by_k[k]["weights"],
                xlim, ylim,
            )
            contours_out.append({
                "cluster_id": -1,
                "levels": joint["contours"],
            })
        clusterings_out[str(int(k))] = {
            "clusters": clusters_out,
            "contours": contours_out,
            "concept_cluster_id": concept_cluster,
        }

    default_k_eff = max(k_min_eff, min(default_k, k_max_eff))
    concepts_out = []
    default_assignment = cluster_by_k[default_k_eff]
    for i, row in summary.iterrows():
        is_explored = row["posterior_freq"] < explored_threshold
        concepts_out.append({
            "id": int(i),
            "concept": row["concept"],
            "pc1": float(row["pc1"]),
            "pc2": float(row["pc2"]),
            "posterior_freq": float(row["posterior_freq"]),
            "mean_coef": float(row["mean_coef"]),
            "cluster_id": int(default_assignment[i]),
            "explored_only": bool(is_explored),
        })

    jitter_points = []
    if gmm_by_k is not None:
        jp = gmm_by_k[default_k_eff]["jittered_points"]
        oi = gmm_by_k[default_k_eff]["origin_idx"]
        jitter_points = [[float(x), float(y), int(idx)]
                         for (x, y), idx in zip(jp, oi)]

    return {
        "concepts": concepts_out,
        "clusterings": clusterings_out,
        "k_min": int(k_min_eff),
        "k_max": int(k_max_eff),
        "default_k": int(default_k_eff),
        "explored_threshold": float(explored_threshold),
        "xlim": [xlim[0], xlim[1]],
        "ylim": [ylim[0], ylim[1]],
        "axis": axis_name,
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "saved_positions": saved_positions,
        "jitter_points": jitter_points,
    }


def _pc_to_px_scales(payload: dict):
    """SVG-pixel scales (inner-box coords).

    Each panel's plot area fills the full inner box (margin.left to
    canvas-right-margin, margin.top to canvas-bottom-margin). Dropping the
    equal-aspect centering trade (which the D3 HTML uses) is deliberate:
    anchoring every cell's plot box to the same (margin.left, margin.top)
    corner is what makes the panels line up across the 2x3 grid. The PC
    axis values stretch to fit the fixed box, which is fine — relative
    layout within each panel is preserved for saved label positions.
    """
    inner_w = _CANVAS_W - _MARGIN["left"] - _MARGIN["right"]
    inner_h = _CANVAS_H - _MARGIN["top"] - _MARGIN["bottom"]
    x_dom_span = payload["xlim"][1] - payload["xlim"][0]
    y_dom_span = payload["ylim"][1] - payload["ylim"][0]
    eff_w = inner_w
    eff_h = inner_h
    x_offset = 0.0
    y_offset = 0.0

    def x_pc_to_px(pc):
        pc = np.asarray(pc, dtype=float)
        return x_offset + (pc - payload["xlim"][0]) / x_dom_span * eff_w

    def y_pc_to_px(pc):
        # SVG y grows downward; mirror d3's scaleLinear range inversion.
        pc = np.asarray(pc, dtype=float)
        return y_offset + eff_h - (pc - payload["ylim"][0]) / y_dom_span * eff_h

    return x_pc_to_px, y_pc_to_px, inner_w, inner_h, x_offset, y_offset, eff_w, eff_h


def _html_centering_shift(payload: dict) -> tuple[float, float]:
    """Delta to apply to saved HTML label positions to re-center them here.

    The D3 HTML lays out an equal-aspect PC box CENTERED in its inner area:
        unit_px = min(inner_w / x_span, inner_h / y_span)
        xOffset = (inner_w - unit_px * x_span) / 2
        yOffset = (inner_h - unit_px * y_span) / 2
    All dragged labelX/labelY and oval cx/cy were stored in that centered
    frame. The static exporter here anchors the plot box to (0, 0) and
    stretches to fill, so saved coordinates need to be shifted by
    (-xOffset_html, -yOffset_html) AND rescaled from the HTML's equal-aspect
    pc-per-px to the stretched pc-per-px.

    Returns (dx, dy) to add to saved pixel coords. Rescaling can't be
    captured as a pure shift when the HTML and here use different
    unit_px, so this helper also returns effective ratios; callers that
    need to rescale an offset-from-anchor use them explicitly.
    """
    inner_w = _CANVAS_W - _MARGIN["left"] - _MARGIN["right"]
    inner_h = _CANVAS_H - _MARGIN["top"] - _MARGIN["bottom"]
    x_span = payload["xlim"][1] - payload["xlim"][0]
    y_span = payload["ylim"][1] - payload["ylim"][0]
    unit_px_html = min(inner_w / x_span, inner_h / y_span)
    html_eff_w = unit_px_html * x_span
    html_eff_h = unit_px_html * y_span
    html_x_offset = (inner_w - html_eff_w) / 2.0
    html_y_offset = (inner_h - html_eff_h) / 2.0
    return -html_x_offset, -html_y_offset


def _remap_saved_px(payload: dict, x_px: float, y_px: float
                    ) -> tuple[float, float]:
    """Map an HTML-saved pixel coord into the stretched coord frame used here.

    Decomposes the saved coord into (anchor_relative_px) in the HTML frame,
    treats labels as living on top of the plot area, and rescales to the
    stretched frame so cluster labels keep their position relative to the
    cluster centroid in PC space.
    """
    inner_w = _CANVAS_W - _MARGIN["left"] - _MARGIN["right"]
    inner_h = _CANVAS_H - _MARGIN["top"] - _MARGIN["bottom"]
    x_span = payload["xlim"][1] - payload["xlim"][0]
    y_span = payload["ylim"][1] - payload["ylim"][0]
    unit_px_html = min(inner_w / x_span, inner_h / y_span)
    html_eff_w = unit_px_html * x_span
    html_eff_h = unit_px_html * y_span
    html_x_off = (inner_w - html_eff_w) / 2.0
    html_y_off = (inner_h - html_eff_h) / 2.0

    # Convert saved pixel → PC → stretched pixel. In the stretched frame
    # x_offset=y_offset=0 and eff_w=inner_w, eff_h=inner_h.
    pc_x = payload["xlim"][0] + (x_px - html_x_off) / html_eff_w * x_span
    pc_y = payload["ylim"][0] + (
        html_eff_h - (y_px - html_y_off)
    ) / html_eff_h * y_span

    new_x = (pc_x - payload["xlim"][0]) / x_span * inner_w
    new_y = inner_h - (pc_y - payload["ylim"][0]) / y_span * inner_h
    return float(new_x), float(new_y)


def _resolve_cluster_labels(payload: dict, k: int) -> list:
    """Resolve the cluster-label items for a given K, applying saved renames.

    Returns a list of dicts: {label, label_x, label_y, visible, member_count,
    posterior_mass, coef_abs_contrib, cluster_id}. Positions are in inner-box
    SVG pixels. `coef_abs_contrib` is the cluster's *expected absolute
    contribution*: Σ_i posterior_freq_i × |mean_coef_i| over its member
    concepts. Using the sum (not the weighted average) means a cluster whose
    members rarely enter the posterior gets a small size even if the few
    kept samples had large |coef|, which is the behavior we want for a
    label-prominence encoding. We use |coef| because clusters often lump
    opposite-signed concepts together.
    """
    x_pc_to_px, y_pc_to_px, inner_w, inner_h, *_ = _pc_to_px_scales(payload)

    clusters = payload["clusterings"][str(int(k))]["clusters"]
    concept_by_name = {c["concept"]: c for c in payload["concepts"]}
    saved = payload.get("saved_positions") or {}
    saved_cl = saved.get("cluster_labels") or {}
    if isinstance(saved_cl, dict):
        saved_for_k = {c["cluster_id"]: c for c in saved_cl.get(str(k), [])}
    else:
        # legacy flat list — treat as applying to default_k only
        if int(k) == int(payload["default_k"]):
            saved_for_k = {c["cluster_id"]: c for c in saved_cl}
        else:
            saved_for_k = {}

    items = []
    for c in clusters:
        default_x = float(x_pc_to_px(c["center_pc1"]))
        default_y = float(y_pc_to_px(c["center_pc2"])) - 20.0
        sv = saved_for_k.get(c["id"])
        if sv and sv.get("labelX") is not None and sv.get("labelY") is not None:
            label_x, label_y = _remap_saved_px(payload,
                                               float(sv["labelX"]),
                                               float(sv["labelY"]))
        else:
            label_x, label_y = default_x, default_y

        coef_abs_contrib = 0.0
        for name in c["member_concepts"]:
            cp = concept_by_name.get(name)
            if cp is None:
                continue
            coef_abs_contrib += (
                float(cp["posterior_freq"]) * abs(float(cp["mean_coef"]))
            )

        items.append({
            "cluster_id": int(c["id"]),
            "label": (sv.get("label") if sv and sv.get("label")
                      else c["default_label"]),
            "label_x": label_x,
            "label_y": label_y,
            "visible": bool(sv["visible"]) if sv and sv.get("visible") is not None
                       else True,
            "member_count": len(c["member_concepts"]),
            "posterior_mass": float(c["posterior_mass"]),
            "coef_abs_contrib": float(coef_abs_contrib),
        })
    return items


def _weighted_centroid_px(payload: dict, member_concepts: list[str]
                          ) -> tuple[float, float]:
    """Posterior-mass-weighted centroid in inner-box pixel space.

    Used both to anchor merged-cluster labels and to fall back when a
    cluster has no saved position override.
    """
    x_pc_to_px, y_pc_to_px, *_ = _pc_to_px_scales(payload)
    concept_by_name = {c["concept"]: c for c in payload["concepts"]}
    sum_w = 0.0
    sum_wx = 0.0
    sum_wy = 0.0
    for name in member_concepts:
        cp = concept_by_name.get(name)
        if cp is None:
            continue
        w = float(cp["posterior_freq"])
        sum_w += w
        sum_wx += w * float(cp["pc1"])
        sum_wy += w * float(cp["pc2"])
    if sum_w <= 0:
        return 0.0, 0.0
    cx = float(x_pc_to_px(sum_wx / sum_w))
    cy = float(y_pc_to_px(sum_wy / sum_w))
    return cx, cy


def _coef_abs_contrib(payload: dict, member_concepts: list[str]) -> float:
    """Σ posterior_freq × |mean_coef| over the given member concepts."""
    concept_by_name = {c["concept"]: c for c in payload["concepts"]}
    total = 0.0
    for name in member_concepts:
        cp = concept_by_name.get(name)
        if cp is None:
            continue
        total += float(cp["posterior_freq"]) * abs(float(cp["mean_coef"]))
    return float(total)


def _saved_section_for_k(payload: dict, section: str, k: int) -> list:
    """Pull the list of per-K saved entries for a given section.

    Supports both the legacy flat-list schema (which applies only to the
    default K) and the K-aware dict schema. Returns [] when nothing saved.
    """
    saved = payload.get("saved_positions") or {}
    sec = saved.get(section)
    if not sec:
        return []
    if isinstance(sec, dict):
        return list(sec.get(str(int(k))) or [])
    if int(k) == int(payload["default_k"]):
        return list(sec)
    return []


def _resolve_cluster_merges(payload: dict, k: int) -> list:
    """Resolve user-authored cluster merges into label items.

    Merges live in `saved_positions.cluster_merges`, scoped per K (because
    cluster_ids are only meaningful inside a specific clustering). Each
    merge entry has `{id, cluster_ids: [...], label, labelX?, labelY?,
    visible?}`. Returned items share the same shape as the regular cluster
    label items so the renderer can treat them uniformly.
    """
    clusters_for_k = payload["clusterings"][str(int(k))]["clusters"]
    clusters_by_id = {c["id"]: c for c in clusters_for_k}

    out = []
    for m in _saved_section_for_k(payload, "cluster_merges", k):
        child_ids = [int(cid) for cid in (m.get("cluster_ids") or [])
                     if int(cid) in clusters_by_id]
        if len(child_ids) < 2:
            continue
        members: list[str] = []
        mass = 0.0
        for cid in child_ids:
            c = clusters_by_id[cid]
            members.extend(c["member_concepts"])
            mass += float(c["posterior_mass"])

        if m.get("labelX") is not None and m.get("labelY") is not None:
            label_x, label_y = _remap_saved_px(payload,
                                               float(m["labelX"]),
                                               float(m["labelY"]))
        else:
            label_x, label_y = _weighted_centroid_px(payload, members)
            label_y -= 20.0  # match the per-cluster default y-nudge

        out.append({
            "cluster_id": None,
            "merge_id": m.get("id") or f"m_{'_'.join(str(c) for c in child_ids)}",
            "merged_cluster_ids": child_ids,
            "label": m.get("label") or "Merged cluster",
            "label_x": label_x,
            "label_y": label_y,
            "visible": bool(m["visible"]) if m.get("visible") is not None else True,
            "member_count": len(members),
            "posterior_mass": float(mass),
            "coef_abs_contrib": _coef_abs_contrib(payload, members),
            "origin": "merge",
        })
    return out


def _resolve_all_cluster_labels(payload: dict, k: int) -> list:
    """Regular cluster labels + merged labels in a single flat list.

    Any cluster absorbed by a merge is auto-hidden (visible=False) so the
    renderer draws only the merged label, not duplicates. The two kinds
    carry an `origin` field ("cluster" or "merge") when callers need to
    distinguish.
    """
    merges = _resolve_cluster_merges(payload, k)
    absorbed = set()
    for m in merges:
        absorbed.update(m["merged_cluster_ids"])

    items = []
    for ci in _resolve_cluster_labels(payload, k):
        ci = dict(ci)
        ci.setdefault("origin", "cluster")
        if ci["cluster_id"] in absorbed:
            ci["visible"] = False
        items.append(ci)
    items.extend(merges)
    return items


def _resolve_concept_labels(payload: dict) -> list:
    """Resolve per-concept labels (anchor + saved label pos + visibility)."""
    x_pc_to_px, y_pc_to_px, inner_w, inner_h, *_ = _pc_to_px_scales(payload)
    saved = payload.get("saved_positions") or {}
    saved_cp = {s["concept"]: s for s in (saved.get("concept_labels") or [])}

    items = []
    for c in payload["concepts"]:
        anchor_x = float(x_pc_to_px(c["pc1"]))
        anchor_y = float(y_pc_to_px(c["pc2"]))
        # D3 default: golden-angle ring at radius 50px.
        theta = c["id"] * 2.399963229
        dx = 50.0 * np.cos(theta)
        dy = -50.0 * np.sin(theta)
        sv = saved_cp.get(c["concept"])
        if sv and sv.get("labelX") is not None and sv.get("labelY") is not None:
            label_x, label_y = _remap_saved_px(payload,
                                               float(sv["labelX"]),
                                               float(sv["labelY"]))
        else:
            label_x, label_y = anchor_x + dx, anchor_y + dy
        items.append({
            "id": c["id"],
            "concept": c["concept"],
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "label_x": label_x,
            "label_y": label_y,
            "visible": bool(sv["visible"]) if sv and sv.get("visible") is not None
                       else False,
            "explored_only": c["explored_only"],
        })
    return items


def _resolve_ovals(payload: dict) -> list:
    """Rescale saved ovals from the HTML's centered equal-aspect frame to the
    stretched frame used here, so hand-drawn ovals keep their position
    relative to the underlying clusters."""
    inner_w = _CANVAS_W - _MARGIN["left"] - _MARGIN["right"]
    inner_h = _CANVAS_H - _MARGIN["top"] - _MARGIN["bottom"]
    x_span = payload["xlim"][1] - payload["xlim"][0]
    y_span = payload["ylim"][1] - payload["ylim"][0]
    unit_px_html = min(inner_w / x_span, inner_h / y_span)
    sx = inner_w / (unit_px_html * x_span)  # px stretch factor x
    sy = inner_h / (unit_px_html * y_span)  # px stretch factor y

    saved = payload.get("saved_positions") or {}
    out = []
    for o in (saved.get("ovals") or []):
        rx = o.get("rx", o.get("r", 0))
        ry = o.get("ry", o.get("r", 0))
        cx, cy = _remap_saved_px(payload, float(o["cx"]), float(o["cy"]))
        out.append({
            "cx": cx,
            "cy": cy,
            # Ovals also need their radii stretched (not just center shifted)
            # so they continue to hug the underlying density after the frame
            # change.
            "rx": float(rx) * sx,
            "ry": float(ry) * sy,
            "label": o.get("label", "") or "",
        })
    return out


_CLUSTER_FONT_MIN = 16.0  # smallest cluster-label font (at |coef|=0)
_CLUSTER_FONT_MAX = 34.0  # largest cluster-label font (at |coef|>=ref)


def _cluster_label_fontsize(coef_abs: float, coef_abs_ref: float) -> float:
    """Linear map |coef| → font size in [_CLUSTER_FONT_MIN, _CLUSTER_FONT_MAX].

    `coef_abs_ref` is the "big" |coef| that pins the top of the scale; values
    at or above it get `_CLUSTER_FONT_MAX`. Pass the same ref across every
    panel in a multi-axis figure so sizes read comparably across cells.
    """
    if coef_abs_ref <= 0 or not np.isfinite(coef_abs_ref):
        return _CLUSTER_FONT_MIN
    t = max(0.0, min(1.0, float(coef_abs) / float(coef_abs_ref)))
    return _CLUSTER_FONT_MIN + t * (_CLUSTER_FONT_MAX - _CLUSTER_FONT_MIN)


def render_payload_to_ax(ax,
                         payload: dict,
                         *,
                         k: int | None = None,
                         show_concept_labels: bool = True,
                         show_cluster_labels: bool = True,
                         show_title: bool = True,
                         show_axes_labels: bool = True,
                         font_scale: float = 1.0,
                         coef_abs_ref: float | None = None) -> None:
    """Render a static version of the posterior panel into an existing Axes.

    Mirrors the D3 template's geometry (1400x900 inner-box pixel space) so
    hand-adjusted saved positions overlay without re-fitting. Call from a
    standalone script or from the 2x3 aggregator.

    `k` selects which clustering to show; defaults to the saved `current_k`
    if present in the positions JSON, else `payload["default_k"]`.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon

    saved = payload.get("saved_positions") or {}
    if k is None:
        k = int(saved.get("current_k", payload["default_k"]))
    k = int(max(payload["k_min"], min(payload["k_max"], k)))

    (x_pc_to_px, y_pc_to_px, inner_w, inner_h,
     x_offset, y_offset, eff_w, eff_h) = _pc_to_px_scales(payload)

    # Draw into the full 1400x900 SVG canvas so the Axes mirrors the HTML
    # outer <svg> exactly. The inner `g` origin lives at (margin.left,
    # margin.top), so x=0 in label coords lands at canvas x=margin.left,
    # etc. — hand-dragged labelX/labelY from the HTML overlay 1:1.
    ax.set_xlim(-_MARGIN["left"], inner_w + _MARGIN["right"])
    ax.set_ylim(inner_h + _MARGIN["bottom"], -_MARGIN["top"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot-area axis lines (left + bottom), ticks, PC labels.
    ax.plot([x_offset, x_offset + eff_w],
            [y_offset + eff_h, y_offset + eff_h],
            color=_AXIS_COLOR, linewidth=0.75, zorder=1)
    ax.plot([x_offset, x_offset],
            [y_offset, y_offset + eff_h],
            color=_AXIS_COLOR, linewidth=0.75, zorder=1)

    def _ticks(lo, hi, n=5):
        import matplotlib.ticker as mtick
        loc = mtick.MaxNLocator(nbins=n)
        return loc.tick_values(lo, hi)

    xt_pc = _ticks(payload["xlim"][0], payload["xlim"][1])
    yt_pc = _ticks(payload["ylim"][0], payload["ylim"][1])
    for v in xt_pc:
        if payload["xlim"][0] <= v <= payload["xlim"][1]:
            xpx = float(x_pc_to_px(v))
            ax.plot([xpx, xpx],
                    [y_offset + eff_h, y_offset + eff_h + 4],
                    color=_AXIS_COLOR, linewidth=0.6, zorder=1)
            ax.text(xpx, y_offset + eff_h + 18, f"{v:g}",
                    ha="center", va="top", color=_AXIS_COLOR,
                    fontsize=18 * font_scale, family="Helvetica")
    for v in yt_pc:
        if payload["ylim"][0] <= v <= payload["ylim"][1]:
            ypx = float(y_pc_to_px(v))
            ax.plot([x_offset - 4, x_offset], [ypx, ypx],
                    color=_AXIS_COLOR, linewidth=0.6, zorder=1)
            ax.text(x_offset - 9, ypx, f"{v:g}",
                    ha="right", va="center", color=_AXIS_COLOR,
                    fontsize=18 * font_scale, family="Helvetica")

    if show_axes_labels:
        ax.text(x_offset + eff_w / 2.0, y_offset + eff_h + 52,
                "PC1",
                ha="center", va="center", color="#555",
                fontsize=22 * font_scale, fontweight=600, family="Helvetica")
        ax.text(x_offset - 54, y_offset + eff_h / 2.0,
                "PC2",
                ha="center", va="center", color="#555",
                fontsize=22 * font_scale, fontweight=600, family="Helvetica",
                rotation=90)

    # Contours (filled HPD polygons), outer level first so inner levels
    # layer on top. Polygons are in PC space; convert to inner-box pixels.
    contours_for_k = payload["clusterings"][str(int(k))]["contours"]
    for lvl in _LEVEL_ORDER:
        for c in contours_for_k:
            lvl_entry = next(
                (le for le in c["levels"]
                 if abs(le["level"] - lvl) < 1e-6), None)
            if lvl_entry is None:
                continue
            for poly in lvl_entry["polygons"]:
                if len(poly) < 3:
                    continue
                pts_x = x_pc_to_px(np.array([p[0] for p in poly]))
                pts_y = y_pc_to_px(np.array([p[1] for p in poly]))
                patch = Polygon(
                    np.column_stack([pts_x, pts_y]),
                    closed=True,
                    facecolor=_HPD_FILLS[lvl],
                    edgecolor="none",
                    alpha=_HPD_ALPHAS[lvl],
                    zorder=2,
                )
                ax.add_patch(patch)

    # Jittered replicates.
    for jp in (payload.get("jitter_points") or []):
        origin = payload["concepts"][int(jp[2])]
        r = 2.5 if origin["explored_only"] else _UNIFORM_DOT_R
        ax.scatter([float(x_pc_to_px(jp[0]))], [float(y_pc_to_px(jp[1]))],
                   s=(r * 2) ** 2 * 0.6, c=_DOT_COLOR, alpha=0.5,
                   edgecolors="none", zorder=3)

    # Concept dots. No stroke so dots read as pure color at small panel size.
    for c in payload["concepts"]:
        r = 2.5 if c["explored_only"] else _UNIFORM_DOT_R
        color = _EXPLORED_COLOR if c["explored_only"] else _DOT_COLOR
        alpha = 0.55 if c["explored_only"] else 1.0
        ax.scatter(
            [float(x_pc_to_px(c["pc1"]))],
            [float(y_pc_to_px(c["pc2"]))],
            s=(r * 2) ** 2 * 0.9,
            c=color, alpha=alpha,
            edgecolors="none", zorder=4,
        )

    # Annotation ovals (user-drawn), with labels.
    for o in _resolve_ovals(payload):
        ellipse = mpatches.Ellipse(
            (o["cx"], o["cy"]),
            width=2 * o["rx"], height=2 * o["ry"],
            facecolor=_OVAL_FILL, alpha=0.10,
            edgecolor="none", zorder=5,
        )
        ax.add_patch(ellipse)
        if o["label"]:
            ax.text(o["cx"], o["cy"], _wrap_two_lines(o["label"]),
                    ha="center", va="center",
                    color=_CLUSTER_TEXT,
                    fontsize=24 * font_scale, fontweight=700,
                    family="Helvetica", zorder=7)

    # Merge leader lines: tie each merged label to the centroids of its
    # component clusters so the regrouping reads at a glance. Draw before
    # labels so bold text sits on top.
    if show_cluster_labels:
        clusters_for_k = payload["clusterings"][str(int(k))]["clusters"]
        centroid_by_id = {c["id"]: c for c in clusters_for_k}
        for m in _resolve_cluster_merges(payload, k):
            if not m["visible"]:
                continue
            for cid in m["merged_cluster_ids"]:
                c = centroid_by_id.get(cid)
                if c is None:
                    continue
                ax.plot(
                    [m["label_x"], float(x_pc_to_px(c["center_pc1"]))],
                    [m["label_y"], float(y_pc_to_px(c["center_pc2"]))],
                    color="#8ab4d8", linewidth=2.0, alpha=0.75,
                    linestyle=(0, (6, 4)), zorder=6,
                )

    # Cluster labels (bold text at centroid, carrying saved renames). Wrap the
    # name onto two lines when long and push the "· mass" suffix onto its own
    # line so the posterior-mass tag never collides with the label. Font size
    # encodes posterior-weighted mean |coef|; use the shared `coef_abs_ref`
    # when provided so sizes stay comparable across axes in the aggregate.
    if show_cluster_labels:
        cluster_items = _resolve_all_cluster_labels(payload, k)
        if coef_abs_ref is None and cluster_items:
            ref_local = max((ci["coef_abs_contrib"] for ci in cluster_items),
                            default=0.0)
        else:
            ref_local = coef_abs_ref or 0.0
        for ci in cluster_items:
            if not ci["visible"]:
                continue
            wrapped = _wrap_two_lines(ci["label"])
            text = f"{wrapped}\n{ci['posterior_mass']:.2f}"
            fs = _cluster_label_fontsize(ci["coef_abs_contrib"], ref_local)
            ax.text(ci["label_x"], ci["label_y"], text,
                    ha="center", va="center",
                    color=_CLUSTER_TEXT,
                    fontsize=fs * font_scale, fontweight=700,
                    family="Helvetica", zorder=7)

    # Per-concept labels + leader lines (only shown ones).
    if show_concept_labels:
        for ci in _resolve_concept_labels(payload):
            if not ci["visible"]:
                continue
            ax.plot([ci["anchor_x"], ci["label_x"]],
                    [ci["anchor_y"], ci["label_y"]],
                    color="#8a8a8a", linewidth=0.75, alpha=0.5, zorder=4)
            ax.text(ci["label_x"], ci["label_y"],
                    _wrap_two_lines(_short_label(ci["concept"])),
                    ha="center", va="center",
                    color="#2b2b2b",
                    fontsize=15 * font_scale, fontweight=700,
                    family="Helvetica", zorder=6)

    if show_title:
        # Draw inside the Axes box so bbox doesn't clip in the aggregate
        # grid (set_title lives outside ylim). Match cluster-label font size
        # so axis titles have the same visual weight as the cluster callouts.
        title = _AXIS_DISPLAY.get(payload["axis"], payload["axis"])
        ax.text(x_offset, -_MARGIN["top"] + 20,
                title,
                ha="left", va="top",
                fontsize=24 * font_scale, fontweight=700,
                family="Helvetica", color="#1a1a1a", zorder=8)


def render_legend_to_ax(ax, explored_threshold: float,
                        font_scale: float = 1.0,
                        coef_abs_ref: float | None = None) -> None:
    """Render the standalone legend panel used in the 2x3 aggregate.

    When `coef_abs_ref` is provided, append a size-scale swatch that shows
    small / medium / large cluster labels at the fonts they will take at
    |coef|=0, ref/2, and ref.
    """
    import matplotlib.patches as mpatches

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    y = 0.92
    lh = 0.085  # line height

    def section_title(txt):
        nonlocal y
        ax.text(0.03, y, txt, ha="left", va="center",
                fontsize=13 * font_scale, fontweight=700,
                family="Helvetica", color="#333")
        y -= lh

    section_title("Posterior HPD")
    for lvl in _LEVEL_ORDER:
        ax.add_patch(mpatches.Rectangle(
            (0.08, y - 0.025), 0.08, 0.05,
            facecolor=_HPD_FILLS[lvl], alpha=_HPD_ALPHAS[lvl],
            edgecolor="none"))
        ax.text(0.20, y, f"{int(lvl * 100)}% HPD",
                ha="left", va="center",
                fontsize=12 * font_scale, family="Helvetica", color="#333")
        y -= lh * 0.9

    y -= lh * 0.3
    section_title("Dots")
    ax.add_patch(mpatches.Circle((0.12, y), 0.025,
                                 facecolor=_DOT_COLOR,
                                 edgecolor="none"))
    ax.text(0.20, y, "concept", ha="left", va="center",
            fontsize=12 * font_scale, family="Helvetica", color="#333")
    y -= lh * 0.9
    ax.add_patch(mpatches.Circle((0.12, y), 0.025,
                                 facecolor=_DOT_COLOR, alpha=0.5,
                                 edgecolor="none"))
    ax.text(0.20, y, "jittered replicate\n(density ∝ posterior)",
            ha="left", va="center",
            fontsize=12 * font_scale, family="Helvetica", color="#333")
    y -= lh * 1.3
    ax.add_patch(mpatches.Circle((0.12, y), 0.018,
                                 facecolor=_EXPLORED_COLOR, alpha=0.55,
                                 edgecolor="none"))
    ax.text(0.20, y,
            f"explored (< {explored_threshold:.2f})",
            ha="left", va="center",
            fontsize=12 * font_scale, family="Helvetica", color="#333")

    if coef_abs_ref and coef_abs_ref > 0:
        y -= lh * 1.6
        section_title("Cluster label size")
        ax.text(0.03, y,
                "font size ∝ Σ posterior_freq × |coef|",
                ha="left", va="center",
                fontsize=11 * font_scale, family="Helvetica", color="#555")
        y -= lh * 0.9
        samples = [
            (0.0, "0"),
            (0.5 * coef_abs_ref, f"{0.5 * coef_abs_ref:.2f}"),
            (coef_abs_ref, f"≥ {coef_abs_ref:.2f}"),
        ]
        x_cursor = 0.08
        for coef_val, lbl in samples:
            fs = _cluster_label_fontsize(coef_val, coef_abs_ref) * font_scale
            ax.text(x_cursor, y, "Aa",
                    ha="left", va="center",
                    fontsize=fs, fontweight=700,
                    family="Helvetica", color=_CLUSTER_TEXT)
            ax.text(x_cursor, y - 0.05, lbl,
                    ha="left", va="top",
                    fontsize=10 * font_scale, family="Helvetica",
                    color="#555")
            x_cursor += 0.30


def _write_matplotlib_image(payload: dict, out_path: str,
                            *, k: int | None = None,
                            dpi: int = 100) -> None:
    """Save a single-axis static panel at exactly _CANVAS_W x _CANVAS_H px.

    DPI is fixed so the figure dimensions equal the HTML <svg> dimensions
    1:1 (1400x900 at dpi=100 → 14in x 9in). The Axes fills the canvas and
    bbox_inches="tight" is NOT used, so coordinates in the exported file
    land at the same pixel offsets as in the HTML.
    """
    import matplotlib.pyplot as plt
    fig_w_in = _CANVAS_W / dpi
    fig_h_in = _CANVAS_H / dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    render_payload_to_ax(ax, payload, k=k)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".png":
        fig.savefig(out_path, dpi=dpi, facecolor="white")
    else:
        fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-history-file", type=str, required=True)
    p.add_argument("--out-html", type=str, required=True)
    p.add_argument("--out-image", type=str, default=None,
                   help="Optional static export (extension picks format: "
                        ".svg/.pdf/.png). Uses saved_positions when available.")
    p.add_argument("--image-k", type=int, default=None,
                   help="K to render in --out-image. Defaults to the saved "
                        "current_k if present, else default_k.")
    p.add_argument("--axis-name", type=str, required=True,
                   help="Rating axis label shown in the HTML title.")
    p.add_argument("--num-posterior-iters", type=int, default=100,
                   help="Trailing MH iterations to average over.")
    p.add_argument("--provider-identity-coef-offset", type=int, default=0,
                   help="Number of leading coefficients that encode provider "
                        "identity (force-keep columns) rather than concepts.")
    p.add_argument("--embedding-model", type=str,
                   default="all-MiniLM-L6-v2")
    p.add_argument("--explored-threshold", type=float, default=0.05,
                   help="Concepts with posterior_freq below this are drawn as "
                        "small gray 'explored but rarely kept' markers.")
    p.add_argument("--k-min", type=int, default=2,
                   help="Smallest K to precompute for the HTML K slider.")
    p.add_argument("--k-max", type=int, default=10,
                   help="Largest K to precompute for the HTML K slider.")
    p.add_argument("--default-k", type=int, default=4,
                   help="K selected on initial page load.")
    p.add_argument("--cluster-method", type=str, default="agglomerative",
                   choices=["agglomerative", "gmm"],
                   help="'agglomerative' clusters in full embedding space + "
                        "per-cluster KDE contours. 'gmm' fits a joint Gaussian "
                        "mixture in PC space; HPD regions are closed-form "
                        "ellipses from each component's covariance.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (used by GMM EM init + jitter).")
    p.add_argument("--gmm-eps-jitter", type=float, default=0.02,
                   help="Std of Gaussian noise (in PC units) added to each "
                        "replicated point before GMM fit. Larger -> smoother "
                        "density + wider ellipses; 0 -> pure replication.")
    args = p.parse_args()

    saved_positions = None
    positions_path = os.path.join(
        os.path.dirname(os.path.abspath(args.out_html)) or ".",
        "posterior_concepts_positions.json",
    )
    if os.path.exists(positions_path):
        try:
            with open(positions_path, "r", encoding="utf-8") as f:
                saved_positions = json.load(f)
            print(f"loaded saved positions from {positions_path}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: could not load {positions_path}: {e}")

    history = TrainingHistory.load(args.training_history_file)
    payload = _build_payload(
        history,
        num_posterior_iters=args.num_posterior_iters,
        provider_identity_coef_offset=args.provider_identity_coef_offset,
        embedding_model=args.embedding_model,
        explored_threshold=args.explored_threshold,
        k_min=args.k_min,
        k_max=args.k_max,
        default_k=args.default_k,
        cluster_method=args.cluster_method,
        seed=args.seed,
        gmm_eps_jitter=args.gmm_eps_jitter,
        axis_name=args.axis_name,
        saved_positions=saved_positions,
    )

    os.makedirs(os.path.dirname(args.out_html) or ".", exist_ok=True)
    html = _D3_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)
    n_explored = sum(1 for c in payload["concepts"] if c["explored_only"])
    print(f"wrote {args.out_html}  ({len(payload['concepts'])} concepts, "
          f"K ∈ [{payload['k_min']},{payload['k_max']}] precomputed, "
          f"default K={payload['default_k']}, {n_explored} explored-only)")

    if args.out_image:
        _write_matplotlib_image(payload, args.out_image, k=args.image_k)
        print(f"wrote {args.out_image}")


if __name__ == "__main__":
    main()
