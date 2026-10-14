"""
Generate a standalone D3 HTML visualization of BC-LLM's posterior concepts.

Loads a TrainingHistory pickle, embeds each concept's text, runs PCA, and
writes an interactive HTML file where:
  - each unique concept is a dot anchored at its (PC1, PC2) position
  - dot color = posterior mean coef (RdBu_r), size = posterior_freq
  - every concept gets a draggable label; labels start on a short ring
    around their dot and stay wherever the user drops them (no sim)
  - users can draw annotation ovals around clusters, and download a
    `posterior_concepts_positions.json` capturing label positions +
    ovals. If that JSON sits next to the HTML at build time, those
    positions and ovals are restored on the next render.

One HTML file per invocation; see exp_cbm/sconscript for how this is
wired into the pipeline.
"""
import argparse
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


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
    --oval-tint: #4a6b8a;
    --cluster-text: #2b3a4a;
    --page-text: #1a1a1a;
    --muted-text: #4a4a4a;
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
                           margin-top: 4px; border: 0.5px solid rgba(0,0,0,0.3); }
  .side-col li label { cursor: pointer; flex: 1 1 auto; }
  .side-col li.is-hidden label { color: #999; text-decoration: line-through; }
  .side-col li.hidden-by-search { display: none; }
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
  button { font-size: 12px; padding: 5px 10px; margin-right: 6px; cursor: pointer;
           background: #f4f4f4; border: 1px solid #c0c0c0; border-radius: 3px;
           font-family: inherit; }
  button:hover { background: #e8e8e8; }
  button.active { background: #ffe08a; border-color: #d4a017; }
  /* interactive-only chrome; stripped from PDF export clone */
  .dot    { cursor: pointer; }
  .label  { cursor: move; user-select: none; }
  .label.hidden { opacity: 0; pointer-events: none; }
  .leader.hidden { opacity: 0; }
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
  <button onclick="resetPositions()">reset label positions</button>
  <button onclick="showAllLabels()">show all labels</button>
  <button onclick="hideAllLabels()">hide all labels</button>
  <button onclick="exportPositions()">download positions JSON</button>
  <button onclick="exportPdf()">download PDF</button>
  <button id="annoBtn" onclick="toggleAnnoMode()">draw annotation oval</button>
  <button onclick="clearOvals()">clear ovals</button>
  <span id="annoHint" style="font-size:12px;color:#666;margin-left:8px;"></span>
</div>
<div class="layout">
  <div class="plot-col">
    <svg id="chart" width="1400" height="900" font-family="Helvetica"></svg>
  </div>
  <div class="side-col">
    <h3>Concept labels</h3>
    <div class="side-sub">Click a label on the plot to hide it. Check a box here to show it again.</div>
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

document.getElementById("title").textContent =
  `BC-LLM posterior concepts — ${payload.axis} (${payload.identity_label})`;
document.getElementById("subtitle").textContent =
  `${payload.nodes.length} concepts · dot size ∝ posterior frequency · ` +
  `color = posterior mean coefficient (red = positive, blue = negative) · ` +
  `PCA on sentence embeddings (PC1 ${(payload.pc1_var*100).toFixed(1)}%, ` +
  `PC2 ${(payload.pc2_var*100).toFixed(1)}%)`;

const svg = d3.select("#chart");
const width = +svg.attr("width");
const height = +svg.attr("height");
const margin = {top: 30, right: 40, bottom: 30, left: 40};
const innerW = width - margin.left - margin.right;
const innerH = height - margin.top - margin.bottom;
const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

const nodes = payload.nodes;
const pc1Ext = d3.extent(nodes, d => d.pc1);
const pc2Ext = d3.extent(nodes, d => d.pc2);
const padFrac = 0.08;
const xScale = d3.scaleLinear()
  .domain([pc1Ext[0] - padFrac*(pc1Ext[1]-pc1Ext[0]),
           pc1Ext[1] + padFrac*(pc1Ext[1]-pc1Ext[0])])
  .range([0, innerW]);
const yScale = d3.scaleLinear()
  .domain([pc2Ext[0] - padFrac*(pc2Ext[1]-pc2Ext[0]),
           pc2Ext[1] + padFrac*(pc2Ext[1]-pc2Ext[0])])
  .range([innerH, 0]);

// Deterministic initial label placement: spread labels on a short ring
// around their dot so none start exactly on top of the dot. No sim.
const RING_RADIUS = 60;
function initialOffset(i, n) {
  const theta = 2 * Math.PI * i / Math.max(n, 1);
  return [RING_RADIUS * Math.cos(theta), -RING_RADIUS * Math.sin(theta)];
}

// Saved positions may be a flat list [{concept,labelX,labelY}]
// or an object {labels: [...], ovals: [...]}.  Legacy exports used
// "circles" with a single radius `r` — migrate to {rx, ry}.
const savedRaw = payload.saved_positions;
let savedLabelList = null;
let savedOvalList = [];
if (savedRaw) {
  if (Array.isArray(savedRaw)) {
    savedLabelList = savedRaw;
  } else {
    savedLabelList = savedRaw.labels || null;
    if (savedRaw.ovals) {
      savedOvalList = savedRaw.ovals;
    } else if (savedRaw.circles) {
      savedOvalList = savedRaw.circles.map(c => ({
        cx: c.cx, cy: c.cy, rx: c.r, ry: c.r, label: c.label || "",
      }));
    }
  }
}
const savedByConcept = {};
if (savedLabelList) {
  savedLabelList.forEach(s => { savedByConcept[s.concept] = s; });
}

// Strip shared boilerplate from the concept text so on-plot labels stay
// compact. Keeps the raw concept text available for the tooltip.
function shortLabel(s) {
  let t = s.trim();
  t = t.replace(/^Does Answer B\s+/i, "");
  t = t.replace(/^Is Answer B\s+/i, "");
  t = t.replace(/\s+than Answer A\??\s*$/i, "");
  // Capitalize first letter now that the sentence prefix is gone.
  if (t.length) t = t.charAt(0).toUpperCase() + t.slice(1);
  return t;
}

// Darker variant of a hex color (for dot stroke derived from fill).
function darken(hex, k) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3
    ? h.split("").map(c => c + c).join("")
    : h, 16);
  let r = (n >> 16) & 0xff, gC = (n >> 8) & 0xff, b = n & 0xff;
  r = Math.max(0, Math.round(r * (1 - k)));
  gC = Math.max(0, Math.round(gC * (1 - k)));
  b = Math.max(0, Math.round(b * (1 - k)));
  return "#" + [r, gC, b].map(v => v.toString(16).padStart(2, "0")).join("");
}

const items = nodes.map((d, i) => {
  const [dx, dy] = initialOffset(i, nodes.length);
  const anchorX = xScale(d.pc1);
  const anchorY = yScale(d.pc2);
  const saved = savedByConcept[d.concept];
  return {
    ref: d,
    anchorX, anchorY,
    labelX: saved ? saved.labelX : anchorX + dx,
    labelY: saved ? saved.labelY : anchorY + dy,
    hidden: saved && saved.hidden ? true : false,
    r: 5 + 11 * d.posterior_freq,
    initialDx: dx, initialDy: dy,
  };
});

// Ovals go underneath everything else as a tinted-region layer.
const ovalLayer = g.append("g").attr("class", "anno-layer");

const leaders = g.append("g").selectAll("line")
  .data(items).join("line")
  .attr("class", "leader")
  .attr("stroke", "#8a8a8a")
  .attr("stroke-width", 0.75)
  .attr("stroke-opacity", 0.5)
  .attr("x1", d => d.anchorX).attr("y1", d => d.anchorY)
  .attr("x2", d => d.labelX).attr("y2", d => d.labelY)
  .classed("hidden", d => d.hidden);

const dots = g.append("g").selectAll("circle")
  .data(items).join("circle")
  .attr("class", "dot")
  .attr("r", d => d.r)
  .attr("cx", d => d.anchorX).attr("cy", d => d.anchorY)
  .attr("fill", d => d.ref.color)
  .attr("stroke", d => darken(d.ref.color, 0.35))
  .attr("stroke-width", 0.75);

const tooltip = d3.select("#tooltip");
dots
  .on("mouseover", (event, d) => {
    tooltip.style("opacity", 1)
      .html(`<b>${d.ref.concept}</b><br>` +
            `mean coef: ${d.ref.mean_coef.toFixed(3)}<br>` +
            `posterior_freq: ${d.ref.posterior_freq.toFixed(2)}`);
  })
  .on("mousemove", (event) => {
    tooltip.style("left", (event.pageX + 12) + "px")
           .style("top",  (event.pageY + 12) + "px");
  })
  .on("mouseout", () => tooltip.style("opacity", 0));

const LABEL_FONT_SIZE = 22;
const CLUSTER_FONT_SIZE = 34;
const FONT_FAMILY = "Helvetica";

const labels = g.append("g").selectAll("text")
  .data(items).join("text")
  .attr("class", "label")
  .attr("fill", d => darken(d.ref.color, 0.4))
  .attr("font-size", LABEL_FONT_SIZE)
  .attr("font-weight", 700)
  .attr("font-family", FONT_FAMILY)
  .attr("text-anchor", "middle")
  .attr("dominant-baseline", "middle")
  .attr("x", d => d.labelX)
  .attr("y", d => d.labelY)
  .text(d => shortLabel(d.ref.concept))
  .classed("hidden", d => d.hidden)
  .on("click", function(event, d) {
    // bare click (no drag) hides this label
    if (event.defaultPrevented) return;
    setLabelHidden(d, true);
  })
  .call(d3.drag()
    .on("start", function(event) { event.sourceEvent.stopPropagation(); })
    .on("drag", function(event, d) {
      d.labelX = Math.max(0, Math.min(innerW, event.x));
      d.labelY = Math.max(0, Math.min(innerH, event.y));
      d3.select(this).attr("x", d.labelX).attr("y", d.labelY);
      leaders.filter(l => l === d)
        .attr("x2", d.labelX).attr("y2", d.labelY);
    }));

const _rowByItem = new Map();  // item -> {li, cb}
function setLabelHidden(item, hidden) {
  item.hidden = hidden;
  labels.filter(l => l === item).classed("hidden", hidden);
  leaders.filter(l => l === item).classed("hidden", hidden);
  const row = _rowByItem.get(item);
  if (row) {
    row.cb.checked = !hidden;
    row.li.classList.toggle("is-hidden", hidden);
  }
}

function showAllLabels() { items.forEach(d => setLabelHidden(d, false)); }
function hideAllLabels() { items.forEach(d => setLabelHidden(d, true)); }

// Sidebar list: colored dot + checkbox + concept text per item.
const listEl = document.getElementById("conceptList");
items.forEach(d => {
  const li = document.createElement("li");
  if (d.hidden) li.classList.add("is-hidden");
  const id = "cb_" + Math.random().toString(36).slice(2, 10);
  const dotMark = document.createElement("span");
  dotMark.className = "dot-mark";
  dotMark.style.background = d.ref.color;
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.id = id;
  cb.checked = !d.hidden;
  cb.addEventListener("change", () => setLabelHidden(d, !cb.checked));
  const lbl = document.createElement("label");
  lbl.htmlFor = id;
  lbl.textContent = shortLabel(d.ref.concept);
  lbl.title = d.ref.concept;  // full text on hover
  li.appendChild(dotMark);
  li.appendChild(cb);
  li.appendChild(lbl);
  listEl.appendChild(li);
  _rowByItem.set(d, {li, cb});
});

// Search box filter.
document.getElementById("conceptSearch").addEventListener("input", (ev) => {
  const q = ev.target.value.toLowerCase().trim();
  _rowByItem.forEach((row, item) => {
    const hay = (item.ref.concept + " " + shortLabel(item.ref.concept)).toLowerCase();
    const match = !q || hay.includes(q);
    row.li.classList.toggle("hidden-by-search", !match);
  });
});

function resetPositions() {
  items.forEach(d => {
    d.labelX = d.anchorX + d.initialDx;
    d.labelY = d.anchorY + d.initialDy;
  });
  labels.attr("x", d => d.labelX).attr("y", d => d.labelY);
  leaders.attr("x2", d => d.labelX).attr("y2", d => d.labelY);
}

function exportPositions() {
  const out = {
    labels: items.map(d => ({
      concept: d.ref.concept,
      labelX: Math.round(d.labelX),
      labelY: Math.round(d.labelY),
      hidden: !!d.hidden,
    })),
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
  a.href = url;
  a.download = "posterior_concepts_positions.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
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
  const pdf = new jsPDF({
    orientation: w >= h ? "landscape" : "portrait",
    unit: "pt",
    format: [w, h],
  });
  // svg2pdf reads SVG presentation attributes, not CSS classes. We've put
  // every styling that matters as attributes on the live elements, so a
  // simple clone works. We only need to strip interactive-only chrome:
  // delete badges, resize handles, and any .hidden-marked labels/leaders.
  const clone = svgNode.cloneNode(true);
  clone.querySelectorAll(
    ".anno-delete-bg, .anno-delete-x, .anno-handle, .hidden"
  ).forEach(n => n.remove());
  await window.svg2pdf.svg2pdf(clone, pdf, {x: 0, y: 0, width: w, height: h});
  pdf.save("posterior_concepts.pdf");
}

// --- annotation ovals ---------------------------------------------------
// (ovalLayer was created earlier so it sits under dots/labels)
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
    hint.textContent = "click-drag on the plot to draw an oval · drag the corner handle to resize · dblclick to delete · shift-click to rename";
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
    .attr("fill", OVAL_FILL)
    .attr("fill-opacity", 0.1)
    .attr("stroke", "none");
  enter.append("circle").attr("class", "anno-handle")
    .attr("r", 5)
    .attr("fill", "#d4a017")
    .attr("stroke", "#ffffff")
    .attr("stroke-width", 1);
  enter.append("circle").attr("class", "anno-delete-bg")
    .attr("r", 8)
    .attr("fill", "#c0392b")
    .attr("stroke", "#ffffff")
    .attr("stroke-width", 1);
  enter.append("text").attr("class", "anno-delete-x")
    .attr("fill", "#ffffff")
    .attr("font-size", 12)
    .attr("font-weight", 700)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .text("×");
  enter.append("text").attr("class", "anno-label")
    .attr("fill", CLUSTER_TEXT)
    .attr("font-size", CLUSTER_FONT_SIZE)
    .attr("font-weight", 700)
    .attr("font-family", FONT_FAMILY)
    .attr("text-anchor", "middle")
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
    .attr("cx", d => d.cx + d.rx)
    .attr("cy", d => d.cy + d.ry)
    .on("mousedown", handleMouseDown);
  const deleteOval = (event, d) => {
    event.stopPropagation();
    event.preventDefault();
    ovals = ovals.filter(o => o !== d);
    if (selectedOval === d) selectedOval = null;
    renderOvals();
  };
  merged.select("circle.anno-delete-bg")
    .attr("cx", d => d.cx + d.rx)
    .attr("cy", d => d.cy - d.ry)
    .on("mousedown", deleteOval)
    .on("mouseenter", function() {
      this.parentNode.classList.add("show-delete");
    })
    .on("mouseleave", function() {
      this.parentNode.classList.remove("show-delete");
    });
  merged.select("text.anno-delete-x")
    .attr("x", d => d.cx + d.rx)
    .attr("y", d => d.cy - d.ry);
  merged.select("text.anno-label")
    .attr("x", d => d.cx)
    .attr("y", d => d.cy)
    .text(d => d.label || "")
    .on("mousedown", (event, d) => {
      event.stopPropagation();
      const name = prompt("Cluster label:", d.label || "");
      if (name !== null) { d.label = name; renderOvals(); }
    })
    .on("mouseenter", function() {
      this.parentNode.classList.add("show-delete");
    })
    .on("mouseleave", function() {
      this.parentNode.classList.remove("show-delete");
    });
}

function ovalMouseDown(event, d) {
  event.stopPropagation();
  if (event.shiftKey) {
    const name = prompt("Cluster label:", d.label || "");
    if (name !== null) { d.label = name; renderOvals(); }
    return;
  }
  selectedOval = d;
  const [x0, y0] = d3.pointer(event, g.node());
  const cx0 = d.cx, cy0 = d.cy;
  function onMove(ev) {
    const [x, y] = d3.pointer(ev, g.node());
    d.cx = cx0 + (x - x0);
    d.cy = cy0 + (y - y0);
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

// start-drag on empty canvas (anno mode only) creates a new oval
svg.on("mousedown", function(event) {
  if (!annoMode) return;
  if (event.target.closest(".anno")) return;
  const [x0, y0] = d3.pointer(event, g.node());
  const o = {_id: Date.now() + Math.random(),
             cx: x0, cy: y0, rx: 1, ry: 1, label: ""};
  ovals.push(o);
  selectedOval = o;
  renderOvals();
  function onMove(ev) {
    const [mx, my] = d3.pointer(ev, g.node());
    // treat the initial point as one corner of the bounding box, so the
    // oval tracks a free-form drag rather than growing symmetrically.
    const cx = (x0 + mx) / 2;
    const cy = (y0 + my) / 2;
    o.cx = cx; o.cy = cy;
    o.rx = Math.max(1, Math.abs(mx - cx));
    o.ry = Math.max(1, Math.abs(my - cy));
    renderOvals();
  }
  function onUp() {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    if (o.rx < 8 || o.ry < 8) {   // too small -> discard
      ovals = ovals.filter(x => x !== o);
      selectedOval = null;
      renderOvals();
      return;
    }
    const name = prompt("Cluster label (optional):", "");
    if (name !== null) o.label = name;
    renderOvals();
  }
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
});

// hydrate saved ovals
if (savedOvalList && savedOvalList.length) {
  ovals = savedOvalList.map(o => ({
    _id: Date.now() + Math.random(),
    cx: o.cx, cy: o.cy,
    rx: o.rx != null ? o.rx : o.r,
    ry: o.ry != null ? o.ry : o.r,
    label: o.label || "",
  }));
  renderOvals();
}
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
    """Per-concept posterior summary over the trailing posterior window."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-history-file", type=str, required=True)
    p.add_argument("--out-html", type=str, required=True)
    p.add_argument("--axis-name", type=str, required=True,
                   help="Rating axis label shown in the HTML title.")
    p.add_argument("--identity-label", type=str, default="identity unknown",
                   help="Shown in the HTML title alongside the axis.")
    p.add_argument("--num-posterior-iters", type=int, default=100,
                   help="Trailing MH iterations to average over.")
    p.add_argument("--provider-identity-coef-offset", type=int, default=0,
                   help="Number of leading coefficients that encode provider "
                        "identity (force-keep columns) rather than concepts.")
    p.add_argument("--embedding-model", type=str,
                   default="all-MiniLM-L6-v2")
    args = p.parse_args()

    history = TrainingHistory.load(args.training_history_file)
    long_df = _load_iter_rows(history, args.provider_identity_coef_offset)
    if long_df.empty:
        raise RuntimeError(
            f"No rows loaded from {args.training_history_file}")

    summary = _summarize_concepts(long_df, args.num_posterior_iters)
    if len(summary) < 2:
        raise RuntimeError(
            f"Need >=2 unique concepts in the posterior window; got "
            f"{len(summary)}")

    embedder = SentenceTransformer(args.embedding_model, local_files_only=True)
    emb = embedder.encode(summary["concept"].tolist(), show_progress_bar=False)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(emb)
    summary["pc1"] = pcs[:, 0]
    summary["pc2"] = pcs[:, 1]

    vmax = float(np.abs(summary["mean_coef"]).max()) or 1.0
    cmap = mpl.colormaps["RdBu_r"]
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    nodes = []
    for i, row in summary.iterrows():
        nodes.append({
            "id": int(i),
            "concept": row["concept"],
            "pc1": float(row["pc1"]),
            "pc2": float(row["pc2"]),
            "mean_coef": float(row["mean_coef"]),
            "posterior_freq": float(row["posterior_freq"]),
            "color": to_hex(cmap(norm(row["mean_coef"]))),
        })

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

    payload = {
        "nodes": nodes,
        "vmax": vmax,
        "axis": args.axis_name,
        "identity_label": args.identity_label,
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "saved_positions": saved_positions,
    }

    os.makedirs(os.path.dirname(args.out_html) or ".", exist_ok=True)
    html = _D3_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out_html}  ({len(nodes)} concepts)")


if __name__ == "__main__":
    main()
