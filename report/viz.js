/* Minimal SVG chart kit for the report. Specs follow the dataviz method: thin marks, 4px rounded data-ends,
   2px lines, >=8px markers with a 2px surface ring, hairline solid grid, legend for >=2 series, selective
   direct labels, hover tooltips that never gate (every chart has a table twin). Text never wears series color. */
const NS = "http://www.w3.org/2000/svg";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const pct = (x, d = 0) => (x == null || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(d)}%`);
const fix = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));

const tip = document.createElement("div");
tip.className = "viz-tip";
tip.hidden = true;
document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tip));
function showTip(evt, rows) {
  tip.replaceChildren();
  for (const [value, label, color] of rows) {
    const r = document.createElement("div");
    r.className = "viz-tip-row";
    if (color) {
      const k = document.createElement("span");
      k.className = "viz-tip-key";
      k.style.background = color;
      r.appendChild(k);
    }
    const v = document.createElement("strong");
    v.textContent = value;
    const l = document.createElement("span");
    l.textContent = label;
    r.append(v, l);
    tip.appendChild(r);
  }
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => (tip.hidden = true);
function hover(node, rowsFn) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", (e) => showTip(e, rowsFn()));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => {
    const b = node.getBoundingClientRect();
    showTip({ clientX: b.right, clientY: b.top }, rowsFn());
  });
  node.addEventListener("blur", hideTip);
}

/* Each chart mounts into <figure class="viz" data-chart="..."> and gets a table twin. */
function frame(fig, height, margin) {
  const box = fig.querySelector(".viz-plot");
  box.replaceChildren();
  const W = Math.max(320, box.clientWidth || 640);
  const svg = el("svg", { viewBox: `0 0 ${W} ${height}`, width: "100%", role: "img",
    "aria-label": fig.querySelector("figcaption")?.textContent?.trim() || "chart" }, box);
  return { svg, W, H: height, m: margin, iw: W - margin.l - margin.r, ih: height - margin.t - margin.b };
}
function tableTwin(fig, head, rows) {
  const host = fig.querySelector(".viz-table");
  if (!host) return;
  host.replaceChildren();
  const t = document.createElement("table");
  const tr = t.createTHead().insertRow();
  for (const h of head) { const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); }
  const tb = t.createTBody();
  for (const r of rows) { const row = tb.insertRow(); for (const c of r) row.insertCell().textContent = c; }
  host.appendChild(t);
}
function roundedBar(svg, x0, y, w, h, fill, r = 4) {
  // square at the baseline (x0), rounded at the data end
  r = Math.min(r, h / 2, Math.max(0, w));
  const d = w <= 0 ? "" : `M${x0},${y} H${x0 + w - r} Q${x0 + w},${y} ${x0 + w},${y + r} V${y + h - r} Q${x0 + w},${y + h} ${x0 + w - r},${y + h} H${x0} Z`;
  return el("path", { d, fill }, svg);
}
function xAxis(g, sx, ticks, y0, ih, fmt, label, W, m) {
  for (const t of ticks) {
    const x = sx(t);
    el("line", { x1: x, x2: x, y1: y0, y2: y0 + ih, stroke: css("--grid"), "stroke-width": 1 }, g);
    const tx = el("text", { x, y: y0 + ih + 16, "text-anchor": "middle", class: "viz-tick" }, g);
    tx.textContent = fmt(t);
  }
  if (label) {
    const tl = el("text", { x: m.l + (W - m.l - m.r) / 2, y: y0 + ih + 34, "text-anchor": "middle", class: "viz-axis-label" }, g);
    tl.textContent = label;
  }
}

/* Horizontal bars with 95% CI whiskers. rows: {label, p, lo, hi, color?, note?} */
function barsCI(fig, rows, { max = 1, fmt = pct, axis = "", color = css("--series-1"), sub } = {}) {
  const rowH = 30, m = { t: 8, r: 64, b: axis ? 44 : 28, l: Math.min(190, 12 + 7.2 * Math.max(...rows.map((r) => r.label.length))) };
  const f = frame(fig, m.t + m.b + rows.length * rowH, m);
  const sx = (v) => m.l + (v / max) * f.iw;
  xAxis(f.svg, sx, [0, 0.25, 0.5, 0.75, 1].map((t) => t * max), m.t, rows.length * rowH, fmt, axis, f.W, m);
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + 7, h = 14;
    const lab = el("text", { x: m.l - 10, y: y + h / 2 + 4, "text-anchor": "end", class: "viz-label" }, f.svg);
    lab.textContent = r.label;
    const bar = roundedBar(f.svg, m.l, y, sx(r.p) - m.l, h, r.color || color);
    if (r.lo != null) {
      el("line", { x1: sx(r.lo), x2: sx(r.hi), y1: y + h / 2, y2: y + h / 2, stroke: css("--ink-2"), "stroke-width": 1.5 }, f.svg);
      for (const v of [r.lo, r.hi]) el("line", { x1: sx(v), x2: sx(v), y1: y + 3, y2: y + h - 3, stroke: css("--ink-2"), "stroke-width": 1.5 }, f.svg);
    }
    const vt = el("text", { x: sx(r.hi ?? r.p) + 8, y: y + h / 2 + 4, class: "viz-value" }, f.svg);
    vt.textContent = fmt(r.p);
    const hit = el("rect", { x: 0, y: y - 7, width: f.W, height: rowH, fill: "transparent" }, f.svg);
    hover(hit, () => [[fmt(r.p), r.label, r.color || color], ...(r.lo != null ? [[`${fmt(r.lo)} – ${fmt(r.hi)}`, "95% CI"]] : []), ...(r.note ? [[r.note, ""]] : [])]);
  });
  tableTwin(fig, ["", sub || "value", "95% CI"], rows.map((r) => [r.label, fmt(r.p), r.lo != null ? `${fmt(r.lo)} – ${fmt(r.hi)}` : ""]));
}

/* Multi-series line chart. series: {name, color, points: [{x, y, lo?, hi?}]} */
function lines(fig, series, { xFmt = (x) => x, yFmt = pct, yMin = 0, yMax = 1, xLabel = "", yLabel = "", refs = [], logX = false, xTicks, rotateX = false, endLabels = true } = {}) {
  const m = { t: 14, r: endLabels ? 120 : 40, b: rotateX ? 96 : 46, l: rotateX ? 104 : 50 };
  const f = frame(fig, 300, m);
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
  const tx = (x) => (logX ? Math.log10(x) : x);
  const x0 = tx(xs[0]), x1 = tx(xs[xs.length - 1]);
  const sx = (x) => m.l + ((tx(x) - x0) / (x1 - x0 || 1)) * f.iw;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * f.ih;
  const yt = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));
  for (const t of yt) {
    el("line", { x1: m.l, x2: m.l + f.iw, y1: sy(t), y2: sy(t), stroke: css("--grid"), "stroke-width": 1 }, f.svg);
    el("text", { x: m.l - 8, y: sy(t) + 4, "text-anchor": "end", class: "viz-tick" }, f.svg).textContent = yFmt(t);
  }
  for (const x of xTicks || xs) {
    const ty = m.t + f.ih + 16;
    el("text", rotateX ? { x: sx(x), y: ty, "text-anchor": "end", transform: `rotate(-35 ${sx(x)} ${ty})`, class: "viz-tick" }
      : { x: sx(x), y: ty + 2, "text-anchor": "middle", class: "viz-tick" }, f.svg).textContent = xFmt(x);
  }
  if (xLabel) el("text", { x: m.l + f.iw / 2, y: m.t + f.ih + 38, "text-anchor": "middle", class: "viz-axis-label" }, f.svg).textContent = xLabel;
  if (yLabel) el("text", { x: 12, y: m.t + f.ih / 2, transform: `rotate(-90 12 ${m.t + f.ih / 2})`, "text-anchor": "middle", class: "viz-axis-label" }, f.svg).textContent = yLabel;
  for (const r of refs) {
    el("line", { x1: m.l, x2: m.l + f.iw, y1: sy(r.y), y2: sy(r.y), stroke: css("--ink-2"), "stroke-width": 1 }, f.svg);
    el("text", { x: m.l + f.iw + 6, y: sy(r.y) + 4, class: "viz-note" }, f.svg).textContent = r.label;
  }
  for (const s of series) {
    const pts = s.points.filter((p) => p.y != null && !Number.isNaN(p.y));
    if (pts.some((p) => p.lo != null)) {
      const band = pts.map((p) => `${sx(p.x)},${sy(p.hi)}`).concat(pts.slice().reverse().map((p) => `${sx(p.x)},${sy(p.lo)}`));
      el("polygon", { points: band.join(" "), fill: s.color, "fill-opacity": 0.1 }, f.svg);
    }
    el("polyline", { points: pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, f.svg);
    for (const p of pts) el("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, f.svg);
  }
  // direct end labels (<=4 series), nudged apart only if they would overlap
  if (endLabels && series.length <= 4) {
    const ends = series.map((s) => { const p = s.points.filter((q) => q.y != null).at(-1); return { s, y: sy(p.y) }; }).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
    for (const e of ends) el("text", { x: m.l + f.iw + 8, y: e.y + 4, class: "viz-label" }, f.svg).textContent = e.s.name;
  }
  const cross = el("line", { y1: m.t, y2: m.t + f.ih, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, f.svg);
  const hit = el("rect", { x: m.l, y: m.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
  let cur = xs[0];
  hit.addEventListener("pointermove", (e) => {
    const b = f.svg.getBoundingClientRect(), px = ((e.clientX - b.left) / b.width) * f.W;
    cur = xs.reduce((a, x) => (Math.abs(sx(x) - px) < Math.abs(sx(a) - px) ? x : a), xs[0]);
    cross.setAttribute("x1", sx(cur)); cross.setAttribute("x2", sx(cur)); cross.setAttribute("visibility", "visible");
    showTip(e, [[xFmt(cur), xLabel || "x"], ...series.map((s) => { const p = s.points.find((q) => q.x === cur); return [p ? yFmt(p.y) : "–", s.name, s.color]; })]);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  legend(fig, series.map((s) => [s.name, s.color, "line"]));
  tableTwin(fig, [xLabel || "x", ...series.map((s) => s.name)], xs.map((x) => [xFmt(x), ...series.map((s) => { const p = s.points.find((q) => q.x === x); return p ? yFmt(p.y) : "–"; })]));
}

function legend(fig, items) {
  const host = fig.querySelector(".viz-legend");
  if (!host || items.length < 2) return;
  host.replaceChildren();
  for (const [name, color, kind] of items) {
    const it = document.createElement("span");
    it.className = "viz-legend-item";
    const k = document.createElement("span");
    k.className = kind === "line" ? "viz-key-line" : "viz-key-rect";
    k.style.background = color;
    const t = document.createElement("span");
    t.textContent = name;
    it.append(k, t);
    host.appendChild(it);
  }
}

/* Heatmap of rates on the sequential blue ramp. */
function heatmap(fig, rowNames, colNames, val, { fmt = pct, rowLabel = "", colLabel = "" } = {}) {
  const ramp = ["--seq-100", "--seq-200", "--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"].map(css);
  const m = { t: 112, r: 96, b: 12, l: 120 };
  const box = fig.querySelector(".viz-plot");
  const W0 = Math.max(560, box.clientWidth || 640);
  const cw = Math.max(26, Math.min(44, (W0 - m.l - m.r) / colNames.length));
  const ch = 26, W = m.l + m.r + cw * colNames.length, H = m.t + m.b + ch * rowNames.length;
  box.replaceChildren();
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: Math.min(W, W0), role: "img", "aria-label": fig.querySelector("figcaption")?.textContent?.trim() || "heatmap" }, box);
  colNames.forEach((c, j) => {
    const x = m.l + j * cw + cw / 2, t = el("text", { x, y: m.t - 8, transform: `rotate(-55 ${x} ${m.t - 8})`, class: "viz-tick" }, svg);
    t.textContent = c;
  });
  rowNames.forEach((r, i) => {
    el("text", { x: m.l - 8, y: m.t + i * ch + ch / 2 + 4, "text-anchor": "end", class: "viz-label" }, svg).textContent = r;
    colNames.forEach((c, j) => {
      const v = val(i, j);
      const fill = v == null ? css("--grid") : ramp[Math.min(ramp.length - 1, Math.floor(v.p * ramp.length))];
      const cell = el("rect", { x: m.l + j * cw + 1, y: m.t + i * ch + 1, width: cw - 2, height: ch - 2, rx: 3, fill }, svg);
      hover(cell, () => [[v == null ? "n/a" : fmt(v.p), `${r} · ${c}`, fill], ...(v ? [[`${v.k}/${v.n}`, "violations / decisions"]] : [])]);
    });
  });
  const lg = fig.querySelector(".viz-legend");
  if (lg) {
    lg.replaceChildren();
    const s = document.createElement("span"); s.className = "viz-scale";
    ramp.forEach((c, i) => { const b = document.createElement("span"); b.style.background = c; b.title = `${Math.round((100 * i) / ramp.length)}–${Math.round((100 * (i + 1)) / ramp.length)}%`; s.appendChild(b); });
    const a = document.createElement("span"); a.textContent = "0%"; const z = document.createElement("span"); z.textContent = "100% violation rate";
    lg.append(a, s, z);
  }
  tableTwin(fig, [rowLabel, ...colNames], rowNames.map((r, i) => [r, ...colNames.map((_, j) => { const v = val(i, j); return v ? `${fmt(v.p)} (${v.k}/${v.n})` : "–"; })]));
}

/* Dot + interval per row, optional reference lines. rows: {label, groups: [{name, color, p, lo, hi}]} */
function dots(fig, rows, { min = 0, max = 1, fmt = pct, axis = "", refs = [], ticks } = {}) {
  const rowH = 34, m = { t: 10, r: 24, b: axis ? 46 : 30, l: Math.min(200, 12 + 7.2 * Math.max(...rows.map((r) => r.label.length))) };
  const f = frame(fig, m.t + m.b + rows.length * rowH, m);
  const sx = (v) => m.l + ((v - min) / (max - min)) * f.iw;
  xAxis(f.svg, sx, ticks || [0, 0.25, 0.5, 0.75, 1].map((t) => min + t * (max - min)), m.t, rows.length * rowH, fmt, axis, f.W, m);
  for (const r of refs) {
    el("line", { x1: sx(r.x), x2: sx(r.x), y1: m.t, y2: m.t + rows.length * rowH, stroke: css("--ink-2"), "stroke-width": 1 }, f.svg);
    el("text", { x: sx(r.x) + 4, y: m.t + 10, class: "viz-note" }, f.svg).textContent = r.label;
  }
  const names = [...new Map(rows.flatMap((r) => r.groups.map((g) => [g.name, g.color]))).entries()];
  rows.forEach((r, i) => {
    const yc = m.t + i * rowH + rowH / 2;
    el("text", { x: m.l - 10, y: yc + 4, "text-anchor": "end", class: "viz-label" }, f.svg).textContent = r.label;
    r.groups.forEach((g, k) => {
      const y = yc + (k - (r.groups.length - 1) / 2) * 9;
      if (g.lo != null) el("line", { x1: sx(g.lo), x2: sx(g.hi), y1: y, y2: y, stroke: g.color, "stroke-width": 2, "stroke-linecap": "round" }, f.svg);
      el("circle", { cx: sx(g.p), cy: y, r: 5, fill: g.color, stroke: css("--surface"), "stroke-width": 2 }, f.svg);
    });
    const hit = el("rect", { x: 0, y: yc - rowH / 2, width: f.W, height: rowH, fill: "transparent" }, f.svg);
    hover(hit, () => r.groups.map((g) => [`${fmt(g.p)}${g.lo != null ? `  [${fmt(g.lo)}, ${fmt(g.hi)}]` : ""}`, `${r.label} · ${g.name}`, g.color]));
  });
  legend(fig, names.map(([n, c]) => [n, c, "rect"]));
  tableTwin(fig, ["", ...names.map(([n]) => n)], rows.map((r) => [r.label, ...names.map(([n]) => { const g = r.groups.find((q) => q.name === n); return g ? `${fmt(g.p)}${g.lo != null ? ` [${fmt(g.lo)}, ${fmt(g.hi)}]` : ""}` : "–"; })]));
}

window.viz = { barsCI, lines, heatmap, dots, pct, fix, css };

/* XY chart: each series is its own polyline over (x, y) points (frontier plots). Hover = nearest point. */
function xy(fig, series, { xMin = 0, xMax = 1, yMin = 0, yMax = 1, xFmt = pct, yFmt = pct, xLabel = "", yLabel = "", marks = [] } = {}) {
  const m = { t: 14, r: 24, b: 46, l: 54 };
  const f = frame(fig, 320, m);
  const sx = (x) => m.l + ((x - xMin) / (xMax - xMin)) * f.iw;
  const sy = (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * f.ih;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const yv = yMin + t * (yMax - yMin), xv = xMin + t * (xMax - xMin);
    el("line", { x1: m.l, x2: m.l + f.iw, y1: sy(yv), y2: sy(yv), stroke: css("--grid"), "stroke-width": 1 }, f.svg);
    el("text", { x: m.l - 8, y: sy(yv) + 4, "text-anchor": "end", class: "viz-tick" }, f.svg).textContent = yFmt(yv);
    el("text", { x: sx(xv), y: m.t + f.ih + 18, "text-anchor": "middle", class: "viz-tick" }, f.svg).textContent = xFmt(xv);
  }
  el("text", { x: m.l + f.iw / 2, y: m.t + f.ih + 38, "text-anchor": "middle", class: "viz-axis-label" }, f.svg).textContent = xLabel;
  el("text", { x: 12, y: m.t + f.ih / 2, transform: `rotate(-90 12 ${m.t + f.ih / 2})`, "text-anchor": "middle", class: "viz-axis-label" }, f.svg).textContent = yLabel;
  const all = [];
  for (const s of series) {
    const pts = s.points.slice().sort((a, b) => a.x - b.x);
    if (pts.length > 1) el("polyline", { points: pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }, f.svg);
    for (const p of pts) { el("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, f.svg); all.push({ s, p }); }
  }
  for (const mk of marks) {
    el("rect", { x: sx(mk.x) - 5, y: sy(mk.y) - 5, width: 10, height: 10, fill: mk.color || css("--ink-2"), stroke: css("--surface"), "stroke-width": 2 }, f.svg);
    el("text", { x: sx(mk.x) + 9, y: sy(mk.y) + 4, class: "viz-note" }, f.svg).textContent = mk.label;
    all.push({ s: { name: mk.label, color: mk.color || css("--ink-2") }, p: mk });
  }
  const hit = el("rect", { x: m.l, y: m.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
  hit.addEventListener("pointermove", (e) => {
    const b = f.svg.getBoundingClientRect(), px = ((e.clientX - b.left) / b.width) * f.W, py = ((e.clientY - b.top) / b.height) * f.H;
    const near = all.reduce((a, q) => (Math.hypot(sx(q.p.x) - px, sy(q.p.y) - py) < Math.hypot(sx(a.p.x) - px, sy(a.p.y) - py) ? q : a), all[0]);
    showTip(e, [[`${yFmt(near.p.y)} ${yLabel.toLowerCase()}`, near.s.name, near.s.color], [`${xFmt(near.p.x)} ${xLabel.toLowerCase()}`, near.p.note || ""]]);
  });
  hit.addEventListener("pointerleave", hideTip);
  legend(fig, [...series.map((s) => [s.name, s.color, "line"]), ...marks.map((mk) => [mk.label, mk.color || css("--ink-2"), "rect"])]);
  tableTwin(fig, ["series", xLabel, yLabel, ""], all.map(({ s, p }) => [s.name, xFmt(p.x), yFmt(p.y), p.note || ""]));
}
window.viz.xy = xy;
