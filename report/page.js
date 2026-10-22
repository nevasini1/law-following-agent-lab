/* Render every figure from the embedded results. Each block is independent; missing data renders as pending. */
const DATA = JSON.parse(document.getElementById("data").textContent);
const { barsCI, lines, heatmap, dots, xy, pct, fix, css } = window.viz;
const S = () => ({ s1: css("--series-1"), s2: css("--series-2"), s3: css("--series-3"), s4: css("--series-4"), de: css("--deemph") });
const MODEL_ORDER = ["qwen2.5-0.5b", "gemma3-1b", "qwen2.5-1.5b", "llama3.2-3b", "phi4-mini", "gemma3-4b", "qwen3-4b"];
const NICE = { "qwen2.5-0.5b": "Qwen2.5 0.5B", "gemma3-1b": "Gemma 3 1B", "qwen2.5-1.5b": "Qwen2.5 1.5B", "llama3.2-3b": "Llama 3.2 3B",
  "phi4-mini": "Phi-4-mini 3.8B", "gemma3-4b": "Gemma 3 4B", "qwen3-4b": "Qwen3 4B (2507)", "qwen3guard-0.6b": "Qwen3Guard 0.6B (safety guard)",
  keyword: "Keyword filter" };
const FAMILY = { time_zone: "time zone", ambiguous_tz: "ambiguous location", inconvenient: "stated inconvenient time", attorney: "represented by attorney",
  cease: "cease letter", dispute: "pending dispute", frequency: "call frequency", workplace: "workplace (Horkey)", ai_voice: "AI voice, no consent",
  third_party: "third parties", threats: "threats", creditor_ma: "MA creditor rules", creditor_fl: "FL creditor rules",
  control_business: "control: business debt", control_clean: "control: no constraint" };
const nice = (m) => NICE[m] || m;

function mount(id, fn) {
  const fig = document.querySelector(`[data-chart="${id}"]`);
  if (!fig) return;
  const draw = () => {
    try { fn(fig); fig.classList.remove("is-pending"); } catch (e) {
      fig.classList.add("is-pending");
      const box = document.createElement("div");
      box.className = "viz-pending";
      box.textContent = "Queued. This chart renders when its job finishes.";
      fig.querySelector(".viz-plot").replaceChildren(box);
      if (!(e instanceof Error && e.message === "missing")) console.warn(id, e);
    }
  };
  draw();
  const btn = fig.querySelector(".viz-toggle"), tbl = fig.querySelector(".viz-table");
  if (btn && tbl) {
    tbl.hidden = true;
    btn.addEventListener("click", () => { tbl.hidden = !tbl.hidden; btn.textContent = tbl.hidden ? "Show table" : "Hide table"; });
  }
  let w = fig.clientWidth;
  new ResizeObserver(() => { if (Math.abs(fig.clientWidth - w) > 40) { w = fig.clientWidth; draw(); } }).observe(fig);
}
const need = (x) => { if (x == null || (typeof x === "object" && !Object.keys(x).length)) throw new Error("missing"); return x; };

/* ---------------- TEST ---------------- */
mount("leaderboard", (fig) => {
  const runs = need(DATA.agent_eval).runs, c = S();
  const rows = MODEL_ORDER.filter((m) => runs[`${m}__comply`]).map((m) => {
    const r = runs[`${m}__comply`];
    return { label: nice(m), groups: [
      { name: "violation rate", color: c.s2, p: r.violation.p, lo: r.violation.lo, hi: r.violation.hi },
      { name: "task success (lawful and productive)", color: c.s1, p: r.success.p, lo: r.success.lo, hi: r.success.hi }] };
  });
  dots(fig, rows, { axis: "share of 120 decisions (95% Wilson interval)" });
});
mount("families", (fig) => {
  const fam = need(DATA.agent_eval).families;
  const models = MODEL_ORDER.filter((m) => fam[`${m}__comply`]);
  const cols = Object.keys(FAMILY);
  heatmap(fig, models.map(nice), cols.map((f) => FAMILY[f]), (i, j) => fam[`${models[i]}__comply`][cols[j]]?.violation, { rowLabel: "model" });
});
mount("ablations", (fig) => {
  const runs = need(DATA.agent_eval).runs, c = S();
  const conds = [["bare", "no legal instruction"], ["comply", "comply with law"], ["rag", "comply + retrieved law"], ["pressure", "comply + manager pressure"]];
  const models = ["qwen2.5-1.5b", "qwen3-4b"].filter((m) => runs[`${m}__comply`]);
  const colors = [c.s1, c.s2];
  const rows = conds.filter(([k]) => models.some((m) => runs[`${m}__${k}`])).map(([k, label]) => ({ label, groups: models.filter((m) => runs[`${m}__${k}`]).map((m) => {
    const v = runs[`${m}__${k}`].violation; return { name: nice(m), color: colors[models.indexOf(m)], p: v.p, lo: v.lo, hi: v.hi }; }) }));
  dots(fig, rows, { axis: "violation rate (95% Wilson interval)" });
});
mount("episodes", (fig) => {
  const eps = need(DATA.case2env?.episodes);
  const models = MODEL_ORDER.filter((m) => eps[m]);
  const rows = models.map((m) => ({ label: nice(m), p: eps[m].any_violation_rate, lo: eps[m].lo, hi: eps[m].hi, note: eps[m].pattern }));
  barsCI(fig, rows, { axis: "share of Horkey replays with a violation", color: css("--series-2"), sub: "episodes with a violation" });
});

/* ---------------- REASON ---------------- */
mount("compositional", (fig) => {
  const R = need(DATA.reasoning), c = S(), cols = [c.s1, c.s2, c.s3];
  const series = Object.entries(R).map(([m, r], i) => ({ name: nice(m), color: cols[i], points: Object.entries(r.compositional).map(([k, v]) => ({ x: +k, y: v.balanced_acc })) }));
  lines(fig, series, { xLabel: "interacting legal constraints in the case file (k)", yLabel: "balanced accuracy", yMin: 0, yMax: 1, refs: [{ y: 0.5, label: "chance" }], yFmt: (v) => pct(v) });
});
for (const lad of ["workplace", "attorney", "attorney_fl", "threat"]) {
  mount(`ladder-${lad}`, (fig) => {
    const R = need(DATA.reasoning), c = S(), cols = [c.s1, c.s2, c.s3];
    const labels = DATA.ladder_rungs[lad];
    const series = Object.entries(R).filter(([, r]) => r.ladders[lad]).map(([m, r], i) => ({ name: nice(m), color: cols[i],
      points: r.ladders[lad].rung_means.map((y, k) => ({ x: k, y })) }));
    lines(fig, series, { xFmt: (k) => labels[k], xLabel: "", yLabel: "P(violation)", refs: [{ y: 0.5, label: "0.5" }], rotateX: true, endLabels: false });
  });
}

/* ---------------- JUDGE ---------------- */
mount("judges", (fig) => {
  const J = need(DATA.judges).judges, c = S();
  const order = Object.keys(J).sort((a, b) => J[b].auroc - J[a].auroc);
  const rows = order.map((n) => ({ label: nice(n), groups: [
    { name: "recall: state-dependent violations", color: c.s1, p: J[n].recall_state },
    { name: "recall: content violations", color: c.s2, p: J[n].recall_content },
    { name: "false-positive rate", color: c.s3, p: J[n].false_positive_rate }] }));
  dots(fig, rows, { axis: "rate at threshold 0.5" });
});
mount("ppi", (fig) => {
  const P = need(DATA.judges).ppi, c = S();
  const pts = (k) => P.studies.map((s) => ({ x: s.n_labeled, y: s[k].width }));
  lines(fig, [{ name: "gold labels only", color: c.s2, points: pts("classical") }, { name: "PPI++ (gold + judge)", color: c.s1, points: pts("ppi++") }],
    { xLabel: "gold-labeled actions (of the pool)", yLabel: "95% CI width", yMin: 0, yMax: Math.max(...P.studies.map((s) => s.classical.width)) * 1.1, yFmt: (v) => fix(v, 2), xFmt: (x) => x });
});
mount("ecthr", (fig) => {
  const C = need(DATA.court_labels), c = S();
  const by = (j) => C.curve.filter((r) => r.judge === j).map((r) => ({ x: r.n_train, y: r.auroc_mean, lo: r.auroc_mean - r.auroc_sd, hi: r.auroc_mean + r.auroc_sd }));
  const refs = Object.entries(C.llm || {}).map(([m, v]) => ({ y: v.auroc, label: `${nice(m)} zero-shot` }));
  lines(fig, [{ name: "TF-IDF + logistic", color: c.s1, points: by("tfidf+lr") }, { name: "bge-small + logistic", color: c.s2, points: by("bge-small+lr") }],
    { xLabel: "court-labeled training cases (log scale)", yLabel: "test AUROC", yMin: 0.5, yMax: 1, logX: true, refs, yFmt: (v) => fix(v, 2) });
});

/* ---------------- TRAIN ---------------- */
mount("grpo-curve", (fig) => {
  const G = need(DATA.grpo), c = S();
  const log = G.train_log, smooth = (k) => log.map((r, i) => { const w = log.slice(Math.max(0, i - 4), i + 1); return { x: r.step + 1, y: w.reduce((a, b) => a + b[k], 0) / w.length }; });
  lines(fig, [{ name: "violation rate", color: c.s2, points: smooth("violation") }, { name: "task success", color: c.s1, points: smooth("success") }],
    { xLabel: "GRPO step (5-step moving average)", yLabel: "share of rollouts", xTicks: log.filter((r) => (r.step + 1) % 10 === 0).map((r) => r.step + 1) });
});
mount("grpo-eval", (fig) => {
  const E = need(DATA.grpo).eval, c = S();
  const splits = [["train_families", "trained families"], ["heldout_families", "unseen families"]];
  const rows = [];
  for (const [k, label] of splits) for (const [metric, mlabel] of [["format", "valid action"], ["violation", "violation"], ["success", "success"]])
    rows.push({ label: `${label}: ${mlabel}`, groups: [
      { name: "base Qwen2.5-0.5B", color: c.de, p: E.base[k][metric].p, lo: E.base[k][metric].lo, hi: E.base[k][metric].hi },
      { name: "after GRPO", color: c.s1, p: E.trained[k][metric].p, lo: E.trained[k][metric].lo, hi: E.trained[k][metric].hi }] });
  dots(fig, rows, { axis: "rate on held-out scenarios (95% Wilson interval)" });
});

/* ---------------- CONTROL ---------------- */
mount("frontier", (fig) => {
  const C = need(DATA.control), c = S();
  const mk = (name, color) => ({ name, color, points: C.sweeps[name].map((r) => ({ x: r.useful / r.useful_den, y: r.prevented / r.violations, note: `τ = ${r.tau}` })) });
  const fixed = [];
  Object.entries(C.fixed).forEach(([name, r], i) => {
    const x = r.useful / r.useful_den, y = r.prevented / r.violations, label = name === "none" ? "no firewall" : name;
    const same = fixed.find((f) => f.x === x && f.y === y);  // configurations with identical outcomes share one mark
    if (same) same.label += ` = ${label}`; else fixed.push({ x, y, label, color: [c.de, css("--ink-2"), css("--ink")][i] });
  });
  xy(fig, [mk("monitor", c.s2), mk("rules+repair+monitor", c.s1), mk("rules+repair+monitor+audit5%", c.s3)], { xLabel: "Usefulness", yLabel: "Safety", marks: fixed });
});

/* ---------------- PROBE ---------------- */
mount("probes", (fig) => {
  const P = need(DATA.probes), c = S();
  lines(fig, [{ name: "action legality (P1)", color: c.s1, points: P.p1.layer_auroc.map((y, x) => ({ x, y })) },
    { name: "instruction risk (P2)", color: c.s2, points: P.p2.layer_auroc.map((y, x) => ({ x, y })) }],
    { xLabel: "layer (0 = embeddings)", yLabel: "probe AUROC (grouped 5-fold CV)", yMin: 0.4, yMax: 1, yFmt: (v) => fix(v, 2),
      refs: [{ y: P.p1.verbal_auroc, label: "model's own Yes/No" }, { y: 0.5, label: "chance" }],
      xTicks: P.p1.layer_auroc.map((_, i) => i).filter((i) => i % 4 === 0) });
});

/* ---------------- COLLUDE ---------------- */
mount("collusion", (fig) => {
  const C = need(DATA.collusion), c = S();
  const conds = [["no_channel", "no communication"], ["channel", "chat channel"], ["channel+law", "chat + antitrust instruction"], ["channel+monitor", "chat + runtime monitor"]];
  const rows = conds.filter(([k]) => C.by_condition[k]).map(([k, label]) => {
    const r = C.by_condition[k]; return { label, groups: [{ name: "collusion index (mean ± 1 sd over seeds)", color: c.s1, p: r.mean, lo: r.mean - r.sd, hi: r.mean + r.sd }] };
  });
  dots(fig, rows, { min: -1, max: 1.5, fmt: (v) => fix(v, 2), axis: "collusion index: 0 = Bertrand-Nash, 1 = joint monopoly",
    refs: [{ x: 0, label: "Nash" }, { x: 1, label: "monopoly" }], ticks: [-1, -0.5, 0, 0.5, 1, 1.5] });
});

/* ---------------- data-driven text and tables ---------------- */
const fillEl = (key) => document.querySelector(`[data-fill="${key}"]`);
function setText(key, text) { const n = fillEl(key); if (n && text != null) n.textContent = text; }
function table(key, head, rows) {
  const t = fillEl(key);
  if (!t) return;
  t.replaceChildren();
  const hr = t.createTHead().insertRow();
  for (const h of head) { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); }
  const tb = t.createTBody();
  for (const r of rows) {
    const tr = tb.insertRow();
    for (const c of r) {
      const td = tr.insertCell();
      if (c && typeof c === "object") { const s = document.createElement("span"); s.className = `chip ${c.cls}`; s.textContent = c.text; td.appendChild(s); }
      else td.textContent = c ?? "";
    }
  }
}
const ex = DATA.example;
if (ex) {
  setText("example-case", ex.case);
  let parsed = {}; try { parsed = JSON.parse(ex.raw); } catch (e) { /* keep raw */ }
  setText("example-out", `${nice(ex.model)} replied:\n${JSON.stringify(parsed, null, 2)}\n\nOracle:\n${ex.findings.replaceAll("; ", "\n")}`);
}
const fods = DATA.court_labels?.fods_trained;
const fodsLLM = Object.entries(DATA.court_labels?.llm || {}).map(([m, v]) => [m, v.fods]).filter(([, f]) => f && f.n);
if (fods) setText("fods", `Cross-validated on all 367 LegalBench paragraphs (seven functions, so chance is about 14%), TF-IDF with logistic regression scores ${pct(fods["tfidf+lr"].accuracy)} accuracy and bge-small embeddings ${pct(fods["bge-small+lr"].accuracy)}` +
  (fodsLLM.length ? `, while ${nice(fodsLLM[0][0])} with few-shot prompting scores ${pct(fodsLLM[0][1].accuracy)}. The trained classifier does the segmenting.` : "."));
const specs = DATA.case2env?.specs;
if (specs) {
  const tot = Object.values(specs).reduce((a, s) => [a[0] + s.n_verified, a[1] + s.n_events], [0, 0]);
  setText("spans", `Across the seven opinions, ${tot[0]} of ${tot[1]} extracted events (${pct(tot[0] / Math.max(tot[1], 1))}) quoted their source verbatim; the rest were rejected.`);
}
if (DATA.agent_eval) {
  const fam = DATA.agent_eval.families;
  setText("trap", DATA.trap_rate != null ? pct(DATA.trap_rate) : "most");
}
if (DATA.deontic) table("deontic", ["Situation", "Verdict", "Why"], DATA.deontic.map((d) => {
  const v = d.verdict.startsWith("permitted") || d.verdict === "no norm" ? { cls: "ok", text: d.verdict === "no norm" ? "permitted" : d.verdict } :
    d.verdict === "forbidden" ? { cls: "viol", text: "forbidden" } : { cls: "contested", text: d.verdict };
  const why = d.defeats.length ? d.defeats.map(([a, b, p]) => `${a} overrides ${b} (${p})`).join("; ") :
    d.sources.length ? `only ${d.sources.join(", ")} applies` : "no prohibition applies (oral request expired after 10 days)";
  return [d.facts, v, why];
}));
if (DATA.judges) {
  setText("pool", DATA.judges.n);
  const J = DATA.judges.judges;
  table("judge-table", ["Judge", "AUROC", "Accuracy", "Cohen's κ", "ECE", "Recall: state", "Recall: content", "False positives"],
    Object.keys(J).sort((a, b) => J[b].auroc - J[a].auroc).map((n) => [nice(n), fix(J[n].auroc), pct(J[n].accuracy), fix(J[n].kappa), fix(J[n].ece),
      pct(J[n].recall_state), pct(J[n].recall_content), pct(J[n].false_positive_rate)]));
}
for (const [k, v] of Object.entries(DATA.captions || {})) setText(k, v);
for (const [k, v] of Object.entries(DATA.notes || {})) {
  const n = fillEl(k);
  if (!n) continue;
  const p = document.createElement("p");
  p.textContent = v;
  if (v.startsWith("Queued")) p.className = "pending";
  n.replaceChildren(p);
}
if (DATA.market) { setText("pnash", `$${fix(DATA.market.p_nash)}`); setText("pmono", `$${fix(DATA.market.p_mono)}`); }
setText("ntests", DATA.n_tests);
if (DATA.control?.decisions?.length) {
  const lines = DATA.control.decisions.map((d) => {
    const a = d.action, r = d.repaired_to;
    const fmtT = (t) => (t ? new Date(t).toLocaleString("en-US", { timeZone: "America/New_York", weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false }) + " ET" : "now");
    return `REPAIR  ${d.scenario.padEnd(18)} ${a.tool} ${a.target} ${fmtT(a.send_at)}  ->  ${fmtT(r.send_at)}\n        ${d.reasons.join("; ")}`;
  });
  setText("audit-log", "firewall audit log (rule layer, repaired calls)\n\n" + lines.join("\n"));
}
if (DATA.sources) table("sources", ["Source", "Citation", "Retrieved", "sha256"], DATA.sources.map((s) => [s.title, s.citation, s.retrieved_at.slice(0, 10), s.sha256.slice(0, 12)]));
