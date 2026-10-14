/* =============================================================
   ConvexGuard — app.js
   All logic runs entirely in the browser; zero network requests.
   Depends on: window.DATA (from data.js), Chart.js, KaTeX
   ============================================================= */

"use strict";

// ── Palette ───────────────────────────────────────────────────
const C_BLUE = "#1a56db";
const C_RED = "#c0392b";
const C_GRAY = "#888888";
const C_GRID = "#e8e8e8";
const C_BLUE_T = "rgba(26, 86, 219, 0.2)";
const C_RED_T = "rgba(192, 57, 43, 0.2)";

// ── Data references from data.js ──────────────────────────────
const D = window.DATA || {};
const M = D.model || {};
const FEATURES = M.feature_names || [];
const W = M.w || [];
const B = M.b || 0;
const MEAN = M.mean || [];
const SCALE = M.scale || [];
const THRESH = M.threshold || 0;

const KEY_FEATURES = ["Amount", "Hour", "V4", "V9", "V10", "V12", "V14", "V17"];

// Active Chart.js instances registry
const CHARTS = {};

// ── Shared Chart.js defaults ──────────────────────────────────
if (window.Chart) {
  Chart.defaults.font.family = "system-ui, -apple-system, 'Segoe UI', sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = "#5a5a5a";
  if (Chart.defaults.plugins && Chart.defaults.plugins.legend) {
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
  }
}

function chartOptions(extra = {}) {
  return Object.assign({
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { position: "top" },
      tooltip: { mode: "index" },
    },
    scales: {
      x: { grid: { color: C_GRID } },
      y: { grid: { color: C_GRID } },
    },
  }, extra);
}

function makeChart(id, cfg) {
  const el = document.getElementById(id);
  if (!el || !window.Chart) return null;
  if (CHARTS[id]) {
    try { CHARTS[id].destroy(); } catch (e) { }
  }
  CHARTS[id] = new Chart(el, cfg);
  return CHARTS[id];
}

function r(v, d = 4) { return typeof v === "number" ? v.toFixed(d) : v; }
function round(v, d = 4) { return Math.round(v * 10 ** d) / 10 ** d; }

function getLegitMeans() {
  const means = {};
  FEATURES.forEach((f, i) => {
    means[f] = MEAN[i] !== undefined ? MEAN[i] : 0;
  });
  return means;
}

// ─────────────────────────────────────────────────────────────
// 1. CHECK A TRANSACTION
// ─────────────────────────────────────────────────────────────
function initCheck() {
  const adv = document.getElementById("adv-inputs");
  if (adv && adv.children.length === 0) {
    const legit = getLegitMeans();
    FEATURES.forEach(f => {
      if (KEY_FEATURES.includes(f) || f === "LogAmount") return;
      const g = document.createElement("div");
      g.className = "form-group";
      g.innerHTML = `<label for="inp-${f}">${f}</label>
        <input id="inp-${f}" type="number" step="0.001" value="${r(legit[f], 4)}" oninput="runScore()">`;
      adv.appendChild(g);
    });
  }
  runScore();
}

function getInputValues() {
  const vals = {};
  FEATURES.forEach((f, i) => {
    const el = document.getElementById("inp-" + f);
    if (el) {
      vals[f] = parseFloat(el.value) || 0;
    } else {
      vals[f] = MEAN[i] !== undefined ? MEAN[i] : 0;
    }
  });

  // Derived LogAmount from #inp-Amount
  const amtEl = document.getElementById("inp-Amount");
  if (amtEl && FEATURES.includes("LogAmount")) {
    const amt = Math.max(0, parseFloat(amtEl.value) || 0);
    vals["LogAmount"] = Math.log(1 + amt);
  }

  // Derived Hour from #inp-Hour
  const hrEl = document.getElementById("inp-Hour");
  if (hrEl && FEATURES.includes("Hour")) {
    vals["Hour"] = Math.round(parseFloat(hrEl.value) || 0) % 24;
  }

  return vals;
}

function computeScore(vals) {
  let score = B;
  const terms = [];
  FEATURES.forEach((f, i) => {
    const raw = vals[f] !== undefined ? vals[f] : (MEAN[i] || 0);
    const mean = MEAN[i] !== undefined ? MEAN[i] : 0;
    const scale = SCALE[i] || 1;
    const z = (raw - mean) / scale;
    const wi = W[i] !== undefined ? W[i] : 0;
    const c = wi * z;
    score += c;
    terms.push({ feature: f, weight_z: c, z, raw });
  });
  return { score, terms };
}

function runScore() {
  const vals = getInputValues();
  const { score, terms } = computeScore(vals);
  const isFraud = score >= THRESH;

  // Single verdict badge with text + icon
  const vEl = document.getElementById("verdict");
  if (vEl) {
    vEl.textContent = isFraud ? "⚠ FRAUD  —  f(x) ≥ threshold" : "✓ SAFE  —  f(x) < threshold";
    vEl.className = isFraud ? "fraud" : "safe";
  }

  const scEl = document.getElementById("score-val");
  if (scEl) {
    scEl.textContent = `f(x) = ${score.toFixed(6)}    (Threshold: ${THRESH.toFixed(6)})`;
  }

  // Formula line with user numbers
  const fEl = document.getElementById("formula-line");
  if (fEl) {
    const sorted = [...terms].sort((a, b) => Math.abs(b.weight_z) - Math.abs(a.weight_z));
    const top4 = sorted.slice(0, 4);
    const parts = top4.map(t => `${t.weight_z >= 0 ? "+" : ""}${t.weight_z.toFixed(4)} [${t.feature}]`);
    fEl.textContent = `f(x) = wᵀz + b = ${parts.join("  ")}  + … + (${B.toFixed(4)}) = ${score.toFixed(6)}`;
  }

  const box = document.getElementById("result-box");
  if (box) box.style.display = "";

  drawAxisChart(score);
  drawHistChart(score);
  drawContribChart(terms);
}

// 1. Decision-score axis with threshold and margin lines
function drawAxisChart(score) {
  const canvas = document.getElementById("chart-axis");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  const parentW = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
  const w = Math.max(parentW || 380, 320);
  const h = 180;

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, w, h);

  const min = Math.min(-6, THRESH - 3, score - 1.5);
  const max = Math.max(6, THRESH + 3, score + 1.5);
  const pad = 40;
  const toX = val => pad + ((val - min) / (max - min)) * (w - 2 * pad);
  const yAxis = 110;

  // Background regions: Safe vs Fraud
  const xThresh = toX(THRESH);
  ctx.fillStyle = "rgba(26, 86, 219, 0.06)";
  ctx.fillRect(pad, 30, Math.max(0, xThresh - pad), yAxis - 10);
  ctx.fillStyle = "rgba(192, 57, 43, 0.06)";
  ctx.fillRect(xThresh, 30, Math.max(0, w - pad - xThresh), yAxis - 10);

  // Region headers
  ctx.font = "10px system-ui";
  ctx.fillStyle = "#1a56db";
  ctx.textAlign = "left";
  ctx.fillText("Safe zone (f < t)", pad + 8, 45);
  ctx.fillStyle = "#c0392b";
  ctx.fillText("Fraud zone (f ≥ t)", Math.min(w - pad - 95, xThresh + 8), 45);

  // Axis line
  ctx.beginPath();
  ctx.strokeStyle = "#555555";
  ctx.lineWidth = 1.5;
  ctx.moveTo(pad - 10, yAxis);
  ctx.lineTo(w - pad + 10, yAxis);
  ctx.stroke();

  // Axis ticks
  const tickStep = 2;
  const firstTick = Math.ceil(min / tickStep) * tickStep;
  ctx.font = "10px monospace";
  ctx.fillStyle = "#777777";
  ctx.textAlign = "center";
  for (let tv = firstTick; tv <= max; tv += tickStep) {
    const tx = toX(tv);
    if (tx >= pad && tx <= w - pad) {
      ctx.beginPath();
      ctx.moveTo(tx, yAxis - 4);
      ctx.lineTo(tx, yAxis + 4);
      ctx.strokeStyle = "#aaaaaa";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillText(tv.toString(), tx, yAxis + 16);
    }
  }

  // Margin lines: threshold - 1 and threshold + 1
  [THRESH - 1, THRESH + 1].forEach((mVal, idx) => {
    const mx = toX(mVal);
    ctx.beginPath();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = "#999999";
    ctx.lineWidth = 1;
    ctx.moveTo(mx, 55);
    ctx.lineTo(mx, yAxis + 8);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "9px system-ui";
    ctx.fillStyle = "#888888";
    ctx.fillText(idx === 0 ? "Margin (-1)" : "Margin (+1)", mx, yAxis + 28);
  });

  // Threshold line
  ctx.beginPath();
  ctx.setLineDash([4, 3]);
  ctx.strokeStyle = "#222222";
  ctx.lineWidth = 1.5;
  ctx.moveTo(xThresh, 50);
  ctx.lineTo(xThresh, yAxis + 8);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = "bold 10px system-ui";
  ctx.fillStyle = "#222222";
  ctx.fillText(`Threshold (${THRESH.toFixed(2)})`, xThresh, 62);

  // User score pin
  const xScore = toX(score);
  const isFraud = score >= THRESH;
  ctx.beginPath();
  ctx.strokeStyle = isFraud ? C_RED : C_BLUE;
  ctx.lineWidth = 2;
  ctx.moveTo(xScore, yAxis - 28);
  ctx.lineTo(xScore, yAxis);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(xScore, yAxis, 5, 0, Math.PI * 2);
  ctx.fillStyle = isFraud ? C_RED : C_BLUE;
  ctx.fill();

  ctx.font = "bold 11px system-ui";
  ctx.fillStyle = isFraud ? C_RED : C_BLUE;
  ctx.fillText(`Your score: ${score.toFixed(2)}`, xScore, yAxis - 34);
}

// 2. Score distribution histogram with user score marked
function drawHistChart(score) {
  const hist = D.metrics?.score_hist || { fraud: [], legit: [] };
  const allScores = [...(hist.fraud || []), ...(hist.legit || [])];
  if (!allScores.length) return;
  const mn = Math.min(...allScores), mx = Math.max(...allScores);
  const nBins = 35;
  const step = (mx - mn) / nBins;
  const bins = Array.from({ length: nBins }, (_, i) => mn + i * step);
  const count = (arr, b) => arr.filter(v => v >= b && v < b + step).length;
  const fCounts = bins.map(b => count(hist.fraud, b));
  const lCounts = bins.map(b => count(hist.legit, b));
  const labels = bins.map(b => b.toFixed(1));

  let closestIdx = 0;
  let minDiff = Infinity;
  bins.forEach((b, i) => {
    const diff = Math.abs(b - score);
    if (diff < minDiff) { minDiff = diff; closestIdx = i; }
  });

  makeChart("chart-hist", {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Fraud", data: fCounts, backgroundColor: C_RED_T, borderColor: C_RED, borderWidth: 1 },
        { label: "Legit", data: lCounts, backgroundColor: C_BLUE_T, borderColor: C_BLUE, borderWidth: 1 },
      ],
    },
    options: {
      ...chartOptions(),
      scales: {
        x: { title: { display: true, text: "Decision score f(x)" }, grid: { color: C_GRID }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Count" }, grid: { color: C_GRID } },
      },
    },
    plugins: [{
      id: "userScoreLine",
      afterDraw(chart) {
        const meta = chart.getDatasetMeta(0);
        if (!meta || !meta.data || !meta.data[closestIdx]) return;
        const xCoord = meta.data[closestIdx].x;
        const ctx = chart.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.moveTo(xCoord, chart.chartArea.top);
        ctx.lineTo(xCoord, chart.chartArea.bottom);
        ctx.stroke();

        ctx.fillStyle = "#111111";
        ctx.font = "bold 10px system-ui";
        ctx.textAlign = "center";
        ctx.fillText(`Score: ${score.toFixed(2)}`, xCoord, chart.chartArea.top + 10);
        ctx.restore();
      }
    }]
  });
}

// 3. Top-8 feature influence horizontal bars
function drawContribChart(terms) {
  const top8 = [...terms].sort((a, b) => Math.abs(b.weight_z) - Math.abs(a.weight_z)).slice(0, 8);
  makeChart("chart-contrib", {
    type: "bar",
    data: {
      labels: top8.map(c => c.feature),
      datasets: [{
        label: "Feature influence (w_j · z_j)",
        data: top8.map(c => round(c.weight_z, 5)),
        backgroundColor: top8.map(c => c.weight_z > 0 ? C_RED_T : C_BLUE_T),
        borderColor: top8.map(c => c.weight_z > 0 ? C_RED : C_BLUE),
        borderWidth: 1,
      }],
    },
    options: {
      ...chartOptions(),
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "w_j · z_j (positive = fraud, negative = safe)" }, grid: { color: C_GRID } },
        y: { grid: { display: false } },
      },
    },
  });
}

// Load real example
window.loadExample = function (type) {
  const ex = (D.examples?.[type] || [])[0];
  if (!ex) return;
  const feats = ex.features || {};

  FEATURES.forEach(f => {
    const el = document.getElementById("inp-" + f);
    if (el && feats[f] !== undefined) {
      if (el.type === "range") {
        el.value = feats[f];
        const lbl = document.getElementById("val-" + f);
        if (lbl) lbl.textContent = parseFloat(feats[f]).toFixed(2);
      } else {
        el.value = feats[f];
      }
    }
  });

  if (feats["Amount"] !== undefined) {
    const a = document.getElementById("inp-Amount");
    if (a) a.value = feats["Amount"];
  }
  if (feats["Hour"] !== undefined) {
    const h = document.getElementById("inp-Hour");
    if (h) h.value = feats["Hour"];
  }

  runScore();
};

window.runScore = runScore;

// ─────────────────────────────────────────────────────────────
// 3. EDA
// ─────────────────────────────────────────────────────────────
let edaInited = false;
function initEDA() {
  if (edaInited) return;
  edaInited = true;
  const eda = D.eda || {};

  // Snapshot cards
  const snap = eda.snapshot || {};
  const cards = document.getElementById("eda-stats");
  if (cards) {
    cards.innerHTML = `
      <div class="stat-card"><div class="sc-val">${(snap.n_rows || 0).toLocaleString()}</div><div class="sc-label">Total transactions</div></div>
      <div class="stat-card"><div class="sc-val">${(snap.n_fraud || 0).toLocaleString()}</div><div class="sc-label">Fraud cases</div></div>
      <div class="stat-card"><div class="sc-val">${(snap.fraud_pct || 0).toFixed(3)}%</div><div class="sc-label">Fraud rate</div></div>
      <div class="stat-card"><div class="sc-val">${(snap.n_duplicates || 0).toLocaleString()}</div><div class="sc-label">Duplicates removed</div></div>
      <div class="stat-card"><div class="sc-val">${snap.n_missing || 0}</div><div class="sc-label">Missing values</div></div>
    `;
  }

  // Imbalance bar
  const n = snap.n_rows || 1;
  const nf = snap.n_fraud || 0;
  const nl = n - nf;
  makeChart("chart-imbalance", {
    type: "bar",
    data: {
      labels: ["Legitimate", "Fraud"],
      datasets: [{
        label: "Transactions", data: [nl, nf],
        backgroundColor: [C_BLUE_T, C_RED_T],
        borderColor: [C_BLUE, C_RED], borderWidth: 1
      }],
    },
    options: {
      ...chartOptions(), plugins: { legend: { display: false } },
      scales: {
        y: { title: { display: true, text: "Count" }, grid: { color: C_GRID } },
        x: { grid: { display: false } }
      }
    },
  });
  const cap = document.getElementById("imbalance-caption");
  if (cap) {
    cap.textContent = `A naive classifier that always predicts "legitimate" would achieve ${(100 * (1 - (snap.fraud_pct || 0) / 100)).toFixed(2)}% accuracy while catching zero frauds. Accuracy is the wrong metric.`;
  }

  // Amount histogram
  const ah = eda.amount_hist || {};
  const edges = ah.bin_edges || [];
  const midpoints = edges.slice(0, -1).map((e, i) => ((e + edges[i + 1]) / 2).toFixed(2));
  makeChart("chart-amount", {
    type: "bar",
    data: {
      labels: midpoints,
      datasets: [
        { label: "Fraud", data: ah.fraud_counts || [], backgroundColor: C_RED_T, borderColor: C_RED, borderWidth: 1 },
        { label: "Legit", data: ah.legit_counts || [], backgroundColor: C_BLUE_T, borderColor: C_BLUE, borderWidth: 1 },
      ],
    },
    options: {
      ...chartOptions(),
      scales: {
        x: { title: { display: true, text: "Amount (log scale midpoints, $)" }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Count" }, grid: { color: C_GRID } }
      }
    },
  });
  const acap = document.getElementById("amount-caption");
  if (acap) {
    acap.textContent = `${(ah.fraud_tiny_pct || 0)}% of fraud transactions have Amount <= $1 — a common card-testing pattern. Legitimate amounts are more uniformly spread.`;
  }

  // Hour fraud rate
  const hrMap = eda.fraud_rate_by_hour || {};
  const hours = Array.from({ length: 24 }, (_, i) => String(i));
  const rates = hours.map(h => hrMap[h] || 0);
  makeChart("chart-hour", {
    type: "bar",
    data: {
      labels: hours,
      datasets: [{ label: "Fraud rate (%)", data: rates, backgroundColor: C_RED_T, borderColor: C_RED, borderWidth: 1 }],
    },
    options: {
      ...chartOptions(), plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Hour of day" } },
        y: { title: { display: true, text: "Fraud rate (%)" }, grid: { color: C_GRID } }
      }
    },
  });

  // Correlation bar chart
  const corr = eda.correlations || {};
  const sortedFeats = Object.keys(corr).sort((a, b) => Math.abs(corr[b]) - Math.abs(corr[a]));
  makeChart("chart-corr", {
    type: "bar",
    data: {
      labels: sortedFeats,
      datasets: [{
        label: "Correlation with Class",
        data: sortedFeats.map(f => corr[f]),
        backgroundColor: sortedFeats.map(f => corr[f] > 0 ? C_RED_T : C_BLUE_T),
        borderColor: sortedFeats.map(f => corr[f] > 0 ? C_RED : C_BLUE),
        borderWidth: 1,
      }],
    },
    options: {
      ...chartOptions(), plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45 } },
        y: { title: { display: true, text: "Pearson r" }, grid: { color: C_GRID } }
      }
    },
  });

  // Feature dropdown
  const sel = document.getElementById("feat-sel");
  if (sel) {
    sel.innerHTML = "";
    FEATURES.forEach(f => {
      const o = document.createElement("option");
      o.value = o.textContent = f;
      sel.appendChild(o);
    });
    if (sortedFeats[0]) sel.value = sortedFeats[0];
    drawFeatHist();
  }
}

function drawFeatHist() {
  const sel = document.getElementById("feat-sel");
  if (!sel) return;
  const feat = sel.value;
  const fh = (D.eda?.feature_histograms || {})[feat];
  if (!fh) return;
  makeChart("chart-feat", {
    type: "bar",
    data: {
      labels: (fh.centers || []).map(c => c.toFixed(2)),
      datasets: [
        { label: "Fraud", data: fh.fraud || [], backgroundColor: C_RED_T, borderColor: C_RED, borderWidth: 1 },
        { label: "Legit", data: fh.legit || [], backgroundColor: C_BLUE_T, borderColor: C_BLUE, borderWidth: 1 },
      ],
    },
    options: {
      ...chartOptions(),
      scales: {
        x: { title: { display: true, text: feat }, ticks: { maxTicksLimit: 12 } },
        y: { title: { display: true, text: "Count" }, grid: { color: C_GRID } }
      }
    },
  });
}
window.drawFeatHist = drawFeatHist;

// ─────────────────────────────────────────────────────────────
// 4. MODEL
// ─────────────────────────────────────────────────────────────
let modelInited = false;
function initModel() {
  if (modelInited) return;
  modelInited = true;

  // Settings box
  const s = document.getElementById("model-settings");
  if (s) {
    s.innerHTML = `<table style="width:auto;font-size:0.875rem"><tbody>
      <tr><td style="padding:0.2rem 1rem 0.2rem 0;color:var(--muted)">Solver</td><td>Primal subgradient (Pegasos), NumPy only</td></tr>
      <tr><td style="padding:0.2rem 1rem 0.2rem 0;color:var(--muted)">C<sub>+</sub> / C<sub>−</sub></td><td>${M.C_pos || 10} / ${M.C_neg || 1} &nbsp; (ratio ${M.C_ratio || 10})</td></tr>
      <tr><td style="padding:0.2rem 1rem 0.2rem 0;color:var(--muted)">Epochs</td><td>${M.max_epochs || 100}</td></tr>
      <tr><td style="padding:0.2rem 1rem 0.2rem 0;color:var(--muted)">Features</td><td>${M.n_features || 30}</td></tr>
      <tr><td style="padding:0.2rem 1rem 0.2rem 0;color:var(--muted)">Threshold</td><td><code>${(M.threshold || 0).toFixed(6)}</code> (tuned on validation set, F2)</td></tr>
    </tbody></table>`;
  }

  // KaTeX math rendering
  const kat = window.katex || null;
  function renderMath(id, tex) {
    const el = document.getElementById(id);
    if (!el) return;
    if (kat) {
      try {
        katex.render(tex, el, { displayMode: true, throwOnError: false });
      } catch (e) { el.textContent = tex; }
    } else {
      el.innerHTML = `<em>${tex}</em>`;
    }
  }

  renderMath("math-primal",
    "\\min_{\\mathbf{w},b,\\boldsymbol{\\xi}} \\;\\; \\frac{1}{2}\\|\\mathbf{w}\\|^2 + C_+ \\sum_{i:y_i=+1} \\xi_i \\;+\\; C_- \\sum_{i:y_i=-1} \\xi_i");
  renderMath("math-hinge",
    "\\min_{\\mathbf{w},b} \\;\\; \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2 + \\frac{1}{n}\\sum_i C_i \\cdot \\max\\!\\left(0,\\; 1 - y_i(\\mathbf{w}^\\top \\mathbf{x}_i + b)\\right)");
  renderMath("math-lagrangian",
    "L = \\frac{1}{2}\\|\\mathbf{w}\\|^2 + \\sum_i C_i \\xi_i - \\sum_i \\alpha_i \\left[y_i(\\mathbf{w}^\\top \\mathbf{x}_i+b)-1+\\xi_i\\right] - \\sum_i \\mu_i \\xi_i");
  renderMath("math-dual",
    "\\max_{\\boldsymbol{\\alpha}} \\;\\; \\sum_i \\alpha_i - \\frac{1}{2} \\sum_{i,j} \\alpha_i \\alpha_j y_i y_j \\,\\mathbf{x}_i^\\top \\mathbf{x}_j");

  // Compact Optimality check table: 5 rows
  const v = D.verification || {};
  const rows = [
    ["KKT max violation", v.kkt_max_violation, v.kkt_pass],
    ["Primal objective", v.primal_objective, v.gap_pass],
    ["Dual objective", v.dual_objective, v.gap_pass],
    ["Duality gap", v.duality_gap, v.gap_pass],
    ["Weight difference vs scikit-learn", v.weight_diff_vs_sklearn, v.weight_diff_pass],
  ];
  const tbody = document.querySelector("#verify-table tbody");
  if (tbody) {
    tbody.innerHTML = rows.map(([label, val, pass]) => {
      const passCell = pass === null || pass === undefined ? "" :
        pass ? `<td class="pass">PASS</td>` : `<td class="fail">FAIL</td>`;
      return `<tr><td>${label}</td><td class="num">${typeof val === "number" ? val.toFixed(6) : "—"}</td>${passCell}</tr>`;
    }).join("");
  }
}

// ─────────────────────────────────────────────────────────────
// 5. RESULTS
// ─────────────────────────────────────────────────────────────
let resultsInited = false;
function initResults() {
  if (resultsInited) return;
  resultsInited = true;

  const met = D.metrics || {};
  const grid = met.slider_grid || [];

  const slider = document.getElementById("thresh-range");
  if (slider) {
    slider.min = 0;
    slider.max = Math.max(0, grid.length - 1);
    slider.step = 1;

    let bestIdx = 0;
    let bestDist = Infinity;
    grid.forEach((g, i) => {
      const d = Math.abs(g.t - (met.threshold || 0));
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    });
    slider.value = bestIdx;
    updateThreshDisplay(bestIdx);
  }

  const pr = met.pr_curve || {};
  const prec = pr.precision || [];
  const rec = pr.recall || [];

  let threshPrecision = 0, threshRecall = 0;
  if (grid.length > 0) {
    const bestG = grid[parseInt(slider ? slider.value : 0, 10)] || grid[0];
    threshPrecision = bestG.p;
    threshRecall = bestG.r;
  }

  makeChart("chart-pr", {
    type: "line",
    data: {
      datasets: [
        {
          label: `PR curve (AUC = ${(met.pr_auc || 0).toFixed(4)})`,
          data: rec.map((r, i) => ({ x: r, y: prec[i] })),
          borderColor: C_BLUE, backgroundColor: C_BLUE_T, borderWidth: 2, pointRadius: 0, fill: true
        },
        {
          label: "Chosen threshold",
          data: [{ x: threshRecall, y: threshPrecision }],
          borderColor: C_RED, backgroundColor: C_RED,
          pointRadius: 6, pointStyle: "circle", showLine: false
        },
      ],
    },
    options: {
      ...chartOptions(),
      parsing: false,
      scales: {
        x: {
          type: "linear", min: 0, max: 1,
          title: { display: true, text: "Recall" }, grid: { color: C_GRID }
        },
        y: {
          type: "linear", min: 0, max: 1,
          title: { display: true, text: "Precision" }, grid: { color: C_GRID }
        },
      },
    },
  });

  // Comparison table
  const tbody = document.querySelector("#compare-table tbody");
  if (tbody) {
    tbody.innerHTML = (met.comparisons || []).map(c => {
      const isOwn = c.name.includes("C_ratio");
      return `<tr ${isOwn ? 'style="font-weight:600"' : ""}>
        <td>${c.name}</td>
        <td class="num">${(c.pr_auc || 0).toFixed(4)}</td>
        <td class="num">${(c.precision || 0).toFixed(4)}</td>
        <td class="num">${(c.recall || 0).toFixed(4)}</td>
        <td class="num">${c.false_alarms || 0}</td>
      </tr>`;
    }).join("");
  }
}

function onThreshSlider() {
  const el = document.getElementById("thresh-range");
  if (el) updateThreshDisplay(parseInt(el.value, 10));
}
window.onThreshSlider = onThreshSlider;

function updateThreshDisplay(idx) {
  const grid = D.metrics?.slider_grid || [];
  const g = grid[idx];
  if (!g) return;

  const dispEl = document.getElementById("thresh-disp");
  if (dispEl) dispEl.textContent = g.t.toFixed(5);

  const infoEl = document.getElementById("thresh-info");
  if (infoEl) {
    infoEl.textContent = `Precision: ${(g.p * 100).toFixed(1)}%  |  Recall: ${(g.r * 100).toFixed(1)}%  |  F1: ${(g.f1 * 100).toFixed(1)}%`;
  }

  const setT = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setT("cm-tp", g.tp);
  setT("cm-fp", g.fp);
  setT("cm-fn", g.fn);
  setT("cm-tn", g.tn);

  if (CHARTS["chart-pr"]) {
    CHARTS["chart-pr"].data.datasets[1].data = [{ x: g.r, y: g.p }];
    CHARTS["chart-pr"].update("none");
  }

  const cards = document.getElementById("thresh-cards");
  if (cards) {
    cards.innerHTML = [
      { label: "Precision", val: (g.p * 100).toFixed(1) + "%" },
      { label: "Recall", val: (g.r * 100).toFixed(1) + "%" },
      { label: "F1 score", val: (g.f1 * 100).toFixed(1) + "%" },
    ].map(c => `<div class="stat-card"><div class="sc-val">${c.val}</div><div class="sc-label">${c.label}</div></div>`).join("");
  }
}

// ─────────────────────────────────────────────────────────────
// Navigation & Application Bootstrap
// ─────────────────────────────────────────────────────────────
function activateTab(tab) {
  document.querySelectorAll("section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));

  const sec = document.getElementById("tab-" + tab);
  const btn = document.querySelector(`nav button[data-tab="${tab}"]`);
  if (sec && btn) {
    sec.classList.add("active");
    btn.classList.add("active");
    location.hash = tab;
    if (tab === "check") initCheck();
    if (tab === "eda") initEDA();
    if (tab === "model") initModel();
    if (tab === "results") initResults();
  }
}

function initApp() {
  const btns = document.querySelectorAll("nav button");
  btns.forEach(btn => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  const hash = location.hash.replace("#", "");
  if (hash && document.getElementById("tab-" + hash)) {
    activateTab(hash);
  } else {
    activateTab("check");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
