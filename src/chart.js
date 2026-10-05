// Builds the self-contained demo chart (results/chart.html) from runExperiment's output.
// No dependencies: inline SVG drawn by a small script, light and dark themes,
// hover tooltips, and a table view of the same numbers.

import { SITES } from './data.js';

export function buildChartData(results) {
  const ids = Object.keys(results.sites);
  const rows = [
    ...ids.map((id) => ({
      key: id, label: `${SITES[id].label} alone`, group: 'local',
      auc: results.baselines.localOnly[id].all.auc, accuracy: results.baselines.localOnly[id].all.accuracy,
    })),
    { key: 'federated', label: 'Federated (all 4 hospitals)', group: 'federated',
      auc: results.federated.metrics.all.auc, accuracy: results.federated.metrics.all.accuracy },
  ];
  if (results.federatedDropout) {
    rows.push({ key: 'dropout', label: `Federated, ${SITES[results.config.dropout.site].label} drops out`, group: 'dropout',
      auc: results.federatedDropout.metrics.all.auc, accuracy: results.federatedDropout.metrics.all.accuracy });
  }
  rows.push({ key: 'pooled', label: 'Pooled data (upper bound)', group: 'pooled',
    auc: results.baselines.pooled.all.auc, accuracy: results.baselines.pooled.all.accuracy });

  const bestLocal = Math.max(...ids.map((id) => results.baselines.localOnly[id].all.auc));
  return {
    rows,
    curves: {
      federated: results.federated.curve.map((c) => [c.round, c.auc]),
      dropout: results.federatedDropout ? results.federatedDropout.curve.map((c) => [c.round, c.auc]) : null,
    },
    dropout: results.federatedDropout
      ? { round: results.config.dropout.fromRound, label: SITES[results.config.dropout.site].label }
      : null,
    pooledAuc: results.baselines.pooled.all.auc,
    bestLocalAuc: bestLocal,
    testPatients: Object.values(results.sites).reduce((s, x) => s + x.test, 0),
    perSite: results.perSite
      ? Object.entries(results.perSite).map(([id, p]) => ({
          label: SITES[id].label,
          strategy: p.strategyLabel,
          alone: { logLoss: p.alone.logLoss, accuracy: p.alone.accuracy },
          federated: { logLoss: p.federated.logLoss, accuracy: p.federated.accuracy },
          personalized: { logLoss: p.personalized.logLoss, accuracy: p.personalized.accuracy },
          test: results.sites[id].test,
        }))
      : null,
  };
}

export function renderChartHtml(results) {
  const data = JSON.stringify(buildChartData(results)).replace(/</g, '\\u003c');
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FedHeart Results</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 880px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 2px; }
.sub { color: var(--text-secondary); margin: 0 0 24px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; position: relative; }
.card p { color: var(--text-secondary); margin: 0 0 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 0 0 8px; color: var(--text-secondary); font-size: 13px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.line-swatch { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-secondary); font-size: 12px; font-variant-numeric: tabular-nums; }
.tick { fill: var(--muted); }
.tooltip { position: absolute; pointer-events: none; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); opacity: 0; transition: opacity .1s; white-space: nowrap; }
.tooltip b { font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; }
td.num, th.num { text-align: right; }
details summary { cursor: pointer; color: var(--text-secondary); }
</style>
</head>
<body>
<main>
  <h1>FedHeart: one model, four hospitals, no shared patient data</h1>
  <p class="sub" id="subtitle"></p>

  <section class="card">
    <h2>Accuracy of each approach (ROC AUC)</h2>
    <p>Every model is scored on the same held-out patients from all four hospitals. 0.5 is a coin flip, 1.0 is perfect.</p>
    <div class="legend" id="legend1"></div>
    <svg id="dots" role="img" aria-label="Dot plot of ROC AUC for each model"></svg>
    <div class="tooltip" id="tip1"></div>
  </section>

  <section class="card">
    <h2>Federated training, round by round</h2>
    <p id="curve-desc"></p>
    <div class="legend" id="legend2"></div>
    <svg id="curve" role="img" aria-label="Line chart of test AUC per federated round"></svg>
    <div class="tooltip" id="tip2"></div>
  </section>

  <section class="card" id="sites-card" hidden>
    <h2>Each hospital, on its own patients</h2>
    <p>After federated training, each hospital adapts the shared model to its own patients, on site, and keeps whichever version scores best in cross-validation on its own data. Log loss measures how trustworthy the predicted risk percentages are: lower is better.</p>
    <div class="legend" id="legend3"></div>
    <svg id="sites" role="img" aria-label="Dot plot of log loss on each hospital's own test patients: alone, shared model, personalized"></svg>
    <div class="tooltip" id="tip3"></div>
  </section>

  <section class="card">
    <details>
      <summary>Table view</summary>
      <table id="table"><thead><tr><th>Model</th><th class="num">ROC AUC</th><th class="num">Accuracy</th></tr></thead><tbody></tbody></table>
      <table id="sites-table" hidden style="margin-top:16px"><thead><tr><th>Hospital (own test patients)</th><th>Chose</th><th class="num">Alone</th><th class="num">Shared model</th><th class="num">Personalized</th></tr></thead><tbody></tbody></table>
    </details>
  </section>
</main>
<script>
const DATA = ${data};
const NS = 'http://www.w3.org/2000/svg';
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
const f3 = (x) => x.toFixed(3);
const COLORS = { local: 'var(--text-secondary)', federated: 'var(--series-1)', dropout: 'var(--series-2)', pooled: 'var(--text-primary)', personalized: 'var(--series-3)' };

document.getElementById('subtitle').textContent =
  'Logistic regression on the UCI Heart Disease data from Cleveland, Budapest, Switzerland and Long Beach VA. ' +
  DATA.testPatients + ' held-out test patients.';

function legend(id, items) {
  const box = document.getElementById(id);
  box.innerHTML = items.map(([label, color, kind]) =>
    kind === 'line'
      ? '<span><i class="line-swatch" style="border-color:' + color + '"></i>' + label + '</span>'
      : kind === 'dash'
      ? '<span><i class="line-swatch" style="border-top-style:dashed;border-color:' + color + '"></i>' + label + '</span>'
      : '<span><i class="swatch" style="background:' + color + '"></i>' + label + '</span>').join('');
}

function tooltip(tip, card, html, x, y) {
  tip.innerHTML = html;
  tip.style.opacity = 1;
  const r = card.getBoundingClientRect();
  const w = tip.offsetWidth;
  tip.style.left = Math.min(Math.max(x - r.left + 12, 4), r.width - w - 4) + 'px';
  tip.style.top = (y - r.top + 12) + 'px';
}

function drawDots() {
  const svg = document.getElementById('dots');
  svg.innerHTML = '';
  const W = svg.clientWidth || 800, top = 8, bottom = 28;
  // On narrow screens the label sits above its row instead of to the left.
  const compact = W < 560;
  const rowH = compact ? 48 : 34;
  const labelW = compact ? 0 : 230;
  const H = top + DATA.rows.length * rowH + bottom;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  const x0 = compact ? 8 : labelW + 8, x1 = W - 52;
  const lo = 0.5, hi = 1.0;
  const sx = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
  for (let t = 0.5; t <= 1.0001; t += 0.1) {
    el('line', { x1: sx(t), x2: sx(t), y1: top, y2: H - bottom, stroke: 'var(--grid)', 'stroke-width': 1 }, svg);
    el('text', { x: sx(t), y: H - 8, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = t.toFixed(1);
  }
  const tip = document.getElementById('tip1');
  const card = svg.parentElement;
  DATA.rows.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2 + (compact ? 8 : 0);
    const g = el('g', {}, svg);
    el('text', { x: compact ? x0 : labelW, y: compact ? cy - 12 : cy + 4, 'text-anchor': compact ? 'start' : 'end', style: r.group === 'local' ? '' : 'fill:var(--text-primary);font-weight:600' }, g).textContent = r.label;
    el('line', { x1: sx(lo), x2: sx(r.auc), y1: cy, y2: cy, stroke: 'var(--axis)', 'stroke-width': 2, 'stroke-linecap': 'round' }, g);
    el('circle', { cx: sx(r.auc), cy, r: 6, fill: COLORS[r.group], stroke: 'var(--surface-1)', 'stroke-width': 2 }, g);
    el('text', { x: sx(r.auc) + 12, y: cy + 4 }, g).textContent = f3(r.auc);
    const hit = el('rect', { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: 'transparent' }, g);
    hit.addEventListener('pointermove', (e) => tooltip(tip, card, '<b>' + r.label + '</b><br>ROC AUC ' + f3(r.auc) + '<br>Accuracy ' + (r.accuracy * 100).toFixed(1) + '%', e.clientX, e.clientY));
    hit.addEventListener('pointerleave', () => (tip.style.opacity = 0));
  });
  legend('legend1', [
    ['One hospital alone', COLORS.local], ['Federated', COLORS.federated],
    ...(DATA.curves.dropout ? [['Federated with a dropout', COLORS.dropout]] : []),
    ['Pooled data', COLORS.pooled],
  ]);
}

function drawCurve() {
  const svg = document.getElementById('curve');
  svg.innerHTML = '';
  const W = svg.clientWidth || 800, H = Math.max(240, Math.min(320, W * 0.45));
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  const m = { l: 44, r: 16, t: 24, b: 30 };
  const fed = DATA.curves.federated, drop = DATA.curves.dropout;
  const all = [...fed, ...(drop || [])].map((p) => p[1]).concat([DATA.pooledAuc, DATA.bestLocalAuc]);
  const lo = Math.floor((Math.min(...all) - 0.01) * 50) / 50, hi = Math.ceil((Math.max(...all) + 0.005) * 50) / 50;
  const R = fed.length;
  const sx = (r) => m.l + ((r - 1) / Math.max(R - 1, 1)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  for (let t = lo; t <= hi + 1e-9; t += 0.02) {
    el('line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), stroke: 'var(--grid)', 'stroke-width': 1 }, svg);
    el('text', { x: m.l - 8, y: sy(t) + 4, 'text-anchor': 'end', class: 'tick' }, svg).textContent = t.toFixed(2);
  }
  const step = R > 20 ? 5 : 2;
  for (let r = 1; r <= R; r++) {
    if (r !== 1 && r % step !== 0) continue;
    el('text', { x: sx(r), y: H - 10, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = r;
  }
  el('text', { x: W - m.r, y: H - 10 - 14, 'text-anchor': 'end', class: 'tick' }, svg).textContent = 'round';
  const ref = (v, label, color) => {
    el('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: color, 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }, svg);
    el('text', { x: W - m.r, y: sy(v) + 16, 'text-anchor': 'end' }, svg).textContent = label + ' ' + f3(v);
  };
  ref(DATA.pooledAuc, 'Pooled data', 'var(--text-primary)');
  ref(DATA.bestLocalAuc, 'Best hospital alone', 'var(--muted)');
  if (DATA.dropout) {
    const x = sx(DATA.dropout.round - 0.5);
    el('line', { x1: x, x2: x, y1: m.t - 8, y2: H - m.b, stroke: 'var(--axis)', 'stroke-width': 1.5 }, svg);
    const right = x > W * 0.6;
    el('text', { x: right ? x - 6 : x + 6, y: m.t - 10, 'text-anchor': right ? 'end' : 'start' }, svg).textContent = DATA.dropout.label + ' drops out';
  }
  const path = (pts) => pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ',' + sy(p[1]).toFixed(1)).join('');
  el('path', { d: path(fed), fill: 'none', stroke: 'var(--series-1)', 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
  if (drop) el('path', { d: path(drop.filter((p) => p[0] >= DATA.dropout.round - 1)), fill: 'none', stroke: 'var(--series-2)', 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);

  const cross = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--muted)', 'stroke-width': 1, opacity: 0 }, svg);
  const d1 = el('circle', { r: 4, fill: 'var(--series-1)', stroke: 'var(--surface-1)', 'stroke-width': 2, opacity: 0 }, svg);
  const d2 = el('circle', { r: 4, fill: 'var(--series-2)', stroke: 'var(--surface-1)', 'stroke-width': 2, opacity: 0 }, svg);
  const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
  const tip = document.getElementById('tip2');
  const card = svg.parentElement;
  hit.addEventListener('pointermove', (e) => {
    const box = svg.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const r = Math.min(R, Math.max(1, Math.round(1 + ((px - m.l) / (W - m.l - m.r)) * (R - 1))));
    cross.setAttribute('x1', sx(r)); cross.setAttribute('x2', sx(r)); cross.setAttribute('opacity', 1);
    d1.setAttribute('cx', sx(r)); d1.setAttribute('cy', sy(fed[r - 1][1])); d1.setAttribute('opacity', 1);
    let html = '<b>Round ' + r + '</b><br>Federated: ' + f3(fed[r - 1][1]);
    if (drop && r >= DATA.dropout.round) {
      d2.setAttribute('cx', sx(r)); d2.setAttribute('cy', sy(drop[r - 1][1])); d2.setAttribute('opacity', 1);
      html += '<br>With ' + DATA.dropout.label + ' gone: ' + f3(drop[r - 1][1]);
    } else d2.setAttribute('opacity', 0);
    tooltip(tip, card, html, e.clientX, e.clientY);
  });
  hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; [cross, d1, d2].forEach((n) => n.setAttribute('opacity', 0)); });

  legend('legend2', [
    ['Federated, all hospitals', 'var(--series-1)', 'line'],
    ...(drop ? [['Federated after ' + DATA.dropout.label + ' drops out', 'var(--series-2)', 'line']] : []),
    ['Pooled data', 'var(--text-primary)', 'dash'], ['Best hospital alone', 'var(--muted)', 'dash'],
  ]);
  document.getElementById('curve-desc').textContent =
    'ROC AUC of the shared model on all test patients after each round of train-locally, average-centrally.' +
    (DATA.dropout ? ' From round ' + DATA.dropout.round + ', ' + DATA.dropout.label + ' stops responding and the other three carry on.' : '');
}

// One row per hospital: its own model, the shared model and the personalized
// one, each a dot on a shared log-loss axis.
function drawSites() {
  if (!DATA.perSite) return;
  document.getElementById('sites-card').hidden = false;
  const svg = document.getElementById('sites');
  svg.innerHTML = '';
  const W = svg.clientWidth || 800, top = 8, bottom = 28;
  const compact = W < 560;
  const rowH = compact ? 76 : 48;
  const labelW = compact ? 0 : 230;
  const H = top + DATA.perSite.length * rowH + bottom;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  const x0 = compact ? 8 : labelW + 16, x1 = W - 52;
  const SERIES = [['alone', 'Hospital alone', COLORS.local], ['federated', 'Shared model', COLORS.federated], ['personalized', 'Personalized', COLORS.personalized]];
  const hi = Math.ceil(Math.max(...DATA.perSite.flatMap((r) => SERIES.map(([k]) => r[k].logLoss))) * 10) / 10;
  const sx = (v) => x0 + (v / hi) * (x1 - x0);
  for (let t = 0; t <= hi + 1e-9; t += 0.1) {
    el('line', { x1: sx(t), x2: sx(t), y1: top, y2: H - bottom, stroke: 'var(--grid)', 'stroke-width': 1 }, svg);
    el('text', { x: sx(t), y: H - 8, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = t.toFixed(1);
  }
  const tip = document.getElementById('tip3');
  const card = svg.parentElement;
  const pct = (x) => (x * 100).toFixed(1) + '%';
  DATA.perSite.forEach((r, i) => {
    const cy = top + i * rowH + rowH / 2 + (compact ? 10 : 0);
    const g = el('g', {}, svg);
    const tx = compact ? x0 : labelW, anchor = compact ? 'start' : 'end';
    el('text', { x: tx, y: compact ? cy - 22 : cy - 2, 'text-anchor': anchor, style: 'fill:var(--text-primary);font-weight:600' }, g).textContent = r.label;
    el('text', { x: tx, y: compact ? cy - 9 : cy + 13, 'text-anchor': anchor, class: 'tick' }, g).textContent = 'chose: ' + r.strategy;
    const vals = SERIES.map(([k]) => r[k].logLoss);
    el('line', { x1: sx(Math.min(...vals)), x2: sx(Math.max(...vals)), y1: cy, y2: cy, stroke: 'var(--axis)', 'stroke-width': 2, 'stroke-linecap': 'round' }, g);
    // Personalized is drawn last so it sits on top when it equals the shared model.
    for (const [k, , color] of SERIES) el('circle', { cx: sx(r[k].logLoss), cy, r: 6, fill: color, stroke: 'var(--surface-1)', 'stroke-width': 2 }, g);
    // Label only the personalized value, next to its own dot (below it on narrow
    // screens, where the hospital name and choice sit above the row).
    el('text', { x: sx(r.personalized.logLoss), y: compact ? cy + 20 : cy - 12, 'text-anchor': 'middle' }, g).textContent = f3(r.personalized.logLoss);
    const hit = el('rect', { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: 'transparent' }, g);
    hit.addEventListener('pointermove', (e) => tooltip(tip, card,
      '<b>' + r.label + '</b> (' + r.test + ' test patients)<br>Chose: ' + r.strategy +
      SERIES.map(([k, label]) => '<br>' + label + ': log loss ' + f3(r[k].logLoss) + ', accuracy ' + pct(r[k].accuracy)).join(''),
      e.clientX, e.clientY));
    hit.addEventListener('pointerleave', () => (tip.style.opacity = 0));
  });
  legend('legend3', SERIES.map(([, label, color]) => [label, color]));
}

function fillTable() {
  document.querySelector('#table tbody').innerHTML = DATA.rows.map((r) =>
    '<tr><td>' + r.label + '</td><td class="num">' + f3(r.auc) + '</td><td class="num">' + (r.accuracy * 100).toFixed(1) + '%</td></tr>').join('');
  if (!DATA.perSite) return;
  const cell = (m) => '<td class="num">' + f3(m.logLoss) + ' / ' + (m.accuracy * 100).toFixed(1) + '%</td>';
  const t = document.getElementById('sites-table');
  t.hidden = false;
  t.querySelector('tbody').innerHTML = DATA.perSite.map((r) =>
    '<tr><td>' + r.label + '</td><td>' + r.strategy + '</td>' + cell(r.alone) + cell(r.federated) + cell(r.personalized) + '</tr>').join('') +
    '<tr><td colspan="5" style="color:var(--muted)">log loss / accuracy</td></tr>';
}

function draw() { drawDots(); drawCurve(); drawSites(); }
fillTable();
draw();
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(draw, 100); });
</script>
</body>
</html>
`;
}
