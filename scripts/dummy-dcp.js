// Dummy DCP: the coordinator plus four worker processes, one per hospital folder.
// Each worker (scripts/dummy-dcp-worker.js) runs inside its own folder, reads that
// hospital's dataset, trains on it, and sends back only weights, which this
// process averages into the shared model. One page shows hospitals, logs,
// results and the learned weights, with Start and Clear & restart buttons.
// Secure aggregation and differential privacy can be switched on from the page
// (or with --secure / --dp), and at the end each hospital personalizes the
// shared model on site.
//   npm run dummy-dcp [-- --root ~/Desktop --va "~/Desktop/long beach" --rounds 30 --delay 300 --port 8790 --no-open]
//   npm run dummy-dcp -- --secure --dp --noise 3 --epsilon 5
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync, statSync } from 'node:fs';
import { fork, spawn } from 'node:child_process';
import { homedir } from 'node:os';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';
import { federatedTrain } from '../src/coordinator.js';
import { createHub } from '../src/hub.js';
import { SITES, SITE_IDS, publicStandardization } from '../src/data.js';
import { DEFAULT_ROUNDS } from '../src/localTrain.js';
import { parseHyperparameters, printTable, fmt, DP_HYPERPARAMETERS } from './cli.js';

// Default folder name for each hospital, inside --root (the Desktop by default).
const DEFAULT_FOLDERS = { cleveland: 'cleveland', hungarian: 'hungarian', switzerland: 'switzerland', va: 'long beach' };

const { hp, seed, values } = parseHyperparameters({
  root: { type: 'string' },
  ...Object.fromEntries(SITE_IDS.map((id) => [id, { type: 'string' }])),
  port: { type: 'string' },
  delay: { type: 'string' },
  'min-participants': { type: 'string' },
  'round-timeout': { type: 'string' },
  'no-open': { type: 'boolean' },
  secure: { type: 'boolean' },
  dp: { type: 'boolean' },
  noise: { type: 'string' },
  clip: { type: 'string' },
  epsilon: { type: 'string' },
});
const port = Number(values.port ?? 8790);
const delay = Number(values.delay ?? 300);
const minParticipants = Number(values['min-participants'] ?? 2);
const roundTimeoutMs = Number(values['round-timeout'] ?? 30_000);

// Privacy settings for the next run, changeable from the page between runs.
// DP is each hospital's own setting; here the page stands in for the
// hospitals' admins and sends each worker its settings before a run.
let settings = {
  secure: Boolean(values.secure),
  dp: Boolean(values.dp),
  noiseMultiplier: Number(values.noise ?? 3),
  clipNorm: Number(values.clip ?? 1),
  epsilonBudget: values.epsilon === undefined ? null : Number(values.epsilon),
};
const SECURE_MIN = 3;

/** Training settings for a run: under DP, fewer rounds with a larger step (flags still win). */
function runConfig(s = settings) {
  const runHp = s.dp ? { ...DP_HYPERPARAMETERS, ...hp } : hp;
  return { hp: runHp, rounds: runHp.rounds ?? DEFAULT_ROUNDS };
}
const expandHome = (p) => (p.startsWith('~') ? join(homedir(), p.slice(1)) : p);
const rootFolder = resolve(expandHome(values.root ?? join(homedir(), 'Desktop')));
const folders = Object.fromEntries(
  SITE_IDS.map((id) => [id, values[id] ? resolve(expandHome(values[id])) : join(rootFolder, DEFAULT_FOLDERS[id])]),
);

const root = fileURLToPath(new URL('..', import.meta.url));
const pagesDir = join(root, 'dummy-dcp');
const srcDir = join(root, 'src');
const workerScript = join(root, 'scripts', 'dummy-dcp-worker.js');
const resultsFile = join(root, 'results', 'dummy-dcp-model.json');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };

// ---- state the page sees ------------------------------------------------------

let phase = 'loading'; // loading -> ready -> training -> done
let rows = []; // one per finished round: { round, responded, dropped, skipped, attempts, trainLoss, evals, mean }
let final = null;
let current = null; // settings and training config of the latest run
// What the coordinator received from each hospital in the latest round: the
// first few numbers, so the page can show the difference masking makes.
let received = {};
let personalization = null; // site -> { strategy, label, shared, personalized }
// The latest round's weights: the averaged global model, plus each responding
// hospital's own update before averaging. { round, global, hospitals: { site: weights } }
let weights = null;
let latestUpdates = {};
let logs = [];
let runId = 0;
const workers = {}; // site -> { child, state: 'starting' | 'ready' | 'failed' | 'stopped', summary, error, privacy }

const pages = new Set();
const hub = createHub({ onEvent: () => broadcastState() });

function snapshot() {
  const status = Object.fromEntries(hub.status().map((s) => [s.site, s]));
  const shown = phase === 'training' || phase === 'done' ? current : { settings, ...runConfig() };
  return {
    phase,
    config: { rounds: shown.rounds, delay, minParticipants, roundTimeoutMs, hyperparameters: shown.hp, settings, secureMin: SECURE_MIN },
    run: current && { settings: current.settings, rounds: current.rounds },
    hospitals: SITE_IDS.map((id) => ({
      site: id,
      label: SITES[id].label,
      folder: folders[id],
      worker: workers[id]?.state ?? 'stopped',
      error: workers[id]?.error ?? null,
      summary: workers[id]?.summary ? { ...workers[id].summary, publicKey: undefined } : null,
      privacy: workers[id]?.privacy ?? null,
      online: status[id]?.online ?? true,
      available: hub.available().includes(id),
    })),
    rows,
    final,
    weights,
    received,
    personalization,
    logs: logs.slice(-300),
  };
}

function sse(res) {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const keepAlive = setInterval(() => res.write(': keep-alive\n\n'), 15_000);
  res.on('close', () => clearInterval(keepAlive));
  return (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function broadcastState() {
  if (phase === 'loading' || phase === 'ready') phase = hub.available().length >= minParticipants ? 'ready' : 'loading';
  const state = snapshot();
  for (const send of pages) send('state', state);
}

/** Add a line to the page's log. Coordinator lines also go to the terminal. */
function log(source, text, level = 'info') {
  const entry = { t: Date.now(), source, level, text };
  logs.push(entry);
  if (logs.length > 1000) logs = logs.slice(-1000);
  if (source === 'coordinator') console.log(`  ${text}`);
  for (const send of pages) send('log', entry);
}

// ---- workers ------------------------------------------------------------------

function startWorker(site) {
  const folder = folders[site];
  const label = SITES[site].label;
  if (!existsSync(folder) || !statSync(folder).isDirectory()) {
    workers[site] = { state: 'failed', error: `Folder not found: ${folder}` };
    log('coordinator', `${label}: folder not found (${folder}). Pass --${site} <folder> to point at it.`, 'error');
    return;
  }
  const child = fork(workerScript, [site, folder], { cwd: folder, env: { ...process.env, FEDHEART_SEED: String(seed) } });
  const w = (workers[site] = { child, state: 'starting', summary: null, error: null });
  const send = (task) => child.send(task);
  hub.connect(site, send);
  log('coordinator', `${label}: started worker (pid ${child.pid}) in ${folder}`);

  child.on('message', (msg) => {
    if (msg.type === 'log') log(site, msg.text, msg.level);
    else if (msg.type === 'ready') {
      Object.assign(w, { state: 'ready', summary: msg.summary });
      hub.ready(site, msg.summary);
    } else if (msg.type === 'failed') {
      Object.assign(w, { state: 'failed', error: msg.error });
      broadcastState();
    } else if (msg.type === 'privacy') {
      w.privacy = msg.privacy;
      broadcastState();
    } else if (msg.type === 'result') hub.resolve(msg.id, msg.result, msg.error);
  });
  child.on('exit', (code) => {
    if (workers[site] !== w) return; // replaced by a restart
    w.state = w.state === 'failed' ? 'failed' : 'stopped';
    hub.disconnect(site, send);
    if (!w.stopping) log('coordinator', `${label}: worker exited (code ${code})`, 'error');
  });
}

function stopWorkers() {
  for (const w of Object.values(workers)) {
    if (!w.child || w.child.exitCode !== null) continue;
    w.stopping = true;
    w.child.kill();
  }
  for (const id of SITE_IDS) hub.disconnect(id);
}

// ---- training -----------------------------------------------------------------

class Cancelled extends Error {}

function start() {
  const id = ++runId;
  run(id).catch((err) => {
    if (err instanceof Cancelled || id !== runId) return;
    log('coordinator', `Training failed: ${err.message}`, 'error');
    phase = 'ready';
    broadcastState();
  });
}

async function run(id) {
  current = { settings: { ...settings }, ...runConfig() };
  const { hp: runHp, rounds } = current;
  const { secure, dp } = current.settings;
  rows = [];
  final = null;
  weights = null;
  latestUpdates = {};
  received = {};
  personalization = null;
  phase = 'training';
  const joined = hub.available().map((s) => SITES[s].label);
  const modes = [secure && 'secure aggregation', dp && `differential privacy (noise ${current.settings.noiseMultiplier})`].filter(Boolean);
  log('coordinator', `Starting training: ${rounds} rounds with ${joined.length}/4 hospitals (${joined.join(', ')})` +
    (modes.length ? `, with ${modes.join(' and ')}` : ''));
  broadcastState();

  // Each hospital gets its own privacy settings before the run.
  const privacy = dp
    ? { noiseMultiplier: current.settings.noiseMultiplier, clipNorm: current.settings.clipNorm, epsilonBudget: current.settings.epsilonBudget }
    : null;
  await Promise.allSettled(hub.available().map((site) => hub.configure(site, { privacy })));
  if (dp) log('coordinator', 'Differential privacy: features are standardized with public clinical ranges, so no feature statistics are requested.');
  if (secure) log('coordinator', "Secure aggregation: hospitals mask their updates with secrets agreed from each other's public keys. Only the sums are visible here.");

  let previouslyDropped = new Set();
  const startedAt = Date.now();
  const model = await federatedTrain(
    { ...runHp, rounds },
    {
      participants: SITE_IDS.map((site) => hub.participant(site)),
      minParticipants,
      roundTimeoutMs,
      secureAggregation: secure ? { minCohort: SECURE_MIN } : false,
      standardization: dp ? publicStandardization() : undefined,
      onResponse: (site, value, context) => {
        if (!context.round) return; // round 0 is the feature statistics
        received[site] = {
          round: context.round,
          attempt: context.attempt ?? 1,
          masked: Boolean(value.masked),
          preview: value.masked ? value.masked.slice(0, 3) : value.weights.slice(0, 3),
          count: (value.masked ?? value.weights).length,
        };
        // Individual updates are only visible without secure aggregation.
        if (!value.masked) latestUpdates[site] = value;
      },
      onRound: async (entry, latest) => {
        if (id !== runId) throw new Cancelled();
        const dropped = new Set(entry.dropped.map((d) => d.name));
        for (const d of entry.dropped) {
          if (!previouslyDropped.has(d.name)) log('coordinator', `${SITES[d.name].label} dropped out (${d.reason})`, 'warn');
        }
        for (const s of previouslyDropped) if (!dropped.has(s)) log('coordinator', `${SITES[s].label} rejoined`);
        previouslyDropped = dropped;

        const evals = await evaluateEverywhere(latest);
        if (id !== runId) throw new Cancelled();
        const mean = weightedMean(evals);
        rows.push({ ...entry, evals, mean });
        weights = {
          round: entry.round,
          global: latest.weights,
          secure,
          hospitals: secure ? {} : Object.fromEntries(entry.responded.map((s) => [s, latestUpdates[s].weights])),
        };
        log(
          'coordinator',
          `Round ${entry.round}/${rounds}: averaged ${entry.responded.length}/4 hospitals` +
            (secure && !entry.skipped ? ' (only their sum was visible)' : '') +
            (entry.attempts > 1 ? `, after re-running without ${entry.dropped.map((d) => SITES[d.name].label).join(', ')}` : '') +
            (entry.skipped ? ` (skipped: too few hospitals${secure ? `; secure aggregation needs ${SECURE_MIN}` : ''})` : '') +
            `. Holdout AUC ${fmt(mean.auc)}, accuracy ${pct(mean.accuracy)}`,
          entry.skipped ? 'warn' : 'info',
        );
        broadcastState();
        if (delay) await sleep(delay);
      },
    },
  );
  if (id !== runId) throw new Cancelled();

  // Each hospital adapts the shared model to its own patients and keeps it.
  log('coordinator', 'Asking each hospital to personalize the shared model on site.');
  broadcastState();
  const ids = hub.available();
  const settled = await Promise.allSettled(ids.map((site) => hub.personalize(site, model, runHp)));
  if (id !== runId) throw new Cancelled();
  personalization = {};
  settled.forEach((r, i) => {
    if (r.status === 'fulfilled') personalization[ids[i]] = r.value;
  });

  const evals = rows.at(-1)?.evals ?? {};
  const personalizedMean = weightedMean(Object.fromEntries(Object.entries(personalization).map(([s, p]) => [s, p.personalized])));
  const sharedMean = weightedMean(Object.fromEntries(Object.entries(personalization).map(([s, p]) => [s, p.shared])));
  final = {
    evals,
    mean: weightedMean(evals),
    personalized: { shared: sharedMean, personalized: personalizedMean },
    settings: current.settings,
    file: 'results/dummy-dcp-model.json',
    seconds: (Date.now() - startedAt) / 1000,
  };
  phase = 'done';
  await mkdir(join(root, 'results'), { recursive: true });
  await writeFile(resultsFile, JSON.stringify({ ...model, hyperparameters: { ...runHp, rounds }, settings: current.settings }, null, 2));

  console.log("\nFinal model on each hospital's own held-out patients:\n");
  printTable(
    [
      ...SITE_IDS.map((s) => {
        const p = personalization[s];
        const eps = workers[s]?.privacy?.spent?.epsilon;
        return {
          hospital: SITES[s].label,
          patients: evals[s]?.n ?? '-',
          'ROC AUC': fmt(evals[s]?.auc),
          accuracy: pct(evals[s]?.accuracy),
          personalized: p ? `${p.label}: log loss ${fmt(p.shared.logLoss)} -> ${fmt(p.personalized.logLoss)}` : '-',
          ...(dp ? { epsilon: Number.isFinite(eps) ? eps.toFixed(2) : '-' } : {}),
        };
      }),
      { hospital: 'All (patient-weighted)', patients: final.mean.n || '-', 'ROC AUC': fmt(final.mean.auc), accuracy: pct(final.mean.accuracy),
        personalized: `log loss ${fmt(sharedMean.logLoss)} -> ${fmt(personalizedMean.logLoss)}`, ...(dp ? { epsilon: '' } : {}) },
    ],
    ['hospital', 'patients', 'ROC AUC', 'accuracy', 'personalized', ...(dp ? ['epsilon'] : [])],
  );
  console.log('');
  log('coordinator', `Training complete. Holdout AUC ${fmt(final.mean.auc)}, accuracy ${pct(final.mean.accuracy)}. ` +
    `Personalized: log loss ${fmt(sharedMean.logLoss)} -> ${fmt(personalizedMean.logLoss)}. Saved ${final.file}`);
  broadcastState();
}

/** Clear results and logs, restart every worker so it re-reads its folder. */
async function clearAndRestart() {
  runId++; // cancels a run in progress at its next checkpoint
  stopWorkers();
  phase = 'loading';
  rows = [];
  final = null;
  weights = null;
  latestUpdates = {};
  received = {};
  personalization = null;
  current = null;
  logs = [];
  for (const send of pages) send('clear', {});
  await sleep(200);
  log('coordinator', 'Cleared. Restarting the hospital workers.');
  for (const site of SITE_IDS) startWorker(site);
  broadcastState();
}

/** Each available hospital scores the global model on its own holdout. */
async function evaluateEverywhere(model) {
  const ids = hub.available();
  const settled = await Promise.allSettled(ids.map((s) => hub.evaluate(s, model)));
  const out = {};
  settled.forEach((s, i) => {
    if (s.status === 'fulfilled') out[ids[i]] = s.value;
  });
  return out;
}

function weightedMean(evals) {
  const list = Object.values(evals).filter((e) => e.n > 0);
  const n = list.reduce((s, e) => s + e.n, 0);
  const avg = (key) => {
    const ok = list.filter((e) => Number.isFinite(e[key]));
    const m = ok.reduce((s, e) => s + e.n, 0);
    return m ? ok.reduce((s, e) => s + e[key] * e.n, 0) / m : NaN;
  };
  return { n, auc: avg('auc'), accuracy: avg('accuracy'), logLoss: avg('logLoss') };
}

const pct = (x) => (Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : 'n/a');

// ---- HTTP ---------------------------------------------------------------------

async function body(req) {
  let text = '';
  for await (const chunk of req) text += chunk;
  return text ? JSON.parse(text) : {};
}

async function serveFile(res, dir, path) {
  const file = normalize(join(dir, path));
  if (!file.startsWith(dir)) throw new Error('outside root');
  const content = await readFile(file);
  res.writeHead(200, { 'content-type': types[extname(file)] ?? 'text/plain' }).end(content);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (req.method === 'GET' && url.pathname === '/events') {
      const send = sse(res);
      pages.add(send);
      res.on('close', () => pages.delete(send));
      send('state', snapshot());
      return;
    }
    if (req.method === 'POST' && url.pathname === '/start') {
      const n = hub.available().length;
      if (phase === 'training') return res.writeHead(409).end('Training is already running');
      if (n < minParticipants) return res.writeHead(409).end(`Needs at least ${minParticipants} hospitals with data, ${n} ready`);
      if (settings.secure && n < SECURE_MIN) return res.writeHead(409).end(`Secure aggregation needs at least ${SECURE_MIN} hospitals, ${n} ready`);
      start();
      return res.writeHead(204).end();
    }
    if (req.method === 'POST' && url.pathname === '/settings') {
      if (phase === 'training') return res.writeHead(409).end('Settings can change between runs, not during one');
      const next = { ...settings, ...(await body(req)) };
      const ok =
        typeof next.secure === 'boolean' && typeof next.dp === 'boolean' &&
        next.noiseMultiplier > 0 && next.clipNorm > 0 && (next.epsilonBudget === null || next.epsilonBudget > 0);
      if (!ok) return res.writeHead(400).end('Invalid settings');
      settings = { secure: next.secure, dp: next.dp, noiseMultiplier: Number(next.noiseMultiplier), clipNorm: Number(next.clipNorm), epsilonBudget: next.epsilonBudget === null ? null : Number(next.epsilonBudget) };
      log('coordinator', `Next run: secure aggregation ${settings.secure ? 'on' : 'off'}, differential privacy ${settings.dp ? `on (noise ${settings.noiseMultiplier}${settings.epsilonBudget ? `, budget epsilon ${settings.epsilonBudget}` : ''})` : 'off'}`);
      broadcastState();
      return res.writeHead(204).end();
    }
    if (req.method === 'POST' && url.pathname === '/restart') {
      await clearAndRestart();
      return res.writeHead(204).end();
    }
    if (req.method === 'POST' && url.pathname === '/online') {
      const site = url.searchParams.get('site');
      if (!SITES[site]) return res.writeHead(400).end('unknown site');
      const { online } = await body(req);
      hub.setOnline(site, online);
      log('coordinator', `${SITES[site].label} switched ${online ? 'online' : 'offline'}`, online ? 'info' : 'warn');
      return res.writeHead(204).end();
    }
    if (req.method === 'GET' && url.pathname.startsWith('/src/')) {
      return await serveFile(res, srcDir, url.pathname.slice('/src/'.length));
    }
    if (req.method === 'GET') {
      return await serveFile(res, pagesDir, url.pathname === '/' ? 'index.html' : url.pathname.slice(1));
    }
    res.writeHead(404).end('Not found');
  } catch (err) {
    if (!res.headersSent) res.writeHead(err.code === 'ENOENT' || /outside root/.test(err.message) ? 404 : 400).end(String(err.message));
  }
});

function openInBrowser(url) {
  const [cmd, args] =
    process.platform === 'darwin' ? ['open', [url]] : process.platform === 'win32' ? ['cmd', ['/c', 'start', '', url]] : ['xdg-open', [url]];
  spawn(cmd, args, { stdio: 'ignore', detached: true }).on('error', () => {}).unref();
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    stopWorkers();
    process.exit(0);
  });
}

server.listen(port, () => {
  const url = `http://localhost:${port}/`;
  console.log('FedHeart dummy DCP: one worker process per hospital folder; only weights come back here.\n');
  for (const id of SITE_IDS) console.log(`  ${SITES[id].label.padEnd(15)}${folders[id]}`);
  console.log(`\n  Open ${url} and press Start.\n`);
  for (const site of SITE_IDS) startWorker(site);
  if (!values['no-open']) openInBrowser(url);
});
