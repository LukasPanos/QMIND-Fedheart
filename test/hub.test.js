import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { federatedTrain } from '../src/coordinator.js';
import { createLocalParticipant } from '../src/participant.js';
import { createHub } from '../src/hub.js';
import { loadDatasetText } from '../src/hospital.js';
import { createHospitalWorker } from '../src/hospitalWorker.js';
import { createKeyAgreement } from '../src/secureAggregation.js';
import { loadAllSites, SITES, SITE_IDS, DATA_DIR, publicStandardization } from '../src/data.js';

const sites = loadAllSites();

// Stands in for a worker process (scripts/dummy-dcp-worker.js): tasks arrive as
// JSON (like IPC messages) and answers go back as JSON, asynchronously.
function connectWorker(hub, site, data, { log } = {}) {
  const worker = createHospitalWorker(site, data);
  const send = (task) => {
    const t = JSON.parse(JSON.stringify(task));
    setImmediate(async () => {
      try {
        const { result } = await worker.handle(t);
        const copy = JSON.parse(JSON.stringify(result));
        log?.push(copy);
        hub.resolve(t.id, copy);
      } catch (err) {
        hub.resolve(t.id, undefined, err.message);
      }
    });
  };
  hub.connect(site, send);
  hub.ready(site, { rows: data.train.y.length, publicKey: worker.publicKey });
  return send;
}
const connectAll = (hub, opts) => Object.fromEntries(SITE_IDS.map((id) => [id, connectWorker(hub, id, sites[id], opts)]));
const close = (a, b, tol) => a.forEach((x, j) => assert.ok(Math.abs(x - b[j]) < tol, `[${j}] ${x} vs ${b[j]}`));

test('worker hospitals train exactly like local participants', async () => {
  const hub = createHub();
  for (const id of SITE_IDS) connectWorker(hub, id, sites[id]);
  const viaHub = await federatedTrain({ rounds: 3 }, { participants: SITE_IDS.map((id) => hub.participant(id)) });
  const local = await federatedTrain(
    { rounds: 3 },
    { participants: SITE_IDS.map((id) => createLocalParticipant(id, sites[id].train)) },
  );
  assert.deepEqual(viaHub.weights, local.weights);
});

test('a worker that disconnects mid-run is dropped and training continues', async () => {
  const hub = createHub();
  const sends = Object.fromEntries(SITE_IDS.map((id) => [id, connectWorker(hub, id, sites[id])]));
  const { history, weights } = await federatedTrain(
    { rounds: 4 },
    {
      participants: SITE_IDS.map((id) => hub.participant(id)),
      onRound: (entry) => {
        if (entry.round === 2) hub.disconnect('va', sends.va);
      },
    },
  );
  assert.equal(history[1].responded.length, 4);
  for (const h of history.slice(2)) {
    assert.deepEqual(h.dropped.map((d) => d.name), ['va']);
    assert.match(h.dropped[0].reason, /not connected/);
  }
  assert.ok(weights.every(Number.isFinite));
});

test('going offline drops a hospital and coming back online rejoins', async () => {
  const hub = createHub();
  for (const id of SITE_IDS) connectWorker(hub, id, sites[id]);
  const { history } = await federatedTrain(
    { rounds: 3 },
    {
      participants: SITE_IDS.map((id) => hub.participant(id)),
      onRound: (entry) => {
        if (entry.round === 1) hub.setOnline('cleveland', false);
        if (entry.round === 2) hub.setOnline('cleveland', true);
      },
    },
  );
  assert.deepEqual(history[1].dropped.map((d) => d.name), ['cleveland']);
  assert.equal(history[2].responded.length, 4);
});

test('workers only ever send aggregates, weights and holdout metrics', async () => {
  const hub = createHub();
  const log = [];
  for (const id of SITE_IDS) connectWorker(hub, id, sites[id], { log });
  const model = await federatedTrain({ rounds: 2 }, { participants: SITE_IDS.map((id) => hub.participant(id)) });
  for (const id of SITE_IDS) await hub.evaluate(id, model);
  for (const id of SITE_IDS) await hub.personalize(id, model);
  const allowed = ['count', 'sum', 'sumSq', 'weights', 'numSamples', 'loss', 'n', 'auc', 'accuracy', 'logLoss', 'strategy', 'label', 'shared', 'personalized'];
  assert.ok(log.length > 0);
  for (const r of log) for (const key of Object.keys(r)) assert.ok(allowed.includes(key), key);
});

test('a worker parses its file the same way loadAllSites does', () => {
  const text = readFileSync(`${DATA_DIR}/${SITES.va.file}`, 'utf8');
  const { train, test: holdout, rows } = loadDatasetText(text);
  assert.deepEqual(train, sites.va.train);
  assert.deepEqual(holdout, sites.va.test);
  assert.equal(rows, 200);
  assert.throws(() => loadDatasetText('1,2,3\n'), /No patients found/);
  assert.throws(() => loadDatasetText('1,2,3,4,5,6,7,8,9,10,11,12,13,14\n1,2\n'), /Expected 14 columns/);
});

test("UCI's original 76-attribute files parse to the same patients as the processed files", () => {
  // Rebuild va's original-format file from the processed one: each patient is 76
  // whitespace-separated values over several lines, -9 for missing, ending in "name".
  const positions = [3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51, 58];
  const processed = readFileSync(`${DATA_DIR}/${SITES.va.file}`, 'utf8').trim().split('\n');
  const record = (line) => {
    const values = new Array(76).fill('-9');
    values[0] = '1';
    values[75] = 'name';
    line.split(',').forEach((v, i) => { values[positions[i] - 1] = v === '?' ? '-9' : v; });
    return [0, 7, 15, 23, 31, 39, 47, 55, 63, 71].map((start, k, all) => values.slice(start, all[k + 1] ?? 76).join(' ')).join('\n');
  };
  const corrupt = '1 2 3 garbled name';
  const text = [...processed.map(record), corrupt].join('\n');
  const { train, test: holdout, rows, skipped, format } = loadDatasetText(text);
  assert.equal(format, 'original UCI (76 attributes)');
  assert.equal(rows, 200);
  assert.equal(skipped, 1);
  assert.deepEqual(train, sites.va.train);
  assert.deepEqual(holdout, sites.va.test);
});

test('the hospital modules stay free of Node built-ins', () => {
  const portable = ['features.js', 'localTrain.js', 'model.js', 'hospital.js', 'participant.js', 'privacy.js', 'personalization.js'];
  for (const file of portable) {
    const src = readFileSync(new URL(`../src/${file}`, import.meta.url), 'utf8');
    assert.doesNotMatch(src, /from 'node:/, file);
    for (const [, dep] of src.matchAll(/from '\.\/([\w.]+)'/g)) assert.ok(portable.includes(dep), `${file} imports ${dep}`);
  }
});

test('two hospitals derive the same secret from each other\'s public keys, and nobody else can', () => {
  const [a, b, c] = [createKeyAgreement(), createKeyAgreement(), createKeyAgreement()];
  assert.equal(a.secretWith(b.publicKey), b.secretWith(a.publicKey));
  assert.notEqual(a.secretWith(b.publicKey), a.secretWith(c.publicKey));
  assert.match(a.secretWith(b.publicKey), /^[0-9a-f]{64}$/);
});

test('secure aggregation through worker hospitals: same model, and the coordinator only sees masks', async () => {
  const plainHub = createHub();
  connectAll(plainHub);
  const plain = await federatedTrain({ rounds: 3 }, { participants: SITE_IDS.map((id) => plainHub.participant(id)) });

  const hub = createHub();
  connectAll(hub);
  const seen = [];
  const secure = await federatedTrain(
    { rounds: 3 },
    { participants: SITE_IDS.map((id) => hub.participant(id)), secureAggregation: true, onResponse: (name, value) => seen.push(value) },
  );
  close(secure.weights, plain.weights, 1e-5);
  assert.equal(seen.length, SITE_IDS.length * 4); // stats + 3 rounds
  for (const v of seen) assert.deepEqual(Object.keys(v), ['masked']);
});

test('long-lived worker hospitals can train again under secure aggregation, with fresh masks', async () => {
  const hub = createHub();
  connectAll(hub);
  const opts = (seen) => ({
    participants: SITE_IDS.map((id) => hub.participant(id)),
    secureAggregation: true,
    onResponse: (name, value, context) => { if (context.round === 1) seen[name] = value.masked[0]; },
  });
  const a = {};
  const b = {};
  const first = await federatedTrain({ rounds: 2 }, opts(a));
  const again = await federatedTrain({ rounds: 2 }, opts(b));
  assert.ok(again.history.every((h) => !h.skipped), 'the second run is not refused');
  close(again.weights, first.weights, 1e-5);
  for (const id of SITE_IDS) assert.notEqual(a[id], b[id], `${id} reused a mask across runs`);
});

test('secure aggregation through worker hospitals survives a hospital going offline', async () => {
  const hub = createHub();
  connectAll(hub);
  const { history, weights } = await federatedTrain(
    { rounds: 3 },
    {
      participants: SITE_IDS.map((id) => hub.participant(id)),
      secureAggregation: true,
      onRound: (entry) => { if (entry.round === 1) hub.setOnline('switzerland', false); },
    },
  );
  assert.equal(history[1].responded.length, 3);
  assert.equal(history[1].skipped, false);
  assert.deepEqual(history[1].dropped.map((d) => d.name), ['switzerland']);
  assert.ok(weights.every(Number.isFinite));
});

test('differential privacy through worker hospitals: each tracks its epsilon across runs', async () => {
  const hub = createHub();
  connectAll(hub);
  const privacy = { noiseMultiplier: 3, clipNorm: 1, epsilonBudget: Infinity };
  const status = await hub.configure('va', { privacy });
  assert.equal(status.dp, true);
  for (const id of SITE_IDS.filter((s) => s !== 'va')) await hub.configure(id, { privacy });
  const opts = { participants: SITE_IDS.map((id) => hub.participant(id)), standardization: publicStandardization(), secureAggregation: true };
  const first = await federatedTrain({ rounds: 2 }, opts);
  assert.ok(first.history.every((h) => !h.skipped && Number.isNaN(h.trainLoss)));

  const log = [];
  const worker = createHospitalWorker('va', sites.va);
  await worker.handle({ type: 'configure', privacy });
  const out = await worker.handle({ type: 'train', weights: first.weights, hyperparameters: { localEpochs: 2, batchSize: 16 }, round: 1, standardization: publicStandardization() });
  log.push(out.log);
  const after1 = out.privacy.spent.epsilon;
  await worker.handle({ type: 'configure', privacy }); // a new run on the same data
  const out2 = await worker.handle({ type: 'train', weights: first.weights, hyperparameters: { localEpochs: 2, batchSize: 16 }, round: 1, standardization: publicStandardization() });
  assert.ok(out2.privacy.spent.epsilon > after1, 'privacy loss adds up across runs');
  assert.match(log[0], /with DP noise/);

  await worker.handle({ type: 'configure', privacy: null });
  const out3 = await worker.handle({ type: 'train', weights: first.weights, hyperparameters: {}, round: 1, standardization: publicStandardization() });
  assert.equal(out3.privacy.trainedWithoutDp, true);
  await assert.rejects(worker.handle({ type: 'configure', privacy }).then(() => worker.handle({ type: 'stats' })), /does not release exact feature statistics/);
});

test('worker hospitals personalize on site and report only scores', async () => {
  const hub = createHub();
  connectAll(hub);
  const model = await federatedTrain({ rounds: 5 }, { participants: SITE_IDS.map((id) => hub.participant(id)) });
  const swiss = await hub.personalize('switzerland', model);
  assert.deepEqual(Object.keys(swiss).sort(), ['label', 'personalized', 'shared', 'strategy']);
  assert.ok(swiss.personalized.logLoss < swiss.shared.logLoss, JSON.stringify(swiss));
  const shared = await hub.evaluate('switzerland', model);
  assert.deepEqual(swiss.shared, shared, 'the shared score matches the regular holdout evaluation');
});
