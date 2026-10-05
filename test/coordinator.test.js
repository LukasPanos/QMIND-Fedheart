import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fedAvg, federatedTrain } from '../src/coordinator.js';
import { createLocalParticipant, withDropout } from '../src/participant.js';
import { createDcpParticipant } from '../src/dcp.js';
import { loadAllSites, SITE_IDS } from '../src/data.js';
import { evaluate, concatData } from '../src/model.js';
import { trainLocalOnly } from '../src/baselines.js';

const sites = loadAllSites();
const participants = () => SITE_IDS.map((id) => createLocalParticipant(id, sites[id].train));
const test_ = concatData(SITE_IDS.map((id) => sites[id].test));

test('fedAvg is a sample-weighted average', () => {
  const out = fedAvg([
    { weights: [0, 10], numSamples: 1 },
    { weights: [4, 2], numSamples: 3 },
  ]);
  assert.deepEqual(out, [3, 4]);
});

test('federated model beats every hospital training alone', async () => {
  const model = await federatedTrain({ rounds: 20 }, { participants: participants() });
  const fed = evaluate(model, test_).auc;
  for (const id of SITE_IDS) {
    const local = evaluate(trainLocalOnly(sites[id].train, { rounds: 20 }), test_).auc;
    assert.ok(fed > local, `federated ${fed} vs ${id} ${local}`);
  }
});

test('coordinator only receives weights, never rows', async () => {
  const seen = [];
  const spy = participants().map((p) => ({
    ...p,
    async stats() { const s = await p.stats(); seen.push(s); return s; },
    async train(...args) { const r = await p.train(...args); seen.push(r); return r; },
  }));
  await federatedTrain({ rounds: 2 }, { participants: spy });
  for (const s of seen) {
    for (const key of Object.keys(s)) assert.ok(['count', 'sum', 'sumSq', 'weights', 'numSamples', 'loss'].includes(key), key);
  }
});

test('training continues when a hospital drops out (error)', async () => {
  const ps = participants();
  ps[3] = withDropout(ps[3], { fromRound: 3 });
  const { history, weights } = await federatedTrain({ rounds: 6 }, { participants: ps });
  assert.equal(history[1].responded.length, 4);
  for (const h of history.slice(2)) {
    assert.equal(h.responded.length, 3);
    assert.deepEqual(h.dropped.map((d) => d.name), ['va']);
    assert.equal(h.skipped, false);
  }
  assert.ok(weights.every(Number.isFinite));
});

test('a hospital that hangs is timed out and the round still completes', async () => {
  const ps = participants();
  ps[0] = withDropout(ps[0], { fromRound: 2, untilRound: 2, mode: 'hang' });
  const { history } = await federatedTrain({ rounds: 3 }, { participants: ps, roundTimeoutMs: 100 });
  assert.match(history[1].dropped[0].reason, /timed out/);
  assert.equal(history[2].responded.length, 4, 'rejoins after recovering');
});

test('invalid updates are rejected, and too few participants skips the round', async () => {
  const bad = { name: 'bad', stats: async () => ({ count: [1], sum: [0], sumSq: [0] }), train: async () => ({ weights: [NaN], numSamples: 5 }) };
  const ps = [...participants(), bad];
  const { history } = await federatedTrain({ rounds: 1 }, { participants: ps.slice(0, 4).concat(bad) });
  assert.deepEqual(history[0].dropped, [{ name: 'bad', reason: 'invalid update' }]);

  const all = participants().map((p) => withDropout(p, { fromRound: 1 }));
  const res = await federatedTrain({ rounds: 2 }, { participants: all, minParticipants: 2 });
  assert.ok(res.history.every((h) => h.skipped));
  assert.ok(res.weights.every((w) => w === 0));
});

test('DCP participants plug into the same coordinator (mocked compute API)', async () => {
  const jobs = [];
  const compute = {
    marketValue: 'market',
    // Stands in for the scheduler + worker: fetch each URL input, run the work function.
    for(inputs, workFunctionSource, args) {
      const job = { public: {} };
      job.localExec = async function (cores) {
        return job.exec({ localExec: cores });
      };
      job.exec = async function (offer) {
        jobs.push({ job, offer, inputs, args });
        const fn = new Function('progress', `return (${workFunctionSource})`)(() => {});
        const data = JSON.parse(JSON.stringify(sites[inputs[0].hostname].train));
        return { values: () => [fn(data, ...args)] };
      };
      return job;
    },
  };
  const dcp = SITE_IDS.map((id) =>
    createDcpParticipant(id, { dataUrl: `mem://${id}`, compute, computeGroup: { joinKey: id, joinSecret: 's' } }),
  );
  const viaDcp = await federatedTrain({ rounds: 2 }, { participants: dcp });
  const local = await federatedTrain({ rounds: 2 }, { participants: participants() });
  viaDcp.weights.forEach((w, j) => assert.ok(Math.abs(w - local.weights[j]) < 1e-12));
  assert.equal(jobs.length, SITE_IDS.length * 3); // one stats job + two rounds per site
  assert.deepEqual(jobs[0].job.computeGroups, [{ joinKey: 'cleveland', joinSecret: 's' }]);
  assert.equal(jobs[0].offer, 'market');

  jobs.length = 0;
  const viaLocalExec = await federatedTrain(
    { rounds: 1 },
    { participants: SITE_IDS.map((id) => createDcpParticipant(id, { dataUrl: `mem://${id}`, compute, localExec: true })) },
  );
  assert.ok(viaLocalExec.weights.every(Number.isFinite));
  assert.ok(jobs.every((j) => j.offer.localExec === 1));
});
