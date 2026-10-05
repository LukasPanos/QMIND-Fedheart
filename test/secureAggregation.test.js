import { test } from 'node:test';
import assert from 'node:assert/strict';
import { federatedTrain } from '../src/coordinator.js';
import { createLocalParticipant, withDropout } from '../src/participant.js';
import {
  createPairwiseSecrets, withSecureAggregation, maskValues, sumMasked, encodeFixedPoint, decodeFixedPoint,
} from '../src/secureAggregation.js';
import { loadAllSites, SITE_IDS } from '../src/data.js';

const sites = loadAllSites();
const plain = () => SITE_IDS.map((id) => createLocalParticipant(id, sites[id].train));
const secure = (ps = plain()) => {
  const secrets = createPairwiseSecrets(ps.map((p) => p.name));
  return ps.map((p) => withSecureAggregation(p, { secrets: secrets[p.name] }));
};
const close = (a, b, tol, what) => a.forEach((x, j) => assert.ok(Math.abs(x - b[j]) < tol, `${what}[${j}]: ${x} vs ${b[j]}`));

test('fixed point round-trips negative and fractional values', () => {
  for (const x of [0, 1, -1, 0.123456, -98765.4321, 1.3e7]) assert.ok(Math.abs(decodeFixedPoint(encodeFixedPoint(x)) - x) < 1e-7);
  assert.throws(() => encodeFixedPoint(NaN), RangeError);
});

test('pairwise masks hide each value and cancel in the sum', () => {
  const names = ['a', 'b', 'c'];
  const secrets = createPairwiseSecrets(names);
  const values = { a: [1.5, -2, 300], b: [0.25, 4, -100], c: [-1, 0, 0.5] };
  const masked = names.map((self) => maskValues(values[self], { self, cohort: names, secrets: secrets[self], label: 'test' }));
  for (const [i, m] of masked.entries()) {
    const alone = m.map((h) => decodeFixedPoint(BigInt(`0x${h}`)));
    assert.ok(alone.every((v, j) => Math.abs(v - values[names[i]][j]) > 1), 'a single masked vector is not its plaintext');
  }
  close(sumMasked(masked), [0.75, 2, 200.5], 1e-6, 'sum');
});

test('secure aggregation trains the same model as plain FedAvg', async () => {
  const hp = { rounds: 5 };
  const a = await federatedTrain(hp, { participants: plain() });
  const b = await federatedTrain(hp, { participants: secure(), secureAggregation: true });
  close(b.standardization.mean, a.standardization.mean, 1e-6, 'mean');
  close(b.standardization.std, a.standardization.std, 1e-6, 'std');
  close(b.weights, a.weights, 1e-5, 'weights');
  b.history.forEach((h, i) => {
    assert.equal(h.attempts, 1);
    assert.ok(Math.abs(h.trainLoss - a.history[i].trainLoss) < 1e-6);
  });
});

test('the coordinator only ever receives masked vectors', async () => {
  const seen = [];
  await federatedTrain({ rounds: 2 }, {
    participants: secure(),
    secureAggregation: true,
    onResponse: (name, value) => seen.push(value),
  });
  assert.equal(seen.length, SITE_IDS.length * 3);
  for (const v of seen) {
    assert.deepEqual(Object.keys(v), ['masked']);
    assert.ok(v.masked.every((h) => /^[0-9a-f]{16}$/.test(h)));
  }
});

test('after a dropout the round is re-run with the remaining hospitals', async () => {
  const withVaDrop = () => plain().map((p) => (p.name === 'va' ? withDropout(p, { fromRound: 3 }) : p));
  const a = await federatedTrain({ rounds: 5 }, { participants: withVaDrop() });
  const b = await federatedTrain({ rounds: 5 }, { participants: secure(withVaDrop()), secureAggregation: true });
  close(b.weights, a.weights, 1e-5, 'weights');
  assert.equal(b.history[1].attempts, 1);
  for (const h of b.history.slice(2)) {
    assert.equal(h.attempts, 2);
    assert.equal(h.responded.length, 3);
    assert.deepEqual(h.dropped.map((d) => d.name), ['va']);
    assert.equal(h.skipped, false);
  }
});

test('too few hospitals left skips the round instead of exposing anyone', async () => {
  const ps = plain().map((p, i) => (i >= 2 ? withDropout(p, { fromRound: 2 }) : p));
  const { history } = await federatedTrain({ rounds: 2 }, { participants: secure(ps), secureAggregation: true });
  assert.equal(history[0].skipped, false);
  assert.equal(history[1].skipped, true);
  assert.equal(history[1].attempts, 1, 'a cohort of 2 is never asked');
});

test('hospitals refuse small cohorts, unknown peers and invalid updates', async () => {
  const [cle, hun] = secure();
  const w = new Array(19).fill(0);
  const ctx = (cohort) => ({ round: 1, attempt: 1, cohort, standardization: { mean: new Array(18).fill(0), std: new Array(18).fill(1) } });
  await assert.rejects(cle.train(w, {}, ctx(['cleveland', 'hungarian'])), /refusing a cohort of 2/);
  await assert.rejects(cle.train(w, {}, ctx(['cleveland', 'hungarian', 'mallory'])), /shares no secret with mallory/);
  await assert.rejects(hun.train(w, {}, ctx(['cleveland', 'va', 'switzerland'])), /not a member/);

  const broken = withSecureAggregation(
    { name: 'x', stats: async () => ({}), train: async () => ({ weights: [NaN], numSamples: 5, loss: 1 }) },
    { secrets: {} },
  );
  await assert.rejects(broken.train([0], {}, ctx(['x', 'y', 'z'])), /invalid update/);
  await assert.rejects(broken.stats(ctx(['x', 'y', 'z'])), /invalid feature statistics/);
});

test('masks are never reused: a new run gets new masks, and a reused label is refused', async () => {
  const ps = secure();
  const firstRound = async () => {
    const seen = {};
    await federatedTrain({ rounds: 1 }, {
      participants: ps,
      secureAggregation: true,
      onResponse: (name, value, context) => { if (context.round === 1) seen[name] = value.masked; },
    });
    return seen;
  };
  const a = await firstRound();
  const b = await firstRound(); // same hospitals, same secrets, same round number
  for (const id of SITE_IDS) assert.notEqual(a[id][0], b[id][0], `${id} reused its round-1 mask`);

  const [cle] = ps;
  const ctx = { round: 1, attempt: 1, session: 's', cohort: SITE_IDS, standardization: { mean: new Array(18).fill(0), std: new Array(18).fill(1) } };
  await cle.train(new Array(19).fill(0), {}, ctx);
  await assert.rejects(cle.train(new Array(19).fill(0.5), {}, ctx), /refusing to reuse masks/);
});
