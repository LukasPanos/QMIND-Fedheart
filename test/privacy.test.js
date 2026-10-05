import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createPrivacyAccountant, rdpSampledGaussian, dpSgdSteps, PrivacyBudgetExceededError } from '../src/privacy.js';
import { localTrain } from '../src/localTrain.js';
import { createLocalParticipant } from '../src/participant.js';
import { createPairwiseSecrets, withSecureAggregation } from '../src/secureAggregation.js';
import { federatedTrain } from '../src/coordinator.js';
import { buildWorkFunctionSource } from '../src/dcp.js';
import { loadAllSites, SITE_IDS, publicStandardization, FEATURE_BOUNDS, FEATURES, mulberry32 } from '../src/data.js';
import { evaluate, concatData } from '../src/model.js';

const sites = loadAllSites();
const testSet = concatData(SITE_IDS.map((id) => sites[id].test));
const epsilonAfter = (q, noiseMultiplier, steps, delta = 1e-5) => {
  const a = createPrivacyAccountant({ delta });
  a.record({ q, noiseMultiplier, steps });
  return a.epsilon();
};

test('accountant matches the DP-SGD MNIST reference and behaves monotonically', () => {
  // Abadi et al. / TF-Privacy: q = 256/60000, sigma 1.1, 60 epochs, delta 1e-5.
  // TF-Privacy reports 2.92 with the older RDP conversion; the tighter one is lower.
  const mnist = epsilonAfter(256 / 60000, 1.1, Math.round((60 * 60000) / 256));
  assert.ok(mnist > 2.3 && mnist < 2.92, `${mnist}`);
  assert.ok(epsilonAfter(0.1, 1, 200) > epsilonAfter(0.1, 1, 100), 'more steps cost more');
  assert.ok(epsilonAfter(0.1, 2, 100) < epsilonAfter(0.1, 1, 100), 'more noise costs less');
  assert.ok(epsilonAfter(0.05, 1, 100) < epsilonAfter(0.1, 1, 100), 'sampling fewer patients costs less');
  // Without subsampling it is the plain Gaussian mechanism: RDP alpha / (2 sigma^2).
  assert.equal(rdpSampledGaussian(1, 2, 4), 4 / 8);
  assert.ok(Math.abs(rdpSampledGaussian(0.999999, 2, 4) - 0.5) < 1e-4);
});

test('public standardization covers every feature and needs no data', () => {
  assert.equal(FEATURE_BOUNDS.length, FEATURES.length);
  const std = publicStandardization();
  assert.ok(std.std.every((s) => s > 0));
  for (const id of SITE_IDS) {
    for (const row of sites[id].train.X) {
      row.forEach((v, j) => {
        if (v != null) assert.ok(v >= FEATURE_BOUNDS[j][0] - 5 && v <= FEATURE_BOUNDS[j][1] + 5, `${FEATURES[j]} = ${v}`);
      });
    }
  }
});

test('DP-SGD clips each patient: one extreme patient moves the weights by at most lr * clipNorm per step', () => {
  const hp = { localEpochs: 5, batchSize: 1, learningRate: 0.1, l2: 0, clipNorm: 0.5, noiseMultiplier: 1e-9 };
  const out = localTrain({ weights: [0, 0, 0], X: [[1e6, -1e6]], y: [1], hyperparameters: hp });
  const moved = Math.hypot(...out.weights);
  assert.ok(moved <= 0.1 * 0.5 * 5 + 1e-6, `${moved}`);
  // The first step moves the full clipped amount; after it the patient is fit and gradients vanish.
  assert.ok(moved > 0.1 * 0.5 * 0.99, `${moved}`);
});

test('DP-SGD noise ignores the seed, reports no loss, and still learns', () => {
  const rand = mulberry32(3);
  const X = [];
  const y = [];
  for (let i = 0; i < 400; i++) {
    const a = rand() * 2 - 1;
    X.push([a, rand() * 2 - 1]);
    y.push(a > 0 ? 1 : 0);
  }
  const hp = { localEpochs: 10, learningRate: 0.5, noiseMultiplier: 1, clipNorm: 1, seed: 7 };
  const a = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: hp });
  const b = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: hp });
  assert.notDeepEqual(a.weights, b.weights, 'the coordinator-chosen seed does not determine the noise');
  assert.ok(Number.isNaN(a.loss));
  assert.equal(a.numSamples, 400);
  assert.ok(a.weights[0] > 1 && Math.abs(a.weights[1]) < a.weights[0] / 2, `${a.weights}`);
});

test('the DCP work function runs DP-SGD too', () => {
  const fn = new Function('progress', `return (${buildWorkFunctionSource()})`)(() => {});
  const std = publicStandardization();
  const out = fn(sites.va.train, 'train', {
    weights: new Array(19).fill(0),
    hyperparameters: { noiseMultiplier: 2, clipNorm: 1 },
    standardization: std,
  });
  assert.ok(Number.isNaN(out.loss) && out.weights.every(Number.isFinite));
});

test('a hospital enforces its own privacy settings, whatever the coordinator sends', async () => {
  const p = createLocalParticipant('va', sites.va.train, { privacy: { noiseMultiplier: 2 } });
  const ctx = { round: 1, standardization: publicStandardization() };
  const out = await p.train(new Array(19).fill(0), { noiseMultiplier: 0, localEpochs: 2, batchSize: 16 }, ctx);
  assert.ok(Number.isNaN(out.loss), 'trained with DP even though the coordinator asked for none');
  await assert.rejects(p.stats(), /does not release exact feature statistics/);

  const { q, steps } = dpSgdSteps(sites.va.train.y.length, { localEpochs: 2, batchSize: 16 });
  const expected = epsilonAfter(q, 2, steps);
  assert.ok(Math.abs(p.privacySpent().epsilon - expected) < 1e-9);
  assert.equal(p.privacySpent().rounds, 1);

  const again = await p.train(new Array(19).fill(0), {}, ctx);
  assert.equal(again, out, 'asking again for the same round re-sends the same update');
  assert.equal(p.privacySpent().rounds, 1, 'and costs no extra privacy');
});

test('a hospital stops training once its privacy budget is spent', async () => {
  const ps = SITE_IDS.map((id) =>
    createLocalParticipant(id, sites[id].train, { privacy: { noiseMultiplier: 3, epsilonBudget: id === 'switzerland' ? 1.5 : Infinity } }),
  );
  const { history } = await federatedTrain({ rounds: 10 }, { participants: ps, standardization: publicStandardization() });
  const swiss = ps.find((p) => p.name === 'switzerland').privacySpent();
  assert.ok(swiss.epsilon <= 1.5 && swiss.rounds > 0 && swiss.rounds < 10, JSON.stringify(swiss));
  const last = history.at(-1);
  assert.equal(last.responded.length, 3);
  assert.match(last.dropped[0].reason, /used its privacy budget/);
  assert.ok(new PrivacyBudgetExceededError('x', 2, 1) instanceof Error);
});

test('federated training with DP and secure aggregation together still beats chance by far', async () => {
  const secrets = createPairwiseSecrets(SITE_IDS);
  const ps = SITE_IDS.map((id) =>
    withSecureAggregation(createLocalParticipant(id, sites[id].train, { privacy: { noiseMultiplier: 3 } }), { secrets: secrets[id] }),
  );
  const model = await federatedTrain(
    { rounds: 10, learningRate: 0.2 },
    { participants: ps, secureAggregation: true, standardization: publicStandardization() },
  );
  assert.ok(model.history.every((h) => Number.isNaN(h.trainLoss) && !h.skipped));
  const auc = evaluate(model, testSet).auc;
  assert.ok(auc > 0.8, `${auc}`);
  for (const p of ps) assert.ok(p.privacySpent().epsilon < 3.1);
});
