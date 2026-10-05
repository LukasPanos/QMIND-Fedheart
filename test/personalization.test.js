import { test } from 'node:test';
import assert from 'node:assert/strict';
import { personalize, PERSONALIZATION_STRATEGIES } from '../src/personalization.js';
import { createLocalParticipant } from '../src/participant.js';
import { federatedTrain } from '../src/coordinator.js';
import { runExperiment } from '../src/experiment.js';
import { loadAllSites, SITE_IDS, mulberry32, publicStandardization } from '../src/data.js';
import { evaluate } from '../src/model.js';

// Toy hospital whose patients follow y = [a > 0] (or the opposite, if flipped).
function toySite(n, flipped, seed = 1) {
  const rand = mulberry32(seed);
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const a = rand() * 2 - 1;
    X.push([a, rand() * 2 - 1]);
    y.push((a > 0) !== flipped ? 1 : 0);
  }
  return { X, y };
}

test('a hospital whose patients differ from the shared model moves towards its own data', () => {
  const shared = [4, 0, 0]; // predicts disease when a > 0
  const { X, y } = toySite(200, true); // here it is the other way round
  const out = personalize({ weights: shared, X, y, hyperparameters: { learningRate: 0.5 } });
  assert.ok(['local', 'mix25'].includes(out.strategy), out.strategy);
  assert.ok(out.cvLogLoss[out.strategy] < out.cvLogLoss.global / 2);
  assert.ok(out.weights[0] < 0, 'the personalized model learned the local relationship');
});

test('a hospital that matches the shared model keeps it, or something close', () => {
  const shared = [8, 0, 0];
  const { X, y } = toySite(200, false);
  const out = personalize({ weights: shared, X, y });
  assert.notEqual(out.strategy, 'local');
  assert.equal(out.cvLogLoss[out.strategy], Math.min(...Object.values(out.cvLogLoss)));
});

test('personalization is deterministic and refits the winner on all of the hospital\'s rows', () => {
  const { X, y } = toySite(120, true, 3);
  const a = personalize({ weights: [1, 0, 0], X, y });
  const b = personalize({ weights: [1, 0, 0], X, y });
  assert.deepEqual(a, b);
  const winner = PERSONALIZATION_STRATEGIES.find((s) => s.name === a.strategy);
  assert.equal(a.label, winner.label);
  assert.equal(Object.keys(a.cvLogLoss).length, PERSONALIZATION_STRATEGIES.length);
});

test('with too few patients of one class to cross-validate, the shared model is kept', () => {
  const X = [[1, 0], [0.5, 0], [-1, 0], [0.2, 1]];
  const out = personalize({ weights: [1, 0, 0], X, y: [1, 1, 0, 1] });
  assert.equal(out.strategy, 'global');
  assert.deepEqual(out.weights, [1, 0, 0]);
  assert.deepEqual(out.cvLogLoss, {});
});

test('Switzerland, the most different hospital, gains the most from personalizing', async () => {
  const { perSite } = await runExperiment({ dropout: null });
  const swiss = perSite.switzerland;
  assert.notEqual(swiss.strategy, 'global');
  assert.ok(swiss.personalized.logLoss < swiss.federated.logLoss * 0.8, JSON.stringify(swiss));
  const mean = (k) => SITE_IDS.reduce((s, id) => s + perSite[id][k].logLoss, 0) / SITE_IDS.length;
  assert.ok(mean('personalized') < mean('federated'), 'better on average across hospitals');
});

test('hospitals under differential privacy can personalize too', async () => {
  const sites = loadAllSites();
  const ps = SITE_IDS.map((id) => createLocalParticipant(id, sites[id].train, { privacy: { noiseMultiplier: 3 } }));
  const model = await federatedTrain({ rounds: 3 }, { participants: ps, standardization: publicStandardization() });
  const before = ps[2].privacySpent();
  const out = ps[2].personalize(model);
  assert.deepEqual(ps[2].privacySpent(), before, 'personalizing sends nothing, so it spends no privacy');
  assert.ok(evaluate(out.model, sites.switzerland.test).logLoss < evaluate(model, sites.switzerland.test).logLoss);
});
