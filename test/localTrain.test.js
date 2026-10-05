import { test } from 'node:test';
import assert from 'node:assert/strict';
import { localTrain } from '../src/localTrain.js';
import { buildWorkFunctionSource } from '../src/dcp.js';
import { mulberry32 } from '../src/data.js';

function toyData(n = 200, seed = 1) {
  const rand = mulberry32(seed);
  const X = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const a = rand() * 2 - 1;
    const b = rand() * 2 - 1;
    X.push([a, b]);
    y.push(2 * a - b + 0.2 * (rand() - 0.5) > 0 ? 1 : 0);
  }
  return { X, y };
}

test('localTrain lowers the loss and returns only weights, a count and a loss', () => {
  const { X, y } = toyData();
  const before = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: { localEpochs: 0 } });
  const after = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: { localEpochs: 20, learningRate: 0.5 } });
  assert.deepEqual(Object.keys(after).sort(), ['loss', 'numSamples', 'weights']);
  assert.equal(after.numSamples, 200);
  assert.ok(after.loss < before.loss * 0.6, `${after.loss} vs ${before.loss}`);
  assert.ok(after.weights[0] > 0 && after.weights[1] < 0);
});

test('localTrain does not mutate the global weights it was given', () => {
  const { X, y } = toyData(20);
  const w = [0.1, 0.2, 0.3];
  localTrain({ weights: w, X, y });
  assert.deepEqual(w, [0.1, 0.2, 0.3]);
});

test('localTrain is deterministic for a given seed', () => {
  const { X, y } = toyData(50);
  const a = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: { seed: 9 } });
  const b = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: { seed: 9 } });
  assert.deepEqual(a, b);
});

test('FedProx term keeps the update closer to the global weights', () => {
  const { X, y } = toyData();
  const hp = { localEpochs: 10, learningRate: 0.5 };
  const dist = (w) => Math.hypot(...w);
  const plain = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: { ...hp, mu: 0 } });
  const prox = localTrain({ weights: [0, 0, 0], X, y, hyperparameters: { ...hp, mu: 1 } });
  assert.ok(dist(prox.weights) < dist(plain.weights));
});

test('localTrain is self-contained, so it survives serialization (DCP work function)', () => {
  const rebuilt = new Function(`return (${localTrain.toString()})`)();
  const { X, y } = toyData(30);
  const input = { weights: [0, 0, 0], X, y, hyperparameters: { seed: 4 } };
  assert.deepEqual(rebuilt(input), localTrain(input));
});

test('DCP work function source runs standalone for both tasks', () => {
  const { X, y } = toyData(40);
  const calls = [];
  const fn = new Function('progress', `return (${buildWorkFunctionSource()})`)((p) => calls.push(p));

  const stats = fn(JSON.stringify({ X, y }), 'stats', null);
  assert.equal(stats.count[0], 40);

  const out = fn({ X, y }, 'train', {
    weights: [0, 0, 0], standardization: { mean: [0, 0], std: [1, 1] }, hyperparameters: { seed: 2 },
  });
  assert.deepEqual(out, localTrain({ weights: [0, 0, 0], X, y, hyperparameters: { seed: 2 } }));
  assert.equal(calls.at(-1), 1);
});
