// The local training function: what each hospital runs on its own data.
//
// It is deliberately self-contained (no imports, no outer variables) so the
// exact same function can be serialized with .toString() and shipped to a DCP
// worker as a work function. test/localTrain.test.js checks this.
//
// Model: logistic regression. weights has length d + 1; the last entry is the bias.

// Number of federated rounds (coordinator only; never sent to sites).
export const DEFAULT_ROUNDS = 30;

export const DEFAULT_HYPERPARAMETERS = Object.freeze({
  learningRate: 0.05,
  localEpochs: 2,
  batchSize: 16,
  l2: 0.001,
  // FedProx proximal term. 0 gives plain FedAvg; a small value (e.g. 0.01)
  // keeps each hospital's update close to the global model, which helps when
  // hospitals' data differ a lot.
  mu: 0,
  // Differential privacy (DP-SGD). noiseMultiplier > 0 turns it on: each
  // patient's gradient is clipped to clipNorm and Gaussian noise of standard
  // deviation noiseMultiplier * clipNorm is added to every step. A hospital
  // sets these itself (see createLocalParticipant's privacy option).
  clipNorm: 1,
  noiseMultiplier: 0,
  seed: 0,
});

/**
 * Train logistic regression locally, starting from the global weights.
 *
 * @param {object} input
 * @param {number[]} input.weights           global weights (length d + 1)
 * @param {number[][]} input.X               standardized feature rows (stay on site)
 * @param {number[]} input.y                 0/1 labels (stay on site)
 * @param {object} [input.hyperparameters]
 * @returns {{weights: number[], numSamples: number, loss: number}}
 *   Only weights, a sample count and a loss leave the site. Under DP the loss
 *   is NaN, since an exact training loss is not private.
 *
 * Under DP, each step includes each patient with probability batchSize / n
 * (Poisson sampling), and there are round(n / batchSize) steps per epoch. The
 * sampling and the noise use crypto.getRandomValues, never the seed: the
 * coordinator picks the seed, and could otherwise regenerate the noise.
 */
export function localTrain(input) {
  const defaults = { learningRate: 0.05, localEpochs: 2, batchSize: 16, l2: 0.001, mu: 0, clipNorm: 1, noiseMultiplier: 0, seed: 0 };
  const hp = Object.assign({}, defaults, input.hyperparameters || {});
  const X = input.X;
  const y = input.y;
  const n = X.length;
  const d = input.weights.length - 1;
  const globalW = input.weights.slice();
  const w = input.weights.slice();
  const report = typeof progress === 'function' ? progress : () => {};

  if (n === 0) return { weights: w, numSamples: 0, loss: NaN };

  let a = (hp.seed >>> 0) || 1;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  const predict = (row) => {
    let z = w[d];
    for (let j = 0; j < d; j++) z += w[j] * row[j];
    return sigmoid(z);
  };

  const batchSize = Math.max(1, Math.min(hp.batchSize, n));
  const grad = new Array(d + 1);
  const step = (divisor) => {
    for (let j = 0; j <= d; j++) {
      let g = grad[j] / divisor;
      if (j < d) g += hp.l2 * w[j];
      g += hp.mu * (w[j] - globalW[j]);
      w[j] -= hp.learningRate * g;
    }
  };

  if (hp.noiseMultiplier > 0) {
    if (!(hp.clipNorm > 0)) throw new Error('DP-SGD needs clipNorm > 0');
    const cryptoApi = typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function' ? crypto : null;
    if (!cryptoApi) throw new Error('DP-SGD needs crypto.getRandomValues for its noise');
    const buf = new Uint32Array(1024);
    let pos = buf.length;
    const next32 = () => {
      if (pos === buf.length) {
        cryptoApi.getRandomValues(buf);
        pos = 0;
      }
      return buf[pos++];
    };
    // Uniform in (0, 1) with 53 random bits.
    const secureRand = () => ((next32() >>> 5) * 67108864 + (next32() >>> 6) + 0.5) / 9007199254740992;
    const gauss = () => Math.sqrt(-2 * Math.log(secureRand())) * Math.cos(2 * Math.PI * secureRand());
    const q = batchSize / n;
    const stepsPerEpoch = Math.max(1, Math.round(n / batchSize));
    const g = new Array(d + 1);
    for (let epoch = 0; epoch < hp.localEpochs; epoch++) {
      for (let s = 0; s < stepsPerEpoch; s++) {
        grad.fill(0);
        for (let i = 0; i < n; i++) {
          if (secureRand() >= q) continue;
          const row = X[i];
          const err = predict(row) - y[i];
          let norm = err * err;
          for (let j = 0; j < d; j++) {
            g[j] = err * row[j];
            norm += g[j] * g[j];
          }
          g[d] = err;
          const clip = Math.min(1, hp.clipNorm / Math.sqrt(norm));
          for (let j = 0; j <= d; j++) grad[j] += clip * g[j];
        }
        for (let j = 0; j <= d; j++) grad[j] += hp.noiseMultiplier * hp.clipNorm * gauss();
        // Divide by the expected batch size, not the sampled one, which is not private.
        step(batchSize);
      }
      report((epoch + 1) / hp.localEpochs);
    }
    return { weights: w, numSamples: n, loss: NaN };
  }

  const order = Array.from({ length: n }, (_, i) => i);
  for (let epoch = 0; epoch < hp.localEpochs; epoch++) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    for (let start = 0; start < n; start += batchSize) {
      const end = Math.min(start + batchSize, n);
      grad.fill(0);
      for (let k = start; k < end; k++) {
        const row = X[order[k]];
        const err = predict(row) - y[order[k]];
        for (let j = 0; j < d; j++) grad[j] += err * row[j];
        grad[d] += err;
      }
      step(end - start);
    }
    report((epoch + 1) / hp.localEpochs);
  }

  let loss = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.min(Math.max(predict(X[i]), 1e-12), 1 - 1e-12);
    loss -= y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p);
  }
  return { weights: w, numSamples: n, loss: loss / n };
}
