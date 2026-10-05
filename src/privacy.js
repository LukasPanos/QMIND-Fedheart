// Differential privacy accounting for DP-SGD, run by each hospital for itself.
//
// localTrain with noiseMultiplier > 0 runs DP-SGD: every step samples each
// patient with probability q (Poisson sampling), clips each patient's gradient
// to clipNorm, and adds Gaussian noise with standard deviation
// noiseMultiplier * clipNorm to the sum. Every step is a subsampled Gaussian
// mechanism, and this module adds up their privacy cost with Renyi DP
// (Mironov, Talwar and Zhang, 2019) and converts it to (epsilon, delta).
//
// The guarantee is per patient (add or remove one patient's record) and holds
// against anyone who sees the hospital's updates, including the coordinator and
// the other hospitals. Each hospital's patient count is released exactly and
// treated as public.

const ORDERS = Array.from({ length: 255 }, (_, i) => i + 2); // integer Renyi orders 2..256

function logAdd(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** RDP of one step of the Poisson-subsampled Gaussian mechanism at integer order alpha. */
export function rdpSampledGaussian(q, noiseMultiplier, alpha) {
  if (q === 0) return 0;
  if (noiseMultiplier === 0) return Infinity;
  if (q === 1) return alpha / (2 * noiseMultiplier ** 2);
  // A_alpha = sum_k C(alpha, k) (1 - q)^(alpha - k) q^k exp((k^2 - k) / (2 sigma^2)), in log space.
  let logA = -Infinity;
  let logBinom = 0;
  for (let k = 0; k <= alpha; k++) {
    if (k > 0) logBinom += Math.log(alpha - k + 1) - Math.log(k);
    const term = logBinom + (alpha - k) * Math.log(1 - q) + k * Math.log(q) + (k * k - k) / (2 * noiseMultiplier ** 2);
    logA = logAdd(logA, term);
  }
  return logA / (alpha - 1);
}

/**
 * Convert RDP values (one per order) to the smallest epsilon at this delta,
 * using the conversion of Balle et al. (2020).
 */
export function rdpToEpsilon(rdp, delta) {
  let best = Infinity;
  ORDERS.forEach((alpha, i) => {
    const eps = rdp[i] + Math.log((alpha - 1) / alpha) - (Math.log(delta) + Math.log(alpha)) / (alpha - 1);
    if (eps < best) best = eps;
  });
  return Math.max(best, 0);
}

/**
 * What one call of localTrain does under DP: steps and sampling rate. Must
 * match localTrain, which cannot import it (it has to stay self-contained).
 */
export function dpSgdSteps(numSamples, hyperparameters) {
  const batchSize = Math.max(1, Math.min(hyperparameters.batchSize, numSamples));
  const stepsPerEpoch = Math.max(1, Math.round(numSamples / batchSize));
  return { q: batchSize / numSamples, steps: hyperparameters.localEpochs * stepsPerEpoch };
}

/** Running privacy cost for one hospital. */
export function createPrivacyAccountant({ delta = 1e-5 } = {}) {
  const rdp = ORDERS.map(() => 0);
  let rounds = 0;
  const added = (q, noiseMultiplier, steps) => ORDERS.map((alpha, i) => rdp[i] + steps * rdpSampledGaussian(q, noiseMultiplier, alpha));
  return {
    delta,
    /** Epsilon so far, or after `extra` more DP-SGD steps if given. */
    epsilon(extra) {
      return rdpToEpsilon(extra ? added(extra.q, extra.noiseMultiplier, extra.steps) : rdp, delta);
    },
    record({ q, noiseMultiplier, steps }) {
      added(q, noiseMultiplier, steps).forEach((v, i) => { rdp[i] = v; });
      rounds++;
    },
    get rounds() {
      return rounds;
    },
  };
}

export class PrivacyBudgetExceededError extends Error {
  constructor(name, epsilon, budget) {
    super(`${name} has used its privacy budget (another round would bring epsilon to ${epsilon.toFixed(2)}, budget ${budget})`);
    this.name = 'PrivacyBudgetExceededError';
  }
}
