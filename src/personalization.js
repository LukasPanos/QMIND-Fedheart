// Personalization: after federated training, each hospital adapts the shared
// model to its own patients, entirely on site.
//
// The hospitals differ a lot (disease rates from 36% to 93%, Switzerland never
// recorded cholesterol), so the best model for one hospital's patients is not
// always the shared one. Each hospital tries a ladder of candidates, from fully
// shared to fully local, scores each by cross-validation on its own training
// rows, and keeps the best. Nothing is sent anywhere: the coordinator never
// sees the personalized model or which candidate won.
//
// Cross-validation uses log loss, which works even when a hospital has very
// few patients without disease (Switzerland's test split has 2), where AUC
// is meaningless.

import { localTrain, DEFAULT_HYPERPARAMETERS, DEFAULT_ROUNDS } from './localTrain.js';
import { mulberry32 } from './features.js';
import { predictProbaStandardized } from './model.js';

const fineTune = (epochs) => (global, X, y, hp) =>
  localTrain({ weights: global, X, y, hyperparameters: { ...hp, localEpochs: epochs } }).weights;

// Same number of passes over the data as federated training, like trainLocalOnly.
const localOnly = (global, X, y, hp) =>
  localTrain({ weights: global.map(() => 0), X, y, hyperparameters: { ...hp, localEpochs: DEFAULT_ROUNDS * hp.localEpochs } }).weights;

const mix = (shareOfGlobal) => (global, X, y, hp) => {
  const local = localOnly(global, X, y, hp);
  return global.map((w, j) => shareOfGlobal * w + (1 - shareOfGlobal) * local[j]);
};

/** Candidates from fully shared to fully local. fit(globalWeights, X, y, hp) -> weights. */
export const PERSONALIZATION_STRATEGIES = [
  { name: 'global', label: 'Shared model as is', fit: (global) => global.slice() },
  { name: 'fineTune3', label: 'Fine-tuned, 3 epochs', fit: fineTune(3) },
  { name: 'fineTune10', label: 'Fine-tuned, 10 epochs', fit: fineTune(10) },
  { name: 'mix75', label: '75% shared, 25% local', fit: mix(0.75) },
  { name: 'mix50', label: '50% shared, 50% local', fit: mix(0.5) },
  { name: 'mix25', label: '25% shared, 75% local', fit: mix(0.25) },
  { name: 'local', label: 'Local model only', fit: localOnly },
];

function logLoss(weights, X, y) {
  const p = predictProbaStandardized(weights, X);
  let loss = 0;
  p.forEach((pi, i) => {
    const q = Math.min(Math.max(pi, 1e-12), 1 - 1e-12);
    loss -= y[i] ? Math.log(q) : Math.log(1 - q);
  });
  return loss / y.length;
}

/** Stratified fold index per row, so every fold keeps the hospital's disease rate. */
function stratifiedFolds(y, k, seed) {
  const rand = mulberry32(seed);
  const fold = new Array(y.length);
  for (const label of [0, 1]) {
    const idx = y.map((v, i) => (v === label ? i : -1)).filter((i) => i >= 0);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    idx.forEach((i, t) => { fold[i] = t % k; });
  }
  return fold;
}

/**
 * Pick and fit the best candidate for one hospital.
 *
 * @param {object} input
 * @param {number[]} input.weights       the shared model's weights
 * @param {number[][]} input.X           this hospital's rows, standardized like the shared model
 * @param {number[]} input.y
 * @param {object} [input.hyperparameters]
 * @param {Array} [input.strategies]     defaults to PERSONALIZATION_STRATEGIES
 * @param {number} [input.folds=5]
 * @param {number} [input.seed=0]
 * @returns {{ strategy: string, label: string, weights: number[], cvLogLoss: object }}
 *   cvLogLoss maps each candidate to its cross-validated log loss.
 *
 * Caveat: the shared model was trained on these same rows, so cross-validation
 * flatters it slightly. The choice therefore leans towards the shared model,
 * which is the safe direction.
 */
export function personalize({ weights, X, y, hyperparameters = {}, strategies = PERSONALIZATION_STRATEGIES, folds = 5, seed = 0 }) {
  const hp = { ...DEFAULT_HYPERPARAMETERS, ...hyperparameters };
  // Each fold needs patients with and without disease.
  const k = Math.min(folds, ...[0, 1].map((label) => y.filter((v) => v === label).length));
  const cvLogLoss = {};
  if (k >= 2) {
    const fold = stratifiedFolds(y, k, seed);
    for (const s of strategies) {
      let total = 0;
      for (let f = 0; f < k; f++) {
        const train = [];
        const valid = [];
        fold.forEach((g, i) => (g === f ? valid : train).push(i));
        const w = s.fit(weights, train.map((i) => X[i]), train.map((i) => y[i]), hp);
        total += logLoss(w, valid.map((i) => X[i]), valid.map((i) => y[i])) * valid.length;
      }
      cvLogLoss[s.name] = total / y.length;
    }
  }
  // Too few patients to cross-validate: keep the shared model.
  const best = k >= 2
    ? strategies.reduce((a, b) => (cvLogLoss[b.name] < cvLogLoss[a.name] ? b : a))
    : strategies.find((s) => s.name === 'global') ?? strategies[0];
  return { strategy: best.name, label: best.label, weights: best.fit(weights, X, y, hp), cvLogLoss };
}
