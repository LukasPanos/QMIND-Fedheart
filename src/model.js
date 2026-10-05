// Prediction and evaluation for a trained model.
// A model is { weights, standardization }: the weights plus the mean/std the
// features were standardized with, so it can be applied to new patients.

import { standardize } from './features.js';

const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export function zeroWeights(numFeatures) {
  return new Array(numFeatures + 1).fill(0);
}

/** Probabilities for already-standardized rows. */
export function predictProbaStandardized(weights, Xstd) {
  const d = weights.length - 1;
  return Xstd.map((row) => {
    let z = weights[d];
    for (let j = 0; j < d; j++) z += weights[j] * row[j];
    return sigmoid(z);
  });
}

/** Probabilities for raw feature rows (as produced by featurize). */
export function predictProba(model, X) {
  return predictProbaStandardized(model.weights, standardize(X, model.standardization));
}

/** Area under the ROC curve via the rank-sum formula (ties get average rank). */
export function auc(yTrue, scores) {
  const idx = scores.map((s, i) => i).sort((a, b) => scores[a] - scores[b]);
  const ranks = new Array(scores.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]] = avg;
    i = j + 1;
  }
  let nPos = 0;
  let rankSum = 0;
  yTrue.forEach((y, i) => {
    if (y === 1) {
      nPos++;
      rankSum += ranks[i];
    }
  });
  const nNeg = yTrue.length - nPos;
  if (nPos === 0 || nNeg === 0) return NaN;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

export function evaluate(model, { X, y }) {
  const p = predictProba(model, X);
  let correct = 0;
  let loss = 0;
  p.forEach((pi, i) => {
    if ((pi >= 0.5 ? 1 : 0) === y[i]) correct++;
    const q = Math.min(Math.max(pi, 1e-12), 1 - 1e-12);
    loss -= y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q);
  });
  return { n: y.length, accuracy: correct / y.length, auc: auc(y, p), logLoss: loss / y.length };
}

/** Concatenate several { X, y } datasets (used for the pooled test set and baseline). */
export function concatData(datasets) {
  return {
    X: datasets.flatMap((d) => d.X),
    y: datasets.flatMap((d) => d.y),
  };
}
