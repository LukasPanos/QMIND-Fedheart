// Non-federated reference points: one hospital training alone, and all data
// pooled in one place (the upper bound federated learning tries to approach,
// and which privacy rules forbid in practice).

import { combineStats, localStats, standardize } from './data.js';
import { localTrain, DEFAULT_HYPERPARAMETERS, DEFAULT_ROUNDS } from './localTrain.js';
import { concatData, zeroWeights } from './model.js';

/** Train on one dataset, standardized with its own statistics. */
export function trainCentral({ X, y }, hyperparameters = {}) {
  const { rounds = DEFAULT_ROUNDS, ...hp } = { ...DEFAULT_HYPERPARAMETERS, ...hyperparameters };
  // Give baselines the same number of passes over the data as federated training.
  const epochs = rounds * hp.localEpochs;
  const standardization = combineStats([localStats(X)]);
  const { weights } = localTrain({
    weights: zeroWeights(standardization.mean.length),
    X: standardize(X, standardization),
    y,
    hyperparameters: { ...hp, localEpochs: epochs },
  });
  return { weights, standardization };
}

export function trainLocalOnly(siteTrain, hyperparameters) {
  return trainCentral(siteTrain, hyperparameters);
}

export function trainPooled(siteTrains, hyperparameters) {
  return trainCentral(concatData(siteTrains), hyperparameters);
}
