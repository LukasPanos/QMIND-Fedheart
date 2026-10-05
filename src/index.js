// FedHeart: federated logistic regression across hospitals.
//
//   import { federatedTrain, createLocalParticipant } from 'fedheart';
//   const { weights } = await federatedTrain({ rounds: 30, learningRate: 0.05 }, { participants });
//
// Hyperparameters go in, trained weights come out, training data never leaves
// a participant, and a participant that drops out is skipped for that round.

export { federatedTrain, fedAvg, federatedStandardization } from './coordinator.js';
export { localTrain, DEFAULT_HYPERPARAMETERS, DEFAULT_ROUNDS } from './localTrain.js';
export { createLocalParticipant, withDropout, ParticipantUnavailableError } from './participant.js';
export {
  withSecureAggregation, createPairwiseSecrets, maskValues, sumMasked, encodeFixedPoint, decodeFixedPoint,
} from './secureAggregation.js';
export {
  SITES, SITE_IDS, FEATURES, parseSiteFile, loadSite, loadAllSites, featurize, featurizeAll,
  trainTestSplit, localStats, combineStats, standardize, FEATURE_BOUNDS, publicStandardization,
} from './data.js';
export {
  createPrivacyAccountant, rdpSampledGaussian, rdpToEpsilon, dpSgdSteps, PrivacyBudgetExceededError,
} from './privacy.js';
export { evaluate, predictProba, auc, concatData, zeroWeights } from './model.js';
export { trainLocalOnly, trainPooled } from './baselines.js';
export { runExperiment, runBaselines, runFederated, evaluateEverywhere } from './experiment.js';
export { personalize, PERSONALIZATION_STRATEGIES } from './personalization.js';
export { createHub } from './hub.js';
export { createHospital, loadDatasetText } from './hospital.js';
