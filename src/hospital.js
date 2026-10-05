// What a dummy-DCP hospital worker does with its dataset. It has no Node
// imports, so it runs anywhere: src/hospitalWorker.js wraps it for a worker
// process per hospital folder, and test/hub.test.js drives it directly.
//
// The rows stay in this closure. handle(task) returns only what the
// coordinator is allowed to see: aggregates, weights, holdout metrics, or the
// result of personalizing (which candidate won and how it scores, never the
// personalized model itself).

import { parseAnySiteFile, featurizeAll, trainTestSplit } from './features.js';
import { createLocalParticipant } from './participant.js';
import { createPrivacyAccountant } from './privacy.js';
import { evaluate } from './model.js';

/**
 * Parse a site's file (14-column processed format or UCI's original
 * 76-attribute format) and split it the same way loadAllSites does.
 */
export function loadDatasetText(text, { testFraction = 0.3, seed = 42 } = {}) {
  const { records, skipped, format } = parseAnySiteFile(text);
  if (records.length < 10) throw new Error(`Need at least 10 patients, found ${records.length}`);
  const data = featurizeAll(records);
  return {
    ...trainTestSplit(data, { testFraction, seed }),
    rows: records.length,
    skipped,
    format,
    diseaseRate: data.y.reduce((s, v) => s + v, 0) / data.y.length,
  };
}

const holdoutMetrics = (model, test) => {
  const m = evaluate(model, test);
  return { n: m.n, auc: m.auc, accuracy: m.accuracy, logLoss: m.logLoss };
};

/**
 * One hospital. Tasks:
 *   { type: 'configure', privacy }      the hospital's own DP settings for the next run
 *                                       (null turns DP off) -> privacy status
 *   { type: 'stats' }                   -> { count, sum, sumSq } (refused under DP)
 *   { type: 'train', weights, hyperparameters, round, standardization } -> { weights, numSamples, loss }
 *   { type: 'evaluate', weights, standardization } -> holdout { n, auc, accuracy, logLoss }
 *   { type: 'personalize', weights, standardization } -> { strategy, label, shared, personalized }
 *   { type: 'privacy' }                 -> privacy status
 *
 * Privacy loss adds up over every run on the same data, so one accountant
 * lives as long as the hospital. A hospital that has ever trained without DP
 * reports itself as not private.
 */
export function createHospital({ train, test }, { name = 'hospital' } = {}) {
  let privacy = null;
  let accountant = null;
  let trainedWithoutDp = false;
  let participant = createLocalParticipant(name, train);

  const privacyStatus = () => {
    const spent = accountant?.rounds ? { epsilon: accountant.epsilon(), delta: accountant.delta, rounds: accountant.rounds } : null;
    return { dp: Boolean(privacy), settings: privacy, spent, trainedWithoutDp };
  };

  return {
    async handle(task) {
      switch (task.type) {
        case 'configure': {
          privacy = task.privacy ? { ...task.privacy } : null;
          if (privacy) {
            accountant ??= createPrivacyAccountant({ delta: privacy.delta ?? 1e-5 });
            privacy.delta = accountant.delta;
          }
          participant = createLocalParticipant(name, train, { privacy: privacy && { ...privacy, accountant } });
          return privacyStatus();
        }
        case 'stats':
          return participant.stats();
        case 'train': {
          const result = await participant.train(task.weights, task.hyperparameters, {
            round: task.round,
            standardization: task.standardization,
          });
          if (!privacy) trainedWithoutDp = true;
          return result;
        }
        case 'evaluate':
          return holdoutMetrics({ weights: task.weights, standardization: task.standardization }, test);
        case 'personalize': {
          const shared = { weights: task.weights, standardization: task.standardization };
          const { strategy, label, model } = participant.personalize(shared, { hyperparameters: task.hyperparameters });
          return { strategy, label, shared: holdoutMetrics(shared, test), personalized: holdoutMetrics(model, test) };
        }
        case 'privacy':
          return privacyStatus();
        default:
          throw new Error(`Unknown task type "${task.type}"`);
      }
    },
  };
}
