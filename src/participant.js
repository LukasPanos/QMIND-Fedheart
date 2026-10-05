// A participant is one hospital as the coordinator sees it. The coordinator
// only ever calls these two methods, and gets back aggregates and weights:
//
//   stats()                         -> { count, sum, sumSq } per feature
//   train(weights, hp, context)     -> { weights, numSamples, loss }
//
// createLocalParticipant keeps the data in a closure, standing in for a
// hospital's own server. src/dcp.js builds participants with the same shape
// whose work runs on DCP workers instead.

import { localStats, standardize } from './features.js';
import { localTrain, DEFAULT_HYPERPARAMETERS } from './localTrain.js';
import { createPrivacyAccountant, dpSgdSteps, PrivacyBudgetExceededError } from './privacy.js';
import { personalize } from './personalization.js';

export class ParticipantUnavailableError extends Error {
  constructor(name, round) {
    super(`${name} is unavailable in round ${round}`);
    this.name = 'ParticipantUnavailableError';
  }
}

/**
 * @param {string} name
 * @param {{X, y}} data       this hospital's rows, which never leave the closure
 * @param {object} [options.privacy] differential privacy, set by the hospital:
 *   { noiseMultiplier = 3, clipNorm = 1, delta = 1e-5, epsilonBudget = Infinity, accountant }.
 *   Pass an accountant (createPrivacyAccountant) to keep counting across
 *   several training runs on the same data, as privacy loss adds up.
 *   These override whatever the coordinator sends. The hospital tracks its own
 *   epsilon, refuses to train once another round would exceed epsilonBudget,
 *   and refuses to release exact feature statistics (use publicStandardization).
 */
export function createLocalParticipant(name, { X, y }, { privacy } = {}) {
  let cachedStd = null;
  let cachedFor = null;
  const dp = privacy ? { noiseMultiplier: 3, clipNorm: 1, delta: 1e-5, ...privacy } : null;
  // No budget means unlimited. Infinity arrives as null after JSON (IPC, HTTP), so accept that too.
  if (dp) dp.epsilonBudget = dp.epsilonBudget == null ? Infinity : Number(dp.epsilonBudget);
  if (dp && !(dp.noiseMultiplier > 0 && dp.clipNorm > 0)) throw new Error(`${name}: privacy needs noiseMultiplier > 0 and clipNorm > 0`);
  const accountant = dp && (dp.accountant ?? createPrivacyAccountant({ delta: dp.delta }));
  let lastRelease = null;

  return {
    name,
    async stats() {
      if (dp) throw new Error(`${name} does not release exact feature statistics under differential privacy`);
      return localStats(X);
    },
    async train(weights, hyperparameters, { round, standardization }) {
      if (cachedFor !== standardization) {
        cachedStd = standardize(X, standardization);
        cachedFor = standardization;
      }
      if (!dp) return localTrain({ weights, X: cachedStd, y, hyperparameters });

      // Sending an update that was already released costs no extra privacy, so
      // a round re-run after someone else's dropout reuses it.
      const key = `${round}|${weights.join(',')}`;
      if (lastRelease?.key === key) return lastRelease.result;
      const hp = { ...DEFAULT_HYPERPARAMETERS, ...hyperparameters, noiseMultiplier: dp.noiseMultiplier, clipNorm: dp.clipNorm };
      const cost = { ...dpSgdSteps(X.length, hp), noiseMultiplier: dp.noiseMultiplier };
      const epsilon = accountant.epsilon(cost);
      if (epsilon > dp.epsilonBudget) throw new PrivacyBudgetExceededError(name, epsilon, dp.epsilonBudget);
      const result = localTrain({ weights, X: cachedStd, y, hyperparameters: hp });
      accountant.record(cost);
      lastRelease = { key, result };
      return result;
    },
    /**
     * Adapt the shared model to this hospital's patients (see personalization.js).
     * Runs entirely on site; the coordinator never calls it. Returns the
     * personalized model, which in a real deployment stays at the hospital.
     * Under DP, note that the personalized model is fit without noise: the
     * guarantee covers what the hospital sent, not this model.
     */
    personalize(model, options = {}) {
      const result = personalize({ ...options, weights: model.weights, X: standardize(X, model.standardization), y });
      return { ...result, model: { weights: result.weights, standardization: model.standardization } };
    },
    /** This hospital's privacy cost so far: { epsilon, delta, rounds }, or null without DP. */
    privacySpent() {
      return dp ? { epsilon: accountant.epsilon(), delta: dp.delta, rounds: accountant.rounds } : null;
    },
  };
}

/**
 * Wrap a participant so it drops out for rounds in [fromRound, untilRound].
 * mode 'error' fails fast (connection refused); mode 'hang' never answers,
 * which exercises the coordinator's round timeout.
 */
export function withDropout(participant, { fromRound, untilRound = Infinity, mode = 'error' }) {
  return {
    ...participant,
    async train(weights, hyperparameters, context) {
      if (context.round >= fromRound && context.round <= untilRound) {
        if (mode === 'hang') return new Promise(() => {});
        throw new ParticipantUnavailableError(participant.name, context.round);
      }
      return participant.train(weights, hyperparameters, context);
    },
  };
}
