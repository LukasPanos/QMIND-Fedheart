// The coordinator: runs the train-locally, average-centrally loop (FedAvg).
// It never sees a patient row, only what participants choose to return.

import { combineStats } from './data.js';
import { DEFAULT_HYPERPARAMETERS, DEFAULT_ROUNDS } from './localTrain.js';
import { zeroWeights } from './model.js';
import { isMaskedPayload, sumMasked } from './secureAggregation.js';

/** Weighted average of participant weights, weighted by their sample counts. */
export function fedAvg(updates) {
  const total = updates.reduce((s, u) => s + u.numSamples, 0);
  if (total <= 0) throw new Error('fedAvg needs at least one update with numSamples > 0');
  const out = new Array(updates[0].weights.length).fill(0);
  for (const u of updates) {
    const share = u.numSamples / total;
    u.weights.forEach((w, j) => {
      out[j] += share * w;
    });
  }
  return out;
}

function withTimeout(promise, ms, label) {
  if (!Number.isFinite(ms)) return promise;
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms} ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/** Call every participant with a timeout; onResponse sees each answer as it arrives. */
function ask(participants, call, { timeoutMs, label, onResponse }) {
  return Promise.allSettled(
    participants.map((p) =>
      withTimeout(
        // Promise.resolve().then(...) turns a synchronous throw into a rejection.
        Promise.resolve().then(() => call(p)),
        timeoutMs,
        `${p.name} ${label}`,
      ).then((value) => {
        if (onResponse) onResponse(p.name, value);
        return value;
      }),
    ),
  );
}

const reasonOf = (s) => (s.status === 'rejected' ? s.reason?.message ?? String(s.reason) : 'invalid update');

function mostCommon(values) {
  const tally = new Map();
  for (const v of values) tally.set(v, (tally.get(v) ?? 0) + 1);
  return [...tally.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
}

/**
 * Secure aggregation: ask a cohort for masked vectors and return their sum.
 * Masks only cancel when every member answers, so after a dropout the rest are
 * asked again with the smaller cohort, until everyone answers or fewer than
 * minCohort are left. `length` is the expected vector length (by default the
 * most common one in the first answers).
 */
async function secureSum(participants, call, { length, minCohort, timeoutMs, label, onResponse, session }) {
  let cohort = participants;
  let attempts = 0;
  const dropped = [];
  while (cohort.length >= minCohort) {
    attempts++;
    const context = { cohort: cohort.map((p) => p.name), attempt: attempts, session };
    const settled = await ask(cohort, (p) => call(p, context), {
      timeoutMs,
      label,
      onResponse: onResponse && ((name, value) => onResponse(name, value, context)),
    });
    length ??= mostCommon(settled.filter((s) => s.status === 'fulfilled').map((s) => s.value?.masked?.length));
    const ok = settled.map((s) => s.status === 'fulfilled' && isMaskedPayload(s.value?.masked, length));
    settled.forEach((s, i) => {
      if (!ok[i]) dropped.push({ name: cohort[i].name, reason: reasonOf(s) });
    });
    if (ok.every(Boolean)) {
      return { sum: sumMasked(settled.map((s) => s.value.masked)), responded: context.cohort, dropped, attempts };
    }
    cohort = cohort.filter((_, i) => ok[i]);
  }
  return { sum: null, responded: [], dropped, attempts };
}

function secureOptions(secureAggregation, minParticipants = 1) {
  if (!secureAggregation) return null;
  const { minCohort = 3 } = secureAggregation === true ? {} : secureAggregation;
  return { minCohort: Math.max(minCohort, minParticipants) };
}

function isValidUpdate(u, length) {
  return (
    u &&
    Array.isArray(u.weights) &&
    u.weights.length === length &&
    u.weights.every(Number.isFinite) &&
    Number.isFinite(u.numSamples) &&
    u.numSamples > 0
  );
}

/**
 * Ask every participant for feature statistics and agree on one standardization.
 * With secureAggregation, the coordinator only sees the summed statistics.
 */
export async function federatedStandardization(
  participants,
  { timeoutMs = Infinity, secureAggregation, onResponse, session = crypto.randomUUID() } = {},
) {
  const secure = secureOptions(secureAggregation);
  if (secure) {
    const { sum, dropped } = await secureSum(participants, (p, context) => p.stats({ round: 0, ...context }), {
      minCohort: secure.minCohort,
      timeoutMs,
      label: 'stats',
      onResponse,
      session,
    });
    if (!sum || sum.length % 3 !== 0) {
      const reasons = dropped.map((d) => `${d.name}: ${d.reason}`);
      throw new Error(`Secure aggregation of feature statistics failed (${reasons.join('; ')})`);
    }
    const d = sum.length / 3;
    return combineStats([{ count: sum.slice(0, d), sum: sum.slice(d, 2 * d), sumSq: sum.slice(2 * d) }]);
  }

  const settled = await ask(participants, (p) => p.stats({ round: 0 }), {
    timeoutMs,
    label: 'stats',
    onResponse: onResponse && ((name, value) => onResponse(name, value, { round: 0 })),
  });
  const responses = settled.filter((s) => s.status === 'fulfilled').map((s) => s.value);
  const valid = (s) =>
    s && ['count', 'sum', 'sumSq'].every((k) => Array.isArray(s[k]) && s[k].every(Number.isFinite));
  // A malformed response must not poison everyone's standardization: keep only
  // well-formed stats with the most common feature count.
  const d = mostCommon(responses.filter(valid).map((s) => s.count.length));
  const stats = responses.filter((s) => valid(s) && s.count.length === d && s.sum.length === d && s.sumSq.length === d);
  if (stats.length === 0) {
    const reasons = settled.map((s, i) => `${participants[i].name}: ${s.status === 'rejected' ? s.reason?.message ?? s.reason : 'invalid stats'}`);
    throw new Error(`No participant returned valid feature statistics (${reasons.join('; ')})`);
  }
  return combineStats(stats);
}

/**
 * Train one shared model across participants without moving their data.
 *
 * @param {object} hyperparameters   see DEFAULT_HYPERPARAMETERS; plus `rounds`
 * @param {object} options
 * @param {Array}  options.participants      objects with stats() and train()
 * @param {number} [options.minParticipants=1] fewer responses than this and the
 *                                           round is skipped (weights unchanged)
 * @param {number} [options.roundTimeoutMs]  a participant that has not answered
 *                                           by then counts as dropped for the round
 * @param {object} [options.standardization] skip the stats step and use this
 * @param {number[]} [options.initialWeights]
 * @param {Function} [options.onRound]       awaited after each round with
 *                                           (entry, model)
 * @param {boolean|object} [options.secureAggregation] participants return
 *                                           masked vectors (see withSecureAggregation)
 *                                           and the coordinator only learns their sum;
 *                                           { minCohort = 3 } sets the smallest cohort
 * @param {Function} [options.onResponse]    called with (name, response, context) for
 *                                           every answer, i.e. what the coordinator sees
 * @returns {Promise<{weights, standardization, history}>}
 */
export async function federatedTrain(hyperparameters = {}, options = {}) {
  const { rounds = DEFAULT_ROUNDS, ...localHp } = hyperparameters;
  const hp = { ...DEFAULT_HYPERPARAMETERS, ...localHp };
  const {
    participants,
    minParticipants = 1,
    roundTimeoutMs = Infinity,
    onRound,
    secureAggregation,
    onResponse,
  } = options;
  const secure = secureOptions(secureAggregation, minParticipants);
  // Fresh for every run, so hospitals never reuse masks from an earlier run.
  const session = crypto.randomUUID();
  if (!participants || participants.length === 0) throw new Error('federatedTrain needs at least one participant');

  const standardization =
    options.standardization ??
    (await federatedStandardization(participants, { timeoutMs: roundTimeoutMs, secureAggregation, onResponse, session }));
  let weights = options.initialWeights ? options.initialWeights.slice() : zeroWeights(standardization.mean.length);
  const history = [];
  // Each participant gets its own deterministic seed per round.
  const hpFor = (p, round) => ({ ...hp, seed: hp.seed + round * 1000 + participants.indexOf(p) });

  async function plainRound(round) {
    const context = { round, standardization };
    const settled = await ask(participants, (p) => p.train(weights, hpFor(p, round), context), {
      timeoutMs: roundTimeoutMs,
      label: `round ${round}`,
      onResponse: onResponse && ((name, value) => onResponse(name, value, context)),
    });

    const updates = [];
    const responded = [];
    const dropped = [];
    settled.forEach((s, i) => {
      const name = participants[i].name;
      if (s.status === 'fulfilled' && isValidUpdate(s.value, weights.length)) {
        updates.push(s.value);
        responded.push(name);
      } else {
        dropped.push({ name, reason: reasonOf(s) });
      }
    });

    const skipped = updates.length < minParticipants;
    if (!skipped) weights = fedAvg(updates);

    // Hospitals under differential privacy report no loss (NaN).
    const withLoss = updates.filter((u) => Number.isFinite(u.loss));
    const lossSamples = withLoss.reduce((s, u) => s + u.numSamples, 0);
    return {
      round,
      responded,
      dropped,
      skipped,
      trainLoss: lossSamples > 0 ? withLoss.reduce((s, u) => s + u.loss * u.numSamples, 0) / lossSamples : NaN,
    };
  }

  // The masked vector is [n * weights, n, n * loss, n if a loss was reported],
  // so the sums give the sample-weighted average (FedAvg) and the mean training loss.
  async function secureRound(round) {
    const d = weights.length;
    const { sum, responded, dropped, attempts } = await secureSum(
      participants,
      (p, context) => p.train(weights, hpFor(p, round), { round, standardization, ...context }),
      { length: d + 3, minCohort: secure.minCohort, timeoutMs: roundTimeoutMs, label: `round ${round}`, session,
        onResponse: onResponse && ((name, value, context) => onResponse(name, value, { round, ...context })) },
    );
    const total = sum ? sum[d] : 0;
    const next = sum && total > 0 ? sum.slice(0, d).map((v) => v / total) : null;
    const skipped = !next || !next.every(Number.isFinite);
    if (!skipped) weights = next;
    return {
      round,
      responded: skipped ? [] : responded,
      dropped,
      skipped,
      attempts,
      trainLoss: !skipped && sum[d + 2] > 0 ? sum[d + 1] / sum[d + 2] : NaN,
    };
  }

  for (let round = 1; round <= rounds; round++) {
    const entry = secure ? await secureRound(round) : await plainRound(round);
    history.push(entry);
    if (onRound) await onRound(entry, { weights: weights.slice(), standardization });
  }

  return { weights, standardization, history };
}
