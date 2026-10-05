// The full FedHeart comparison: each hospital alone, federated (with and
// without a dropout), and the pooled upper bound, all scored on the same
// held-out test patients from every site.

import { loadAllSites, publicStandardization, SITE_IDS, SITES } from './data.js';
import { createLocalParticipant, withDropout } from './participant.js';
import { federatedTrain } from './coordinator.js';
import { createPairwiseSecrets, withSecureAggregation } from './secureAggregation.js';
import { trainLocalOnly, trainPooled } from './baselines.js';
import { evaluate, concatData } from './model.js';
import { DEFAULT_ROUNDS } from './localTrain.js';

/** Score a model on every site's test split and on all test patients combined. */
export function evaluateEverywhere(model, sites) {
  const bySite = {};
  for (const id of Object.keys(sites)) bySite[id] = evaluate(model, sites[id].test);
  const all = evaluate(model, concatData(Object.values(sites).map((s) => s.test)));
  return { all, bySite };
}

export function runBaselines(sites, hyperparameters = {}) {
  const ids = Object.keys(sites);
  const localOnly = {};
  for (const id of ids) localOnly[id] = evaluateEverywhere(trainLocalOnly(sites[id].train, hyperparameters), sites);
  const pooled = evaluateEverywhere(trainPooled(ids.map((id) => sites[id].train), hyperparameters), sites);
  return { localOnly, pooled };
}

/**
 * Run federated training over local participants, scoring the global model on
 * the combined test set after every round (so the demo can chart the curve).
 * dropout: { site, fromRound, untilRound?, mode? } simulates a hospital leaving.
 * secureAggregation: hospitals mask their updates so the coordinator only sees sums.
 * privacy: differential privacy settings each hospital applies to itself (see
 * createLocalParticipant); features are then standardized with public ranges.
 * onResponse: (name, response, context), everything the coordinator receives.
 * personalize: afterwards, each hospital adapts the shared model to its own
 * patients; reports which candidate each picked and how it scores on that
 * hospital's own test patients.
 */
export async function runFederated(
  sites,
  hyperparameters = {},
  { dropout, roundTimeoutMs = 2000, onRound, secureAggregation = false, privacy = null, onResponse, personalize = false } = {},
) {
  const ids = Object.keys(sites);
  const secrets = secureAggregation ? createPairwiseSecrets(ids) : null;
  const participants = ids.map((id) => {
    let p = createLocalParticipant(id, sites[id].train, { privacy });
    if (dropout && dropout.site === id) p = withDropout(p, dropout);
    // Each hospital holds only its own row of pairwise secrets.
    return secrets ? withSecureAggregation(p, { secrets: secrets[id] }) : p;
  });
  const curve = [];
  const result = await federatedTrain(hyperparameters, {
    participants,
    roundTimeoutMs,
    secureAggregation,
    onResponse,
    standardization: privacy ? publicStandardization() : undefined,
    onRound: async (entry, model) => {
      const metrics = evaluateEverywhere(model, sites).all;
      curve.push({ round: entry.round, auc: metrics.auc, accuracy: metrics.accuracy, responded: entry.responded });
      if (onRound) await onRound(entry, model, metrics);
    },
  });
  const privacySpent = privacy ? Object.fromEntries(participants.map((p) => [p.name, p.privacySpent()])) : null;
  const personalized = personalize
    ? Object.fromEntries(
        participants.map((p) => {
          const { strategy, label, model, cvLogLoss } = p.personalize(result, { hyperparameters });
          return [p.name, { strategy, label, cvLogLoss, metrics: evaluate(model, sites[p.name].test) }];
        }),
      )
    : null;
  return { model: result, history: result.history, curve, metrics: evaluateEverywhere(result, sites), privacySpent, personalized };
}

export async function runExperiment({
  hyperparameters = {},
  seed = 42,
  testFraction = 0.3,
  dropout = { site: 'va', fromRound: Math.floor((hyperparameters.rounds ?? DEFAULT_ROUNDS) / 2) + 1 },
  secureAggregation = false,
  privacy = null,
  baselineHyperparameters = hyperparameters,
  onRound,
} = {}) {
  const sites = loadAllSites({ seed, testFraction });
  const baselines = runBaselines(sites, baselineHyperparameters);
  const federated = await runFederated(sites, hyperparameters, {
    secureAggregation,
    privacy,
    personalize: true,
    onRound: onRound && ((...a) => onRound('federated', ...a)),
  });
  const federatedDropout = dropout
    ? await runFederated(sites, hyperparameters, {
        dropout,
        secureAggregation,
        privacy,
        onRound: onRound && ((...a) => onRound('dropout', ...a)),
      })
    : null;
  return {
    config: { hyperparameters, seed, testFraction, dropout, secureAggregation, privacy },
    sites: Object.fromEntries(
      Object.keys(sites).map((id) => [
        id,
        { label: SITES[id].label, train: sites[id].train.y.length, test: sites[id].test.y.length,
          diseaseRate: [...sites[id].train.y, ...sites[id].test.y].reduce((a, b) => a + b, 0) /
            (sites[id].train.y.length + sites[id].test.y.length) },
      ]),
    ),
    baselines,
    federated: {
      metrics: federated.metrics, curve: federated.curve, history: federated.history, model: federated.model,
      privacySpent: federated.privacySpent,
    },
    // Each hospital on its own test patients: training alone, the shared model, and personalized.
    perSite: Object.fromEntries(
      Object.keys(sites).map((id) => [
        id,
        {
          alone: baselines.localOnly[id].bySite[id],
          federated: federated.metrics.bySite[id],
          personalized: federated.personalized[id].metrics,
          strategy: federated.personalized[id].strategy,
          strategyLabel: federated.personalized[id].label,
        },
      ]),
    ),
    federatedDropout: federatedDropout && {
      metrics: federatedDropout.metrics,
      curve: federatedDropout.curve,
      history: federatedDropout.history,
      model: federatedDropout.model,
      privacySpent: federatedDropout.privacySpent,
    },
  };
}

export { SITE_IDS };
