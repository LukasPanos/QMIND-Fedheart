// The Node side of a dummy-DCP hospital worker: what needs Node's crypto, on
// top of the portable createHospital. It adds secure aggregation, entirely on
// the hospital's side:
//   - its own X25519 key pair, whose public half goes to the coordinator;
//   - pairwise secrets with the other hospitals, derived from their public
//     keys (which the coordinator relays) and never sent anywhere;
//   - masking every update with them before it leaves.
// scripts/dummy-dcp-worker.js runs this in a process per hospital folder, and
// test/hub.test.js drives it directly.

import { createHospital } from './hospital.js';
import { createKeyAgreement, withSecureAggregation } from './secureAggregation.js';

const fixed = (x, digits = 3) => (Number.isFinite(x) ? x.toFixed(digits) : 'n/a');

export function createHospitalWorker(site, data) {
  const hospital = createHospital(data, { name: site });
  const keys = createKeyAgreement();
  const secrets = {}; // peer -> shared secret
  const keyOf = {}; // peer -> the public key its secret came from
  const secure = withSecureAggregation(
    {
      name: site,
      stats: () => hospital.handle({ type: 'stats' }),
      train: (weights, hyperparameters, context) => hospital.handle({ type: 'train', weights, hyperparameters, ...context }),
    },
    { secrets },
  );

  // A peer that restarted has a new key pair, so re-derive when its key changes.
  function learnKeys(publicKeys = {}) {
    for (const [peer, key] of Object.entries(publicKeys)) {
      if (peer === site || keyOf[peer] === key) continue;
      secrets[peer] = keys.secretWith(key);
      keyOf[peer] = key;
    }
  }

  async function run(task) {
    // Under secure aggregation the coordinator sends the cohort and its members' public keys.
    if (task.cohort && (task.type === 'stats' || task.type === 'train')) {
      learnKeys(task.publicKeys);
      const context = { round: task.round ?? 0, cohort: task.cohort, attempt: task.attempt, session: task.session, standardization: task.standardization };
      return task.type === 'stats' ? secure.stats(context) : secure.train(task.weights, task.hyperparameters, context);
    }
    return hospital.handle(task);
  }

  /** A line for the hospital's log describing what it just sent, or null. */
  async function describe(task, result) {
    const nTrain = data.train.y.length;
    switch (task.type) {
      case 'configure': {
        const p = result.settings;
        return p
          ? `Privacy settings for the next run: differential privacy on (noise multiplier ${p.noiseMultiplier}, clip norm ${p.clipNorm}` +
              `${Number.isFinite(p.epsilonBudget) ? `, budget epsilon ${p.epsilonBudget}` : ''}).`
          : 'Privacy settings for the next run: differential privacy off.';
      }
      case 'stats':
        return result.masked
          ? `Shared masked feature statistics (${result.masked.length} numbers that only add up across hospitals). 0 patient rows sent.`
          : `Shared feature statistics for ${result.count.length} features. 0 patient rows sent.`;
      case 'train': {
        const status = await hospital.handle({ type: 'privacy' });
        const dp = status.dp ? `with DP noise (epsilon so far ${fixed(status.spent?.epsilon, 2)})` : '';
        const trained = `Round ${task.round}: trained ${task.hyperparameters.localEpochs} epochs on ${nTrain} patients${dp ? ` ${dp}` : ''}`;
        if (result.masked) {
          return `${trained}. Sent back ${result.masked.length} masked numbers` +
            ` (cohort of ${task.cohort.length}${task.attempt > 1 ? `, re-run ${task.attempt}` : ''}), 0 patient rows.`;
        }
        const loss = Number.isFinite(result.loss) ? `, loss ${fixed(result.loss)}` : ', loss not reported';
        return `${trained}${loss}. Sent back ${result.weights.length} weights, 0 patient rows.`;
      }
      case 'personalize':
        return `Personalized the shared model on site: chose "${result.label}". Holdout log loss ${fixed(result.shared.logLoss)} -> ` +
          `${fixed(result.personalized.logLoss)}. The personalized model stays here; only the scores were sent.`;
      default:
        return null;
    }
  }

  return {
    publicKey: keys.publicKey,
    /** Answer a task. Returns { result, log, privacy }: privacy is the status after train/configure. */
    async handle(task) {
      const result = await run(task);
      const log = await describe(task, result);
      const privacy = task.type === 'train' || task.type === 'configure' ? await hospital.handle({ type: 'privacy' }) : null;
      return { result, log, privacy };
    },
  };
}
