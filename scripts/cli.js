// Shared argument parsing and table printing for the scripts.
import { parseArgs } from 'node:util';

export function parseHyperparameters(extraOptions = {}) {
  const { values } = parseArgs({
    options: {
      rounds: { type: 'string' },
      'local-epochs': { type: 'string' },
      'learning-rate': { type: 'string' },
      'batch-size': { type: 'string' },
      l2: { type: 'string' },
      mu: { type: 'string' },
      seed: { type: 'string' },
      ...extraOptions,
    },
  });
  const num = (v) => (v === undefined ? undefined : Number(v));
  const hp = Object.fromEntries(
    Object.entries({
      rounds: num(values.rounds),
      localEpochs: num(values['local-epochs']),
      learningRate: num(values['learning-rate']),
      batchSize: num(values['batch-size']),
      l2: num(values.l2),
      mu: num(values.mu),
    }).filter(([, v]) => v !== undefined),
  );
  return { hp, seed: num(values.seed) ?? 42, values };
}

export function printTable(rows, columns) {
  const widths = columns.map((c) => Math.max(c.length, ...rows.map((r) => String(r[c]).length)));
  const line = (cells) => cells.map((c, i) => String(c).padEnd(widths[i])).join('  ');
  console.log(line(columns));
  console.log(line(widths.map((w) => '-'.repeat(w))));
  for (const r of rows) console.log(line(columns.map((c) => r[c])));
}

export const fmt = (x) => (Number.isFinite(x) ? x.toFixed(3) : 'n/a');

// Differential privacy flags, shared by the federated, demo and privacy scripts.
export const PRIVACY_OPTIONS = {
  dp: { type: 'boolean' },
  noise: { type: 'string' },
  clip: { type: 'string' },
  epsilon: { type: 'string' },
  delta: { type: 'string' },
};

// Training settings that work well under DP: AUC levels off within a few rounds
// and every round costs privacy, so train for fewer rounds with a larger step.
export const DP_HYPERPARAMETERS = { rounds: 10, learningRate: 0.1 };

/** Returns { privacy, hp } with DP defaults filled in, or { privacy: null, hp } without --dp. */
export function parsePrivacy(values, hp) {
  if (!values.dp) return { privacy: null, hp };
  const privacy = {
    noiseMultiplier: Number(values.noise ?? 3),
    clipNorm: Number(values.clip ?? 1),
    delta: Number(values.delta ?? 1e-5),
    epsilonBudget: values.epsilon === undefined ? Infinity : Number(values.epsilon),
  };
  return { privacy, hp: { ...DP_HYPERPARAMETERS, ...hp } };
}

/** One line per hospital with the epsilon it spent. */
export function formatPrivacySpent(privacySpent, label) {
  return Object.entries(privacySpent)
    .map(([id, p]) => `  ${label(id).padEnd(14)} epsilon ${p.epsilon.toFixed(2)} (delta ${p.delta}, ${p.rounds} rounds)`)
    .join('\n');
}
