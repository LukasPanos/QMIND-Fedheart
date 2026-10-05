// The Sunday demo: baselines, federated training with a mid-run dropout, a
// summary table, and results/chart.html.
//   npm run demo [-- --delay 150 --drop va --drop-round 16 --secure --dp]
import { writeFileSync, mkdirSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';
import { SITES } from '../src/data.js';
import { runExperiment } from '../src/experiment.js';
import { renderChartHtml } from '../src/chart.js';
import { DEFAULT_ROUNDS } from '../src/localTrain.js';
import { parseHyperparameters, parsePrivacy, formatPrivacySpent, PRIVACY_OPTIONS, printTable, fmt } from './cli.js';

const { hp: baseHp, seed, values } = parseHyperparameters({
  ...PRIVACY_OPTIONS,
  delay: { type: 'string' },
  drop: { type: 'string' },
  'drop-round': { type: 'string' },
  'no-drop': { type: 'boolean' },
  secure: { type: 'boolean' },
});
const secureAggregation = Boolean(values.secure);
const { privacy, hp } = parsePrivacy(values, baseHp);
const delay = Number(values.delay ?? 0);
const rounds = hp.rounds ?? DEFAULT_ROUNDS;
const dropout = values['no-drop']
  ? null
  : { site: values.drop ?? 'va', fromRound: Number(values['drop-round'] ?? Math.floor(rounds / 2) + 1) };

console.log('FedHeart: four hospitals, one model, no patient data shared\n');
if (secureAggregation) console.log("Secure aggregation on: the coordinator never sees any one hospital's update\n");
if (privacy) console.log(`Differential privacy on: noise multiplier ${privacy.noiseMultiplier}, clip norm ${privacy.clipNorm}, ${hp.rounds} rounds\n`);
const results = await runExperiment({
  hyperparameters: hp,
  baselineHyperparameters: baseHp,
  seed,
  dropout,
  secureAggregation,
  privacy,
  onRound: async (run, entry, _model, m) => {
    if (run !== 'dropout') return;
    for (const d of entry.dropped) {
      if (entry.round === dropout.fromRound) console.log(`  !! ${SITES[d.name].label} went offline (${d.reason}). Continuing with the rest.`);
      if (entry.round === dropout.fromRound && entry.attempts > 1) console.log('     Masks only cancel when everyone answers, so this round was re-run with the other 3.');
    }
    console.log(
      `  round ${String(entry.round).padStart(2)}/${rounds}  hospitals ${entry.responded.length}/4  ` +
        `test AUC ${fmt(m.auc)}  accuracy ${(m.accuracy * 100).toFixed(1)}%`,
    );
    if (delay) await sleep(delay);
  },
});

const r = results;
const rows = [
  ...Object.keys(r.sites).map((id) => ({ model: `${SITES[id].label} alone`, ...metrics(r.baselines.localOnly[id].all) })),
  { model: `Federated (all 4)${privacy ? ', with DP' : ''}`, ...metrics(r.federated.metrics.all) },
  ...(r.federatedDropout
    ? [{ model: `Federated, ${SITES[dropout.site].label} drops at round ${dropout.fromRound}${privacy ? ', with DP' : ''}`, ...metrics(r.federatedDropout.metrics.all) }]
    : []),
  { model: 'Pooled data (upper bound)', ...metrics(r.baselines.pooled.all) },
];
const pct = (x) => `${(x * 100).toFixed(1)}%`;
function metrics(m) {
  return { 'ROC AUC': fmt(m.auc), accuracy: `${(m.accuracy * 100).toFixed(1)}%` };
}
console.log(`\nAll ${Object.values(r.sites).reduce((s, x) => s + x.test, 0)} held-out test patients:\n`);
printTable(rows, ['model', 'ROC AUC', 'accuracy']);

const bestLocal = Math.max(...Object.values(r.baselines.localOnly).map((x) => x.all.auc));
const fed = r.federated.metrics.all.auc;
console.log(
  `\nFederated vs best hospital alone: ${fed > bestLocal ? '+' : ''}${(fed - bestLocal).toFixed(3)} AUC. ` +
    `Gap to pooled upper bound: ${(r.baselines.pooled.all.auc - fed).toFixed(3)}.`,
);
console.log('\nPersonalization: each hospital adapts the shared model to its own patients, on site.');
console.log('Scored on each hospital\'s own test patients. Log loss: lower is better (how trustworthy the risk percentages are).\n');
printTable(
  Object.entries(r.perSite).map(([id, s]) => ({
    hospital: SITES[id].label,
    'chose': s.strategyLabel,
    'alone': `${pct(s.alone.accuracy)}  ${s.alone.logLoss.toFixed(3)}`,
    'shared model': `${pct(s.federated.accuracy)}  ${s.federated.logLoss.toFixed(3)}`,
    'personalized': `${pct(s.personalized.accuracy)}  ${s.personalized.logLoss.toFixed(3)}`,
  })),
  ['hospital', 'chose', 'alone', 'shared model', 'personalized'],
);
console.log('(accuracy, log loss)');

if (privacy) {
  console.log(`\nPrivacy spent by each hospital (federated, all 4):\n${formatPrivacySpent(r.federated.privacySpent, (id) => SITES[id].label)}`);
  console.log('The baselines and pooled model above are trained without DP.');
}

mkdirSync(new URL('../results/', import.meta.url), { recursive: true });
writeFileSync(new URL('../results/results.json', import.meta.url), JSON.stringify(results, null, 2));
writeFileSync(new URL('../results/chart.html', import.meta.url), renderChartHtml(results));
console.log('\nWrote results/results.json and results/chart.html');
