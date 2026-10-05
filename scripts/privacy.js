// The privacy / accuracy trade-off: federated training with differential
// privacy at several noise levels, each repeated because DP noise is random.
//   npm run privacy [-- --repeats 10 --clip 1 --delta 1e-5]
import { writeFileSync, mkdirSync } from 'node:fs';
import { loadAllSites, SITES } from '../src/data.js';
import { runFederated, runBaselines } from '../src/experiment.js';
import { parseHyperparameters, printTable, fmt, DP_HYPERPARAMETERS } from './cli.js';

const { hp: baseHp, seed, values } = parseHyperparameters({
  repeats: { type: 'string' },
  clip: { type: 'string' },
  delta: { type: 'string' },
});
const repeats = Number(values.repeats ?? 10);
const clipNorm = Number(values.clip ?? 1);
const delta = Number(values.delta ?? 1e-5);
const hp = { ...DP_HYPERPARAMETERS, ...baseHp };
const NOISE = [0.5, 0.75, 1, 1.5, 2, 3, 5, 8];

const sites = loadAllSites({ seed });
const baselines = runBaselines(sites);
const bestLocal = Object.entries(baselines.localOnly).sort((a, b) => b[1].all.auc - a[1].all.auc)[0];
const noDp = (await runFederated(sites)).metrics.all;

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = (xs) => Math.sqrt(mean(xs.map((x) => (x - mean(xs)) ** 2)));
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const results = [];
console.log(`Federated training with DP: ${hp.rounds} rounds, clip norm ${clipNorm}, delta ${delta}, ${repeats} repeats per row\n`);
for (const noiseMultiplier of NOISE) {
  const runs = [];
  for (let r = 0; r < repeats; r++) runs.push(await runFederated(sites, hp, { privacy: { noiseMultiplier, clipNorm, delta } }));
  const aucs = runs.map((x) => x.metrics.all.auc);
  const epsilons = runs[0].privacySpent;
  results.push({
    noiseMultiplier,
    epsilon: Object.fromEntries(Object.entries(epsilons).map(([id, p]) => [id, p.epsilon])),
    maxEpsilon: Math.max(...Object.values(epsilons).map((p) => p.epsilon)),
    auc: mean(aucs),
    aucSd: sd(aucs),
    aucMin: Math.min(...aucs),
    aucMax: Math.max(...aucs),
    accuracy: mean(runs.map((x) => x.metrics.all.accuracy)),
  });
  process.stdout.write('.');
}
console.log('\n');

printTable(
  [
    { setting: 'No DP (30 rounds)', 'worst-hospital epsilon': 'none', 'mean AUC': fmt(noDp.auc), 'AUC range': '', accuracy: pct(noDp.accuracy) },
    ...results.map((r) => ({
      setting: `noise multiplier ${r.noiseMultiplier}`,
      'worst-hospital epsilon': r.maxEpsilon.toFixed(2),
      'mean AUC': fmt(r.auc),
      'AUC range': `${fmt(r.aucMin)}-${fmt(r.aucMax)}`,
      accuracy: pct(r.accuracy),
    })),
    {
      setting: `${SITES[bestLocal[0]].label} alone (best single hospital, no DP)`,
      'worst-hospital epsilon': 'none',
      'mean AUC': fmt(bestLocal[1].all.auc),
      'AUC range': '',
      accuracy: pct(bestLocal[1].all.accuracy),
    },
  ],
  ['setting', 'worst-hospital epsilon', 'mean AUC', 'AUC range', 'accuracy'],
);
console.log('\nEpsilon is per hospital; Switzerland, the smallest, always has the largest. Lower epsilon is more private.');

mkdirSync(new URL('../results/', import.meta.url), { recursive: true });
writeFileSync(
  new URL('../results/privacy.json', import.meta.url),
  JSON.stringify({ config: { hp, clipNorm, delta, repeats, seed }, noDp, bestLocal: { site: bestLocal[0], ...bestLocal[1].all }, results }, null, 2),
);
console.log('Wrote results/privacy.json');
