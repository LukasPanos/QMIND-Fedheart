// Milestone 1: each hospital training alone vs. all data pooled together.
//   npm run baselines [-- --rounds 30 --local-epochs 2 --seed 42]
import { loadAllSites, SITES, SITE_IDS } from '../src/data.js';
import { runBaselines } from '../src/experiment.js';
import { parseHyperparameters, printTable, fmt } from './cli.js';

const { hp, seed } = parseHyperparameters();
const sites = loadAllSites({ seed });
const { localOnly, pooled } = runBaselines(sites, hp);

const row = (name, r) => ({
  model: name,
  'AUC (all test)': fmt(r.all.auc),
  'acc (all test)': fmt(r.all.accuracy),
  ...Object.fromEntries(SITE_IDS.map((id) => [`AUC ${SITES[id].label}`, fmt(r.bySite[id].auc)])),
});
const rows = [
  ...SITE_IDS.map((id) => row(`${SITES[id].label} alone`, localOnly[id])),
  row('Pooled (upper bound)', pooled),
];
console.log(`Baselines, test split = 30% of each site, seed ${seed}\n`);
printTable(rows, Object.keys(rows[0]));
