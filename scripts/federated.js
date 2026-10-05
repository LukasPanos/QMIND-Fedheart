// Milestone 2: federated training across the four hospitals, optionally with a dropout.
//   npm run federated [-- --drop va --drop-round 16 --rounds 30]
//   --secure      secure aggregation: the coordinator only learns the sum of updates
//   --show-view   print exactly what the coordinator received from each hospital in round 1
//   --dp          differential privacy (DP-SGD at each hospital); tune with
//                 --noise 3 --clip 1 --epsilon <budget> --delta 1e-5
import { loadAllSites, SITES } from '../src/data.js';
import { runFederated } from '../src/experiment.js';
import { parseHyperparameters, parsePrivacy, formatPrivacySpent, PRIVACY_OPTIONS, fmt } from './cli.js';

const { hp: baseHp, seed, values } = parseHyperparameters({
  ...PRIVACY_OPTIONS,
  drop: { type: 'string' },
  'drop-round': { type: 'string' },
  'drop-mode': { type: 'string' },
  secure: { type: 'boolean' },
  'show-view': { type: 'boolean' },
});
const secureAggregation = Boolean(values.secure);
const { privacy, hp } = parsePrivacy(values, baseHp);
const sites = loadAllSites({ seed });
const dropout = values.drop
  ? { site: values.drop, fromRound: Number(values['drop-round'] ?? 16), mode: values['drop-mode'] ?? 'error' }
  : undefined;
if (dropout && !SITES[dropout.site]) throw new Error(`Unknown site ${dropout.site}`);

const label = (id) => SITES[id].label;
const preview = (xs) => `${xs.slice(0, 3).join('  ')}  ... (${xs.length} values)`;
const view = [];
function onResponse(name, value, context) {
  if (!values['show-view'] || context.round !== 1 || (context.attempt ?? 1) !== 1) return;
  view.push(`  ${label(name).padEnd(14)} ${value.masked ? preview(value.masked) : preview(value.weights.map((w) => w.toFixed(4)))}`);
}

if (secureAggregation) console.log('Secure aggregation on: the coordinator only learns the sum of the updates.\n');
if (privacy) {
  console.log(
    `Differential privacy on: each hospital clips every patient's gradient to ${privacy.clipNorm} ` +
      `and adds noise (multiplier ${privacy.noiseMultiplier}).\n`,
  );
}
const { metrics, privacySpent } = await runFederated(sites, hp, {
  dropout,
  secureAggregation,
  privacy,
  onResponse,
  roundTimeoutMs: 500,
  onRound: (entry, _model, m) => {
    if (entry.round === 1 && view.length) {
      console.log(`What the coordinator received in round 1 (${secureAggregation ? 'masked' : 'weights in the clear'}):`);
      console.log(`${view.join('\n')}\n`);
    }
    const dropped = entry.dropped.length ? `  dropped: ${entry.dropped.map((d) => label(d.name)).join(', ')}` : '';
    const rerun = entry.attempts > 1 ? '  (re-run without the dropped site)' : '';
    console.log(
      `round ${String(entry.round).padStart(2)}  sites ${entry.responded.length}/4  ` +
        `train loss ${fmt(entry.trainLoss)}  test AUC ${fmt(m.auc)}  acc ${fmt(m.accuracy)}${dropped}${rerun}`,
    );
  },
});
console.log(`\nFinal federated model, all test patients: AUC ${fmt(metrics.all.auc)}, accuracy ${fmt(metrics.all.accuracy)}`);
if (privacySpent) console.log(`\nPrivacy spent by each hospital:\n${formatPrivacySpent(privacySpent, label)}`);
