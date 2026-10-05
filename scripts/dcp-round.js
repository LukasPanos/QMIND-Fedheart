// Stretch goal: federated training where each hospital's local step runs on DCP.
//   npm install dcp-client
//   node scripts/serve-sites.js &      (or each hospital's real data URL)
//   npm run dcp -- --rounds 1 --data-url "http://localhost:8787/{site}.json"
//   npm run dcp -- --local     (job.localExec: slices run in this process; needs `npm install dcp-worker`)
//
// Compute groups: set FEDHEART_<SITE>_JOIN_KEY / FEDHEART_<SITE>_JOIN_SECRET
// (e.g. FEDHEART_VA_JOIN_KEY) to pin a hospital's slices to its own workers.
import { SITE_IDS, SITES } from '../src/data.js';
import { federatedTrain } from '../src/coordinator.js';
import { createDcpParticipant } from '../src/dcp.js';
import { parseHyperparameters, fmt } from './cli.js';

const { hp, values } = parseHyperparameters({
  'data-url': { type: 'string' },
  sites: { type: 'string' },
  scheduler: { type: 'string' },
  'timeout-ms': { type: 'string' },
  local: { type: 'boolean' },
});
const template = values['data-url'] ?? 'http://localhost:8787/{site}.json';
const ids = values.sites ? values.sites.split(',') : SITE_IDS;

const participants = ids.map((id) => {
  const key = process.env[`FEDHEART_${id.toUpperCase()}_JOIN_KEY`];
  const secret = process.env[`FEDHEART_${id.toUpperCase()}_JOIN_SECRET`];
  return createDcpParticipant(id, {
    dataUrl: template.replace('{site}', id),
    computeGroup: key ? { joinKey: key, joinSecret: secret } : undefined,
    scheduler: values.scheduler,
    localExec: values.local,
  });
});

const result = await federatedTrain(
  { rounds: 1, ...hp },
  {
    participants,
    roundTimeoutMs: Number(values['timeout-ms'] ?? 10 * 60 * 1000),
    onRound: (entry) => {
      console.log(`round ${entry.round}: ${entry.responded.map((id) => SITES[id].label).join(', ') || 'nobody'} responded, train loss ${fmt(entry.trainLoss)}`);
      for (const d of entry.dropped) console.log(`  dropped ${SITES[d.name]?.label ?? d.name}: ${d.reason}`);
    },
  },
);
console.log(JSON.stringify({ weights: result.weights, standardization: result.standardization }, null, 2));
