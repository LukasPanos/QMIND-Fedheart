// Serves each hospital's training rows as JSON, standing in for the data
// endpoint inside each hospital's network that its own DCP workers read from.
//   node scripts/serve-sites.js [--port 8787]     ->  http://localhost:8787/<site>.json
// For real use, every hospital serves only its own file, reachable only from
// its own compute group's workers.
import { createServer } from 'node:http';
import { parseArgs } from 'node:util';
import { loadAllSites, SITE_IDS } from '../src/data.js';

const { values } = parseArgs({ options: { port: { type: 'string' }, seed: { type: 'string' } } });
const port = Number(values.port ?? 8787);
const sites = loadAllSites({ seed: Number(values.seed ?? 42) });

createServer((req, res) => {
  const id = (req.url ?? '').replace(/^\//, '').replace(/\.json$/, '');
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (!sites[id]) {
    res.writeHead(404, { 'content-type': 'text/plain' }).end(`Unknown site. Try: ${SITE_IDS.map((s) => `/${s}.json`).join(' ')}`);
    return;
  }
  res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(sites[id].train));
}).listen(port, () => console.log(`Serving ${SITE_IDS.map((s) => `http://localhost:${port}/${s}.json`).join(', ')}`));
