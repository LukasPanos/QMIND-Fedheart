// Loading the UCI Heart Disease per-site files from disk. The cleaning and
// featurizing itself lives in features.js (no Node imports, so browsers can
// use it too) and is re-exported here.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { SITES, SITE_IDS, parseSiteFile, featurizeAll, trainTestSplit } from './features.js';

export * from './features.js';

const here = dirname(fileURLToPath(import.meta.url));
export const DATA_DIR = join(here, '..', 'data', 'raw');

export function loadSite(siteId, dataDir = DATA_DIR) {
  const site = SITES[siteId];
  if (!site) throw new Error(`Unknown site "${siteId}". Known: ${SITE_IDS.join(', ')}`);
  return parseSiteFile(readFileSync(join(dataDir, site.file), 'utf8'));
}

/**
 * Load every site, featurize it and split it. Returns { [siteId]: { train, test } }.
 * In a real deployment each hospital would call loadSite + featurizeAll +
 * trainTestSplit for itself; this helper exists for local simulation.
 */
export function loadAllSites({ testFraction = 0.3, seed = 42, dataDir = DATA_DIR, sites = SITE_IDS } = {}) {
  const out = {};
  for (const id of sites) out[id] = trainTestSplit(featurizeAll(loadSite(id, dataDir)), { testFraction, seed });
  return out;
}
