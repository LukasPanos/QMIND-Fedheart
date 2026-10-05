import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  loadSite, loadAllSites, SITE_IDS, FEATURES, parseSiteFile, featurize, trainTestSplit, featurizeAll,
  localStats, combineStats, standardize,
} from '../src/data.js';

test('each site file loads with the expected number of patients', () => {
  const expected = { cleveland: 303, hungarian: 294, switzerland: 123, va: 200 };
  for (const id of SITE_IDS) assert.equal(loadSite(id).length, expected[id], id);
});

test('parsing turns ? and placeholder zeros into null', () => {
  const [r] = parseSiteFile('32,1,1,95,0,?,0,127,0,.7,1,?,?,1\n');
  assert.equal(r.chol, null);
  assert.equal(r.fbs, null);
  assert.equal(r.oldpeak, 0.7);
  assert.equal(r.num, 1);
});

test('featurize produces one value per feature and a binary label', () => {
  const records = loadSite('switzerland');
  for (const r of records) {
    const { x, y } = featurize(r);
    assert.equal(x.length, FEATURES.length);
    assert.ok(y === 0 || y === 1);
  }
  // Switzerland never measured cholesterol.
  const chol = FEATURES.indexOf('chol');
  assert.ok(records.every((r) => featurize(r).x[chol] === null));
  assert.ok(records.every((r) => featurize(r).x[chol + 1] === 1));
});

test('train/test split is stratified, disjoint in size, and reproducible', () => {
  const data = featurizeAll(loadSite('cleveland'));
  const a = trainTestSplit(data, { seed: 3 });
  const b = trainTestSplit(data, { seed: 3 });
  assert.deepEqual(a, b);
  assert.equal(a.train.y.length + a.test.y.length, data.y.length);
  const rate = (ys) => ys.reduce((s, v) => s + v, 0) / ys.length;
  assert.ok(Math.abs(rate(a.train.y) - rate(a.test.y)) < 0.03);
});

test('federated standardization equals pooled standardization', () => {
  const sites = loadAllSites();
  const fromSites = combineStats(SITE_IDS.map((id) => localStats(sites[id].train.X)));
  const pooled = combineStats([localStats(SITE_IDS.flatMap((id) => sites[id].train.X))]);
  fromSites.mean.forEach((m, j) => assert.ok(Math.abs(m - pooled.mean[j]) < 1e-9));
  fromSites.std.forEach((s, j) => assert.ok(Math.abs(s - pooled.std[j]) < 1e-9));
});

test('standardize maps missing values to 0', () => {
  const out = standardize([[null, 3]], { mean: [5, 1], std: [2, 2] });
  assert.deepEqual(out, [[0, 1]]);
});
