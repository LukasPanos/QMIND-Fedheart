// Cleaning and featurizing the UCI Heart Disease per-site files, without
// touching the filesystem, so it runs in Node and in a browser alike (the
// dummy-DCP page imports FEATURES from it directly).
//
// Every function here runs *inside* a hospital: it reads that hospital's own
// rows and never needs another site's. The only thing that leaves a site is
// aggregate statistics (see localStats) and model weights.

export const SITES = {
  cleveland: { label: 'Cleveland', file: 'processed.cleveland.data' },
  hungarian: { label: 'Budapest', file: 'processed.hungarian.data' },
  switzerland: { label: 'Switzerland', file: 'processed.switzerland.data' },
  va: { label: 'Long Beach VA', file: 'processed.va.data' },
};
export const SITE_IDS = Object.keys(SITES);

export const RAW_COLUMNS = [
  'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num',
];

// slope, ca and thal are dropped: they are missing for 35-99% of patients at
// Budapest, Switzerland and Long Beach VA. Missing values in the kept columns
// become 0 after standardization (the federated mean), and the columns with
// site-wide gaps get an explicit "was missing" indicator so the model can
// tell "average" apart from "not measured".
export const FEATURES = [
  'age', 'sex',
  'cp_typical', 'cp_atypical', 'cp_nonanginal', 'cp_asymptomatic',
  'trestbps', 'chol', 'chol_missing', 'fbs', 'fbs_missing',
  'restecg_normal', 'restecg_st', 'restecg_lvh',
  'thalach', 'thalach_missing', 'exang', 'oldpeak',
];

// Plausible clinical range of each feature, from general medical knowledge
// rather than from any hospital's data. Under differential privacy, features
// are standardized with these instead of the federated statistics, since exact
// statistics are not private and noisy ones are too noisy at this size.
export const FEATURE_BOUNDS = [
  [20, 90], [0, 1],                       // age, sex
  [0, 1], [0, 1], [0, 1], [0, 1],         // cp one-hot
  [80, 220], [100, 600], [0, 1],          // trestbps, chol, chol_missing
  [0, 1], [0, 1],                         // fbs, fbs_missing
  [0, 1], [0, 1], [0, 1],                 // restecg one-hot
  [60, 220], [0, 1], [0, 1], [-3, 7],     // thalach, thalach_missing, exang, oldpeak
];

/**
 * A standardization that needs no data: center each feature on the middle of
 * its range, scaled so the range spans SPREAD standard deviations.
 */
export function publicStandardization(bounds = FEATURE_BOUNDS, { spread = 4 } = {}) {
  return {
    mean: bounds.map(([lo, hi]) => (lo + hi) / 2),
    std: bounds.map(([lo, hi]) => (hi - lo) / spread),
  };
}

/** Parse one site's comma-separated file into records. '?' becomes null. */
export function parseSiteFile(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const values = line.split(',');
      if (values.length !== RAW_COLUMNS.length) {
        throw new Error(`Expected ${RAW_COLUMNS.length} columns, got ${values.length}: ${line}`);
      }
      const record = {};
      RAW_COLUMNS.forEach((col, i) => {
        const v = values[i].trim();
        record[col] = v === '?' || v === '' ? null : Number(v);
      });
      return cleanRecord(record);
    });
}

// Where each of the 14 columns sits (1-based) in UCI's original 76-attribute
// records (cleveland.data, hungarian.data, ...). See heart-disease.names.
const ORIGINAL_POSITIONS = [3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51, 58];

/**
 * Parse one of UCI's original 76-attribute files. Each patient spans several
 * whitespace-separated lines and ends with the token "name"; -9 means missing.
 * Records that are not exactly 76 values long (cleveland.data has a corrupted
 * stretch) are skipped and counted. Returns { records, skipped }.
 */
export function parseOriginalSiteFile(text) {
  const records = [];
  let skipped = 0;
  let current = [];
  for (const token of text.split(/\s+/).filter(Boolean)) {
    current.push(token);
    if (token !== 'name') continue;
    const values = current.map(Number);
    current = [];
    if (values.length !== 76 || ORIGINAL_POSITIONS.some((p) => !Number.isFinite(values[p - 1]))) {
      skipped++;
      continue;
    }
    const record = {};
    RAW_COLUMNS.forEach((col, i) => {
      const v = values[ORIGINAL_POSITIONS[i] - 1];
      record[col] = v === -9 ? null : v;
    });
    records.push(cleanRecord(record));
  }
  if (current.length) skipped++;
  return { records, skipped };
}

/** Parse either format: 14 comma-separated columns, or UCI's original 76-attribute records. */
export function parseAnySiteFile(text) {
  const first = text.split(/\r?\n/).find((line) => line.trim()) ?? '';
  if (first.split(',').length === RAW_COLUMNS.length) return { records: parseSiteFile(text), skipped: 0, format: 'processed (14 columns)' };
  const { records, skipped } = parseOriginalSiteFile(text);
  if (records.length === 0) throw new Error('No patients found. Expected 14 comma-separated columns or UCI\'s original 76-attribute format');
  return { records, skipped, format: 'original UCI (76 attributes)' };
}

// A cholesterol or resting blood pressure of 0 is a placeholder for
// "not measured" (all of Switzerland's cholesterol values are 0).
function cleanRecord(record) {
  if (record.chol === 0) record.chol = null;
  if (record.trestbps === 0) record.trestbps = null;
  return record;
}

const oneHot = (value, options) => options.map((o) => (value == null ? 0 : value === o ? 1 : 0));

/** Turn a record into a feature row (nulls for missing values) and a 0/1 label. */
export function featurize(record) {
  const r = record;
  const x = [
    r.age, r.sex,
    ...oneHot(r.cp, [1, 2, 3, 4]),
    r.trestbps, r.chol, r.chol == null ? 1 : 0, r.fbs, r.fbs == null ? 1 : 0,
    ...oneHot(r.restecg, [0, 1, 2]),
    r.thalach, r.thalach == null ? 1 : 0, r.exang, r.oldpeak,
  ];
  // num is 0 (no disease) to 4; any narrowing counts as disease present.
  return { x, y: r.num > 0 ? 1 : 0 };
}

export function featurizeAll(records) {
  const X = [];
  const y = [];
  for (const r of records) {
    const f = featurize(r);
    X.push(f.x);
    y.push(f.y);
  }
  return { X, y };
}

/** Small seeded PRNG so splits and shuffles are reproducible. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(array, rand) {
  const a = array.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Stratified train/test split, so each split keeps the site's disease rate. */
export function trainTestSplit({ X, y }, { testFraction = 0.3, seed = 42 } = {}) {
  const rand = mulberry32(seed);
  const train = { X: [], y: [] };
  const test = { X: [], y: [] };
  for (const label of [0, 1]) {
    const idx = shuffle(y.map((v, i) => (v === label ? i : -1)).filter((i) => i >= 0), rand);
    const nTest = Math.round(idx.length * testFraction);
    idx.forEach((i, k) => {
      const target = k < nTest ? test : train;
      target.X.push(X[i]);
      target.y.push(y[i]);
    });
  }
  return { train, test };
}

/**
 * Per-feature count / sum / sum of squares over the non-missing values.
 * This is what a hospital shares to agree on a common standardization,
 * instead of sharing rows. Self-contained so it can ship as a DCP work function.
 */
export function localStats(X) {
  const d = X[0].length;
  const count = new Array(d).fill(0);
  const sum = new Array(d).fill(0);
  const sumSq = new Array(d).fill(0);
  for (const row of X) {
    for (let j = 0; j < d; j++) {
      const v = row[j];
      if (v == null || Number.isNaN(v)) continue;
      count[j] += 1;
      sum[j] += v;
      sumSq[j] += v * v;
    }
  }
  return { count, sum, sumSq };
}

/** Combine several sites' localStats into a global mean and standard deviation. */
export function combineStats(statsList) {
  const d = statsList[0].count.length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(1);
  for (let j = 0; j < d; j++) {
    let n = 0;
    let s = 0;
    let ss = 0;
    for (const st of statsList) {
      n += st.count[j];
      s += st.sum[j];
      ss += st.sumSq[j];
    }
    if (n === 0) continue;
    mean[j] = s / n;
    const variance = Math.max(ss / n - mean[j] * mean[j], 0);
    std[j] = variance > 1e-12 ? Math.sqrt(variance) : 1;
  }
  return { mean, std };
}

/** Standardize rows with the given stats; missing values become 0 (the mean). */
export function standardize(X, { mean, std }) {
  return X.map((row) => row.map((v, j) => (v == null || Number.isNaN(v) ? 0 : (v - mean[j]) / std[j])));
}
