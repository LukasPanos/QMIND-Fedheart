// One dummy-DCP worker: a separate process that stands in for a DCP worker
// inside one hospital. It runs with its hospital's folder as its working
// directory, reads the dataset there, and answers the coordinator's tasks over
// IPC (see src/hospitalWorker.js). Patient rows never leave this process; only
// aggregates, weights (masked under secure aggregation), holdout metrics and
// personalization scores are sent back. It also keeps a log in the folder.
//   (started by scripts/dummy-dcp.js)  node dummy-dcp-worker.js <site> <folder>
import { readdirSync, readFileSync, appendFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { loadDatasetText } from '../src/hospital.js';
import { createHospitalWorker } from '../src/hospitalWorker.js';

const [site, folder] = process.argv.slice(2);
const seed = Number(process.env.FEDHEART_SEED ?? 42);
const LOG_FILE = 'fedheart-worker.log';

function log(text, level = 'info') {
  try {
    appendFileSync(join(folder, LOG_FILE), `${new Date().toISOString()}  ${text}\n`);
  } catch {
    // A read-only folder shouldn't stop training.
  }
  process.send({ type: 'log', text, level });
}

/** The dataset is the first .data / .csv / .txt file in the folder. */
function findDataFile() {
  const files = readdirSync(folder)
    .filter((f) => !f.startsWith('.') && f !== LOG_FILE && /\.(data|csv|txt)$/i.test(f))
    .filter((f) => statSync(join(folder, f)).isFile())
    .sort();
  if (!files.length) throw new Error(`no .data, .csv or .txt file in ${folder}`);
  return files[0];
}

let worker;
try {
  const file = findDataFile();
  const data = loadDatasetText(readFileSync(join(folder, file), 'utf8'), { seed });
  worker = createHospitalWorker(site, data);
  log(`Worker started in ${folder}. Loaded ${file}: ${data.rows} patients (${data.format})` +
    (data.skipped ? `, skipped ${data.skipped} unreadable records` : '') +
    `. ${data.train.y.length} for training, ${data.test.y.length} held out.`);
  process.send({
    type: 'ready',
    summary: {
      file, folder, rows: data.rows, skipped: data.skipped, format: data.format,
      train: data.train.y.length, holdout: data.test.y.length, diseaseRate: data.diseaseRate,
      // For secure aggregation: the other hospitals derive their shared secrets with us from this.
      publicKey: worker.publicKey,
    },
  });
} catch (err) {
  log(`Couldn't load a dataset: ${err.message}`, 'error');
  process.send({ type: 'failed', error: err.message });
}

process.on('message', async (task) => {
  try {
    if (!worker) throw new Error('no dataset loaded');
    const { result, log: line, privacy } = await worker.handle(task);
    process.send({ type: 'result', id: task.id, result });
    if (line) log(line);
    if (privacy) process.send({ type: 'privacy', privacy });
  } catch (err) {
    process.send({ type: 'result', id: task.id, error: err.message });
    log(`Declined a ${task.type} task: ${err.message}`, 'error');
  }
});

process.on('disconnect', () => process.exit(0));
