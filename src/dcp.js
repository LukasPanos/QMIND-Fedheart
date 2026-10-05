// Running FedHeart's local training on Distributive's Compute Platform (DCP).
//
// Mapping:
//   one hospital        -> one DCP compute group, whose workers run inside that hospital
//   one round at a site -> a job with one slice, deployed only to that compute group
//   the work function   -> the slice's input is the URL of the site's rows (DCP's
//                          remote-data pattern: the worker fetches it, so the rows
//                          go from the hospital's server to the hospital's worker);
//                          the global weights travel as job arguments
//
// The coordinator stays unchanged: createDcpParticipant returns an object with the
// same stats() / train() shape as createLocalParticipant, so federatedTrain cannot
// tell whether a hospital is local or on DCP, and a slice that never comes back is
// handled like any other dropout (roundTimeoutMs).
//
// This module needs `npm install dcp-client` and a DCP keystore. It is starter
// code for the stretch goal; adjust it to the PMs' DCP starter code at camp.

import { localTrain } from './localTrain.js';
import { localStats } from './data.js';

/**
 * Source of the DCP work function. Built as a string so localTrain and
 * localStats (both self-contained) are inlined and nothing else is captured.
 * Called by the worker as fn(data, task, payload):
 *   data     JSON { X, y } with raw feature rows, fetched by the worker from the slice URL
 *   task     'stats' or 'train'
 *   payload  { weights, hyperparameters, standardization } for 'train'
 */
export function buildWorkFunctionSource() {
  return `function fedheartWorkFunction(data, task, payload) {
  progress(0);
  const localTrain = ${localTrain.toString()};
  const localStats = ${localStats.toString()};
  if (typeof data === 'string') data = JSON.parse(data);
  if (task === 'stats') {
    const stats = localStats(data.X);
    progress(1);
    return stats;
  }
  const s = payload.standardization;
  const X = data.X.map((row) => row.map((v, j) => (v == null || Number.isNaN(v) ? 0 : (v - s.mean[j]) / s.std[j])));
  const result = localTrain({ weights: payload.weights, X: X, y: data.y, hyperparameters: payload.hyperparameters });
  progress(1);
  return result;
}`;
}

let dcpReady = null;
/** Initialise dcp-client once and return the compute API. */
export async function loadDcp({ scheduler } = {}) {
  if (!dcpReady) {
    dcpReady = (async () => {
      let dcpClient;
      try {
        dcpClient = await import('dcp-client');
      } catch {
        throw new Error('dcp-client is not installed. Run `npm install dcp-client` to use the DCP executor.');
      }
      const init = dcpClient.init ?? dcpClient.default?.init;
      const { compute } = await (scheduler ? init(scheduler) : init());
      return compute;
    })();
  }
  return dcpReady;
}

/**
 * A participant whose work runs on DCP workers in the hospital's compute group.
 *
 * @param {string} name
 * @param {object} options
 * @param {string} options.dataUrl            where that hospital's workers read its rows
 *                                           (workers must allow this origin: dcp-worker --allowedOrigins)
 * @param {object} [options.computeGroup]     { joinKey, joinSecret } for the hospital's group
 * @param {*}      [options.slicePaymentOffer] passed to job.exec (default compute.marketValue)
 * @param {object} [options.compute]          inject a compute API (tests, or a custom init)
 * @param {Function} [options.onJob]          called with each job before exec, to tweak it
 * @param {boolean|number} [options.localExec] run slices with job.localExec() on a worker
 *                                           inside this process (needs `npm install dcp-worker`);
 *                                           a number sets how many cores it uses
 */
export function createDcpParticipant(name, { dataUrl, computeGroup, slicePaymentOffer, compute, onJob, scheduler, localExec } = {}) {
  if (!dataUrl) throw new Error(`createDcpParticipant(${name}) needs a dataUrl`);
  const workFunction = buildWorkFunctionSource();

  async function runSlice(task, payload, label) {
    const api = compute ?? (await loadDcp({ scheduler }));
    const job = api.for([new URL(dataUrl)], workFunction, [task, payload]);
    job.public = { ...(job.public || {}), name: `FedHeart ${label} (${name})` };
    if (computeGroup) job.computeGroups = [computeGroup];
    if (onJob) await onJob(job);
    const results = localExec
      ? await job.localExec(typeof localExec === 'number' ? localExec : 1)
      : await job.exec(slicePaymentOffer ?? api.marketValue);
    const values = typeof results.values === 'function' ? Array.from(results.values()) : Array.from(results);
    if (!values.length) throw new Error(`${name}: DCP job returned no results`);
    return values[0];
  }

  return {
    name,
    stats: () => runSlice('stats', null, 'stats'),
    train: (weights, hyperparameters, { round, standardization }) =>
      runSlice('train', { weights, hyperparameters, standardization }, `round ${round}`),
  };
}
