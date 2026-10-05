// The hub: hospitals that connect from somewhere else (dummy DCP's worker
// processes) turned into participants the coordinator can drive.
//
// Each hospital gets a `send(task)` channel when it connects (IPC to its worker
// process in scripts/dummy-dcp.js) and answers through `resolve(taskId, result)`.
// A task is one of (see src/hospital.js):
//
//   { type: 'configure', privacy }                            -> privacy status
//   { type: 'stats' }                                         -> { count, sum, sumSq }
//   { type: 'train', weights, hyperparameters, round, standardization } -> { weights, numSamples, loss }
//   { type: 'evaluate', weights, standardization }            -> { n, auc, accuracy, logLoss }
//   { type: 'personalize', weights, standardization }         -> { strategy, label, shared, personalized }
//
// Under secure aggregation, stats and train tasks also carry the round's
// cohort, the attempt number, the run's session ID and the cohort members' public keys (from each
// hospital's ready summary), and the answer is { masked } instead.
//
// participant(site) has the same stats() / train() shape as
// createLocalParticipant and createDcpParticipant, so federatedTrain cannot
// tell a worker process from any other hospital. A hospital that is disconnected,
// offline or has no dataset loaded rejects at once, and the coordinator drops
// it for that round like any other dropout.

export function createHub({ onEvent = () => {} } = {}) {
  const sites = new Map();
  const pending = new Map();
  let nextId = 1;

  const get = (site) => {
    if (!sites.has(site)) sites.set(site, { send: null, online: true, ready: false, summary: null });
    return sites.get(site);
  };
  const bytes = (value) => JSON.stringify(value ?? null).length;

  function failPending(site, reason) {
    for (const [id, p] of pending) {
      if (p.site !== site) continue;
      pending.delete(id);
      p.reject(new Error(reason));
    }
  }

  function unavailable(site) {
    const s = sites.get(site);
    if (!s?.send) return `${site} is not connected`;
    if (!s.online) return `${site} is offline`;
    if (!s.ready) return `${site} has no dataset loaded`;
    return null;
  }

  function run(site, task) {
    const reason = unavailable(site);
    if (reason) return Promise.reject(new Error(reason));
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, { site, type: task.type, resolve, reject });
      onEvent({ type: 'task', site, task: task.type, round: task.round, bytes: bytes(task) });
      try {
        get(site).send({ id, ...task });
      } catch (err) {
        pending.delete(id);
        reject(err);
      }
    });
  }

  return {
    /** A hospital opened its channel. A reconnect replaces the old one. */
    connect(site, send) {
      const s = get(site);
      if (s.send) failPending(site, `${site} reconnected`);
      Object.assign(s, { send, online: true, ready: false, summary: null });
      onEvent({ type: 'connect', site });
    },

    /** Its channel closed. `send` guards against a stale close after a reconnect. */
    disconnect(site, send) {
      const s = sites.get(site);
      if (!s || (send && s.send !== send)) return;
      Object.assign(s, { send: null, ready: false, summary: null });
      failPending(site, `${site} disconnected`);
      onEvent({ type: 'disconnect', site });
    },

    /** The hospital loaded a dataset. `summary` holds counts only, never rows. */
    ready(site, summary) {
      const s = get(site);
      if (!s.send) return;
      Object.assign(s, { ready: true, summary });
      onEvent({ type: 'ready', site, summary });
    },

    /** The hospital's "Go offline" switch: refuses tasks until it comes back. */
    setOnline(site, online) {
      const s = get(site);
      s.online = Boolean(online);
      if (!s.online) failPending(site, `${site} is offline`);
      onEvent({ type: online ? 'online' : 'offline', site });
    },

    /** An answer to task `id`. `error` (a string) rejects it instead. */
    resolve(id, result, error) {
      const p = pending.get(Number(id));
      if (!p) return false;
      pending.delete(Number(id));
      onEvent({ type: 'result', site: p.site, task: p.type, bytes: bytes(result), error });
      if (error) p.reject(new Error(`${p.site}: ${error}`));
      else p.resolve(result);
      return true;
    },

    participant(site) {
      // The coordinator adds { cohort, attempt } to the context under secure aggregation.
      const secureContext = ({ cohort, attempt, session } = {}) =>
        cohort
          ? { cohort, attempt, session, publicKeys: Object.fromEntries(cohort.map((name) => [name, sites.get(name)?.summary?.publicKey])) }
          : {};
      return {
        name: site,
        stats: (context) => run(site, { type: 'stats', ...secureContext(context) }),
        train: (weights, hyperparameters, { round, standardization, ...context }) =>
          run(site, { type: 'train', weights, hyperparameters, round, standardization, ...secureContext(context) }),
      };
    },

    /** Send a hospital its own settings for the next run (in practice its admins would set these). */
    configure(site, { privacy = null } = {}) {
      return run(site, { type: 'configure', privacy });
    },

    /** Ask a hospital to score a model on its own held-out patients. */
    evaluate(site, { weights, standardization }) {
      return run(site, { type: 'evaluate', weights, standardization });
    },

    /** Ask a hospital to personalize the shared model on site; only the scores come back. */
    personalize(site, { weights, standardization }, hyperparameters) {
      return run(site, { type: 'personalize', weights, standardization, hyperparameters });
    },

    /** Hospitals that could take a task right now. */
    available() {
      return [...sites.keys()].filter((site) => !unavailable(site));
    },

    status() {
      return [...sites.entries()].map(([site, s]) => ({
        site,
        connected: Boolean(s.send),
        online: s.online,
        ready: s.ready,
        summary: s.summary,
      }));
    },
  };
}
