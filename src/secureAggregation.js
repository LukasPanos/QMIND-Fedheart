// Secure aggregation: the coordinator learns only the sum of the hospitals'
// updates, never one hospital's update.
//
// Each pair of hospitals shares a secret, agreed out of band (the coordinator
// never sees it). For every message, both expand that secret into the same
// pseudorandom mask vector; the hospital whose name sorts first adds it and
// the other subtracts it, so in the sum over a cohort every mask cancels.
// Values are fixed-point integers mod 2^64 so the cancellation is exact.
//
// Masks must never be reused for a different update: two values masked the
// same way differ by exactly their difference. Every mask label includes a
// session ID that the coordinator draws fresh for each training run, plus the
// round, attempt and cohort, and a hospital refuses to mask different values
// under a label it has used before.
//
// Dropouts: masks only cancel if everyone in the cohort answers. When a
// hospital drops, the coordinator asks the rest again with the smaller cohort,
// and they mask afresh (the cohort and attempt are part of the mask label).
// Hospitals refuse cohorts smaller than minCohort (default 3): with two, each
// could subtract its own update from the average and get the other's.
//
// Threat model: an honest-but-curious coordinator that follows the protocol.
// A coordinator that falsely reports a dropout gets two sums over cohorts that
// differ by one hospital, and so that hospital's update; defending against
// that needs double masking with secret sharing (Bonawitz et al., 2017). The
// sum itself still reveals whatever averaged weights reveal; differential
// privacy is the tool for that.
//
// withSecureAggregation must run on the hospital's side. Wrapping a
// participant whose plaintext update first travels to the coordinator (such
// as a DCP participant today) protects nothing.

import { createHash, createHmac, createPublicKey, diffieHellman, generateKeyPairSync, randomBytes } from 'node:crypto';

// 2^24 gives about 6e-8 resolution. A single entry must stay below
// 2^53 / 2^24 (about 5e8) to encode exactly; the largest FedHeart value is a
// site's sum of squared cholesterol, about 1e7.
export const FIXED_POINT_SCALE = 2 ** 24;
const MAX_ABS = 2 ** 53 / FIXED_POINT_SCALE;

export function encodeFixedPoint(x) {
  if (!Number.isFinite(x) || Math.abs(x) >= MAX_ABS) throw new RangeError(`cannot encode ${x} as fixed point`);
  return BigInt.asUintN(64, BigInt(Math.round(x * FIXED_POINT_SCALE)));
}

export function decodeFixedPoint(v) {
  return Number(BigInt.asIntN(64, v)) / FIXED_POINT_SCALE;
}

/** Expand a pair's secret into `length` pseudorandom 64-bit values (HMAC-SHA256 in counter mode). */
function expandMask(secret, label, length) {
  const out = new Array(length);
  for (let block = 0; block * 4 < length; block++) {
    const bytes = createHmac('sha256', secret).update(`${label}|${block}`).digest();
    for (let k = 0; k < 4 && block * 4 + k < length; k++) out[block * 4 + k] = bytes.readBigUInt64BE(8 * k);
  }
  return out;
}

/**
 * Mask one hospital's values for a cohort. Returns 16-digit hex strings, which
 * look uniformly random on their own and sum (mod 2^64) to the cohort total.
 */
export function maskValues(values, { self, cohort, secrets, label }) {
  const acc = values.map(encodeFixedPoint);
  for (const peer of cohort) {
    if (peer === self) continue;
    const secret = secrets[peer];
    if (!secret) throw new Error(`${self} shares no secret with ${peer}`);
    const mask = expandMask(secret, label, acc.length);
    const add = self < peer;
    for (let j = 0; j < acc.length; j++) acc[j] = BigInt.asUintN(64, add ? acc[j] + mask[j] : acc[j] - mask[j]);
  }
  return acc.map((v) => v.toString(16).padStart(16, '0'));
}

export function isMaskedPayload(masked, length) {
  return Array.isArray(masked) && masked.length === length && masked.every((h) => typeof h === 'string' && /^[0-9a-f]{16}$/.test(h));
}

/** Add masked payloads from a whole cohort; the masks cancel and the true sums come out. */
export function sumMasked(payloads) {
  const acc = payloads[0].map(() => 0n);
  for (const p of payloads) p.forEach((h, j) => { acc[j] = BigInt.asUintN(64, acc[j] + BigInt(`0x${h}`)); });
  return acc.map(decodeFixedPoint);
}

/**
 * One random secret per pair of hospitals: { [name]: { [peer]: hex } }.
 * Each hospital should only ever hold its own row. This is a setup helper for
 * simulations; real hospitals agree on each pair's secret out of band.
 */
export function createPairwiseSecrets(names) {
  const secrets = Object.fromEntries(names.map((n) => [n, {}]));
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const s = randomBytes(32).toString('hex');
      secrets[names[i]][names[j]] = s;
      secrets[names[j]][names[i]] = s;
    }
  }
  return secrets;
}

/**
 * X25519 key agreement, so hospitals can set up their pairwise secrets through
 * the coordinator without it learning them. Each hospital publishes its public
 * key; any two derive the same secret from their own private key and the
 * other's public key. The coordinator only ever relays public keys.
 *
 * Relaying keys this way trusts the coordinator to pass them on unchanged
 * (honest but curious, as for the rest of the protocol). A coordinator that
 * swapped in its own keys could read the masks; real deployments authenticate
 * the public keys, for example with each hospital's certificate.
 */
export function createKeyAgreement() {
  const { publicKey, privateKey } = generateKeyPairSync('x25519');
  return {
    publicKey: publicKey.export({ type: 'spki', format: 'der' }).toString('base64'),
    /** The secret this hospital shares with the owner of `peerPublicKey` (hex). */
    secretWith(peerPublicKey) {
      const peer = createPublicKey({ key: Buffer.from(peerPublicKey, 'base64'), format: 'der', type: 'spki' });
      return createHash('sha256').update(diffieHellman({ privateKey, publicKey: peer })).digest('hex');
    },
  };
}

/**
 * Wrap a participant (on the hospital's side) so it only ever returns masked
 * vectors. The coordinator passes { cohort, attempt } in each call's context.
 *
 *   stats(context)              -> { masked: [...count, ...sum, ...sumSq] }
 *   train(weights, hp, context) -> { masked: [...n * weights, n, n * loss, n] }
 *
 * where n is numSamples. The last two are 0 when the loss is not reported
 * (NaN, as under differential privacy).
 *
 * The coordinator can no longer check individual updates, so the hospital
 * checks its own and refuses to send one that is malformed.
 *
 * `secrets` is read at masking time, so it can be filled in later (as the
 * dummy-DCP workers do once they learn the other hospitals' public keys).
 */
export function withSecureAggregation(participant, { secrets, minCohort = 3 }) {
  const self = participant.name;
  const used = new Map(); // mask label -> the values masked under it
  const mask = (task, values, context = {}) => {
    const { cohort, attempt = 1, round = 0, session = '' } = context;
    if (!Array.isArray(cohort) || !cohort.includes(self)) throw new Error(`${self}: not a member of this cohort`);
    const members = [...new Set(cohort)].sort();
    if (members.length < minCohort) throw new Error(`${self}: refusing a cohort of ${members.length} (minimum ${minCohort})`);
    const label = `${task}|${session}|${round}|${attempt}|${members.join(',')}`;
    const key = values.join(',');
    if (used.has(label) && used.get(label) !== key) throw new Error(`${self}: refusing to reuse masks for a different update`);
    used.set(label, key);
    return { masked: maskValues(values, { self, cohort: members, secrets, label }) };
  };
  return {
    ...participant,
    async stats(context) {
      const s = await participant.stats(context);
      const d = s?.count?.length;
      const ok = d > 0 && ['count', 'sum', 'sumSq'].every((k) => Array.isArray(s[k]) && s[k].length === d && s[k].every(Number.isFinite));
      if (!ok) throw new Error(`${self}: refusing to send invalid feature statistics`);
      return mask('stats', [...s.count, ...s.sum, ...s.sumSq], context);
    },
    async train(weights, hyperparameters, context) {
      const u = await participant.train(weights, hyperparameters, context);
      const ok =
        u && Array.isArray(u.weights) && u.weights.length === weights.length && u.weights.every(Number.isFinite) &&
        Number.isFinite(u.numSamples) && u.numSamples > 0 && (Number.isFinite(u.loss) || Number.isNaN(u.loss));
      if (!ok) throw new Error(`${self}: refusing to send an invalid update`);
      const n = u.numSamples;
      const lossN = Number.isFinite(u.loss) ? n : 0;
      return mask('train', [...u.weights.map((w) => w * n), n, lossN ? u.loss * n : 0, lossN], context);
    },
  };
}
