# FedHeart

Federated logistic regression across hospitals that never share patient data. Built at Camp QMIND with Distributive, as groundwork for a tool that predicts how psoriatic arthritis patients respond to biologic medications.

FedHeart trains one heart disease model on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The data was collected at four hospitals: Cleveland, Budapest, Switzerland and Long Beach VA. Each hospital trains on its own file, sends back only model weights, and a coordinator averages those weights into a shared model (FedAvg).

The package does four things:

- **Takes hyperparameters and returns trained weights:** `federatedTrain(hyperparameters, { participants })`.
- **Never moves training data:** a hospital only ever sends weights, a sample count, a loss, and per-feature count, sum and sum of squares.
- **Keeps working when a participant drops out:** a hospital that errors, hangs past the round timeout, or returns a malformed update is skipped for that round.
- **Runs locally or on DCP:** the same coordinator drives both.
- **Can hide each hospital's update (optional):** with secure aggregation the coordinator only learns the sum across hospitals.
- **Can protect each patient (optional):** with differential privacy, each hospital adds calibrated noise during training and tracks its own privacy budget (ε).
- **Fits each hospital:** after training, each hospital adapts the shared model to its own patients, on site, and keeps whichever version is best for them.

## Quick start

Requires Node.js 20 or newer. There are no runtime dependencies.

```bash
npm test                 # 54 tests
npm run baselines        # Minimum milestone: each hospital alone vs. all data pooled
npm run federated -- --drop va --drop-round 16   # Target milestone: federated training with a dropout
npm run demo             # Sunday demo: everything above, plus results/chart.html
npm run demo -- --delay 150   # Slower round-by-round output for presenting live
npm run demo -- --secure      # The same, with secure aggregation
npm run federated -- --show-view            # What the coordinator receives: plain weights...
npm run federated -- --show-view --secure   # ...versus masked values that only add up to the sum
npm run demo -- --dp --secure    # Differential privacy and secure aggregation together
npm run privacy                  # Privacy vs accuracy: AUC at several noise levels
```

Every script accepts `--rounds`, `--local-epochs`, `--learning-rate`, `--batch-size`, `--l2`, `--mu` and `--seed`.

## Results (seed 42, 30 rounds × 2 local epochs)

All models are scored on the same 276 held-out test patients, a stratified 30% of each hospital's data.

| Model | ROC AUC | Accuracy |
|---|---|---|
| Cleveland alone | 0.836 | 76.4% |
| Budapest alone | 0.864 | 75.7% |
| Switzerland alone | 0.761 | 58.3% |
| Long Beach VA alone | 0.839 | 75.7% |
| **Federated, all 4 hospitals** | **0.886** | **81.2%** |
| **Federated, VA drops out at round 16** | **0.885** | **81.5%** |
| Pooled data (upper bound) | 0.884 | 81.2% |

Federated training beats every hospital training alone, matches the pooled upper bound, and is barely affected when Long Beach VA leaves halfway. The same holds on seeds 1 and 7, and a test checks it. With only 276 test patients, differences of about 0.005 AUC between federated and pooled are noise.

These results are worth having ready for questions:

- **Hospitals are very different.** Disease rates are 46% at Cleveland, 36% at Budapest, 93% at Switzerland and 74% at Long Beach VA. Switzerland never recorded cholesterol, and VA is missing exercise-test results for about a quarter of its patients. This is why one hospital's model transfers badly to the others: Cleveland's model scores 0.51 AUC on Switzerland's patients.
- **Pooling isn't best for every site.** The model trained on Budapest alone scores better on Switzerland's patients than the pooled model does. Federated and pooled models optimize for patients overall, not for each site. Personalization (below) fixes this.
- **AUC levels off after the first round.** It measures how well the model ranks patients, and one round of averaging already gets the ranking mostly right. Later rounds mainly calibrate the probabilities, which shows up in accuracy and log loss.

## How it works

```
                 hyperparameters + global weights
   coordinator  ─────────────────────────────────▶  hospital (x4)
   (FedAvg)     ◀─────────────────────────────────  trains on its own rows
                 new weights + sample count + loss
```

1. **Agree on standardization (round 0).** Each hospital sends per-feature count, sum and sum of squares. The coordinator combines them into a global mean and standard deviation, identical to what pooling would give (a test checks this), without seeing any rows.
2. **Rounds.** The coordinator sends the current weights. Each hospital standardizes its data, runs `localEpochs` of mini-batch SGD, and returns new weights. The coordinator averages them, weighted by sample count.
3. **Dropout.** Responses are gathered with `Promise.allSettled` plus a per-round timeout. Any hospital that throws, times out, or returns weights of the wrong shape or with NaN values is left out of that round's average. If fewer than `minParticipants` respond, the round is skipped and the weights stay unchanged. A hospital can rejoin in a later round.
4. **Secure aggregation (optional).** See below.
5. **Differential privacy (optional).** See below.
6. **Different hospitals (optional).** Setting `mu > 0` adds a FedProx term that keeps each hospital's update close to the global model.
7. **Personalization.** After the last round, each hospital adapts the shared model to its own patients. See below.

### Personalization (`src/personalization.js`)

The four hospitals are very different: disease rates range from 36% to 93%, and Switzerland never recorded cholesterol. So the shared model isn't the best model for every hospital's patients. After federated training, each hospital tries a ladder of candidates, from fully shared to fully local:

- the shared model as is
- fine-tuned on its own data for 3 or 10 epochs
- 75/25, 50/50 and 25/75 mixes of the shared model and a model trained only on its own data
- its own model only

It scores each candidate with 5-fold cross-validation on its own training rows and keeps the best. Everything happens on site: the coordinator never sees the personalized model or which candidate won, and under differential privacy it spends no privacy budget.

Candidates are scored on log loss, which measures how trustworthy the predicted risk percentages are. AUC can't be used here: Switzerland's test split has only 2 patients without disease, so per-site AUC there is meaningless.

```js
const { model, strategy, label } = participant.personalize(sharedModel); // e.g. 'mix25', '25% shared, 75% local'
```

**Results**, each hospital on its own test patients, averaged over 5 data splits (log loss, lower is better / accuracy):

| Hospital | Alone | Shared model | Personalized | Choices over the 5 splits |
|---|---|---|---|---|
| Cleveland | 0.438 / 78.7% | 0.412 / 80.0% | 0.412 / 80.0% | shared model every time |
| Budapest | 0.382 / 84.5% | 0.397 / 83.4% | 0.391 / 83.6% | 75% shared twice, shared model 3 times |
| Switzerland | 0.216 / 93.0% | 0.288 / 91.4% | **0.182 / 94.1%** | 25% shared 3 times, fine-tuned twice |
| Long Beach VA | 0.578 / 73.3% | 0.522 / 77.0% | 0.524 / 77.3% | 75% shared once, shared model 4 times |
| **Mean** | 0.403 / 82.4% | 0.405 / 82.9% | **0.377 / 83.8%** | |

- **The most different hospital gains the most.** Switzerland's personalized model beats both the shared model and Switzerland alone.
- **Hospitals that fit the shared model keep it.** Cleveland always keeps it. Long Beach VA almost always does and comes out even.
- **Personalized is best on average.** It beats both training alone and the shared model.

**Caveats:**
- **Cross-validation slightly favors the shared model,** because the shared model was trained on the same rows. The choice therefore leans toward the shared model, which is the safe direction. It's also why Budapest stays a little worse than training alone.
- **Choices vary with the data split.** With 86–212 training patients per hospital, the winning candidate changes from split to split (see the last column), so quote averages, not one run.
- **Not covered by DP.** Under differential privacy, the personalized model is fit on the hospital's data without noise. That's fine as long as it stays at the hospital; the DP guarantee covers what the hospital sent, not this model.

### Secure aggregation (`src/secureAggregation.js`)

Without it, the coordinator sees each hospital's weights and feature statistics. With it, the coordinator only learns their sum, which is all FedAvg needs.

- **Pairwise masks.** Each pair of hospitals shares a secret that the coordinator never sees. For every message, both hospitals expand it (HMAC-SHA256) into the same random vector. One adds it and the other subtracts it, so the masks cancel when the coordinator adds up the cohort. On its own, a masked vector looks like random noise.
- **Exact arithmetic.** Values are encoded as fixed-point integers (scale 2^24) modulo 2^64, so the masks cancel exactly. Hospitals send `n × weights`, `n` and `n × loss`, and the coordinator divides the sums to get the same weighted average. Secure training matches plain training to within 1e-5 (tested).
- **Round 0 too.** The feature `count`, `sum` and `sumSq` are masked the same way, so per-site means and variances stay hidden.
- **Dropouts.** Masks only cancel when every member of the cohort answers. If a hospital drops out, the coordinator asks the rest again with the smaller cohort, and they mask with fresh values. Hospitals refuse cohorts smaller than 3, because with two of them each could subtract its own update from the average and get the other's. When fewer than 3 are left, the round is skipped.
- **Validation moves to the hospital.** The coordinator can no longer spot a NaN or malformed update, so each hospital checks its own update and refuses to send a bad one. That counts as a dropout.

```js
import { withSecureAggregation, createPairwiseSecrets } from 'fedheart';
const secrets = createPairwiseSecrets(participants.map((p) => p.name)); // simulation only
const secured = participants.map((p) => withSecureAggregation(p, { secrets: secrets[p.name] }));
await federatedTrain(hp, { participants: secured, secureAggregation: true });
```

**What it does not protect against:**
- **A coordinator that breaks the protocol.** It assumes the coordinator follows the protocol (honest but curious). A coordinator that falsely claims a hospital dropped out gets two sums whose difference is that hospital's update. The full protocol (Bonawitz et al., 2017) prevents this with double masking and secret sharing.
- **Leaks from the sum itself.** The averaged model is still shared each round. Differential privacy is the tool for that.
- **DCP, for now.** `withSecureAggregation` must run at the hospital. On DCP the worker's result goes straight to the coordinator, so the masking would have to happen inside the work function. The secrets can't travel as job arguments, so the hospital's data endpoint would need to serve its masks alongside its rows.
- **Shared secrets.** `createPairwiseSecrets` generates every pair's secret in one place, which only makes sense in a simulation. Real hospitals would agree on them out of band, or through an authenticated key exchange.

### Differential privacy (`src/privacy.js`)

Secure aggregation hides each hospital's update, but the averaged model is still shared, and a model can leak facts about the patients it was trained on. Differential privacy (DP) puts a mathematical limit on that: with it, the model looks almost the same whether or not any one patient's record was used. ε measures how close: lower is more private, and single digits are generally considered meaningful.

- **DP-SGD at each hospital.** Each training step samples each patient with probability `batchSize / n`. Each patient's gradient is clipped to `clipNorm`, so no single patient can move the model by more than that, and Gaussian noise of standard deviation `noiseMultiplier × clipNorm` is added. The noise comes from `crypto.getRandomValues`, not from the seed, because the coordinator picks the seed and could otherwise regenerate and subtract the noise.
- **Each hospital protects its own patients.** It trusts nobody, including the coordinator, the other hospitals and the network. It sets its own `noiseMultiplier` and `clipNorm`, ignoring what the coordinator sends. It tracks its ε with Rényi DP accounting, which matches the DP-SGD MNIST reference (tested). It refuses to train once another round would exceed its `epsilonBudget`, which the coordinator handles as a dropout. Re-sending the same update, as in a secure-aggregation re-run, costs no extra privacy.
- **No exact numbers leave the hospital.** Under DP a hospital reports no training loss, and it refuses to release exact feature statistics. Round 0 is replaced by `publicStandardization()`, which centers and scales each feature using plausible clinical ranges (`FEATURE_BOUNDS`), not anyone's data. Noisy statistics would be too noisy at this dataset size.
- **Fewer rounds.** Every round costs privacy, and AUC levels off within a few rounds. So `--dp` defaults to 10 rounds at learning rate 0.1, with `--noise 3` and `--clip 1`.

```js
const participant = createLocalParticipant('va', data, {
  privacy: { noiseMultiplier: 3, clipNorm: 1, delta: 1e-5, epsilonBudget: 5 },
});
await federatedTrain(hp, { participants, standardization: publicStandardization() });
participant.privacySpent(); // { epsilon: 2.41, delta: 1e-5, rounds: 10 }
```

**Privacy vs accuracy** (`npm run privacy`, 10 runs per row; ε shown for the worst-off hospital, Switzerland, the smallest):

| Setting | ε (δ = 1e-5) | Mean AUC (range) | Accuracy |
|---|---|---|---|
| No DP | none | 0.886 | 81.2% |
| Noise 1 | 15.9 | 0.859 (0.855–0.864) | 78.3% |
| Noise 2 | 5.1 | 0.858 (0.847–0.865) | 78.2% |
| **Noise 3 (default)** | **3.0** | **0.858 (0.845–0.869)** | **78.3%** |
| Noise 5 | 1.6 | 0.855 (0.837–0.872) | 78.6% |
| Noise 8 | 0.96 | 0.847 (0.811–0.873) | 77.0% |
| Budapest alone, no DP (best single hospital) | none | 0.864 | 75.7% |

At ε ≈ 3, federated training with DP roughly ties the best single hospital on AUC and beats every hospital alone on accuracy. DP costs about 0.03 AUC compared with federated training without it, and almost all of that comes from clipping and public standardization rather than the noise itself: noise 0.5 scores about the same as noise 3. DP results differ on every run by design, so quote the table, not a single run.

**What it does not cover:**
- **Patient counts are public.** Each hospital's patient count is released exactly and treated as public. The guarantee covers what is in the records.
- **DCP, for now.** The accounting and enforcement live in `createLocalParticipant`, at the hospital. On DCP, the work function does run DP-SGD, but the coordinator writes it and sets its arguments, so hospitals would need to pin the work function and enforce their own noise settings.
- **Floating-point noise.** The noise is ordinary floating-point Gaussian noise. Production DP libraries use discrete or specially rounded noise to close known floating-point attacks.

### Data cleaning (`src/data.js`)

- `?` becomes missing. A cholesterol or resting blood pressure of `0` also becomes missing, since all of Switzerland's cholesterol values are 0.
- `slope`, `ca` and `thal` are dropped because they are missing for 35–99% of patients at three of the four sites.
- `cp` (chest pain type) and `restecg` (resting ECG) are one-hot encoded.
- After standardization, missing values become 0, which is the global mean. `chol_missing`, `fbs_missing` and `thalach_missing` indicator columns tell the model which values were never measured.
- The label is `num > 0`: any narrowing of a major vessel counts as disease.

18 features plus a bias term.

## Using it as a package

```js
import { federatedTrain, createLocalParticipant, withDropout, loadAllSites } from 'fedheart';

const sites = loadAllSites();   // local simulation; each hospital would load only its own file
const participants = Object.entries(sites).map(([id, s]) => createLocalParticipant(id, s.train));

const { weights, standardization, history } = await federatedTrain(
  { rounds: 30, localEpochs: 2, learningRate: 0.05, batchSize: 16, l2: 0.001, mu: 0 },
  { participants, roundTimeoutMs: 60_000, minParticipants: 2 },
);
```

A participant is any object with this shape:

```js
{
  name: 'cleveland',
  stats: async () => ({ count, sum, sumSq }),
  train: async (weights, hyperparameters, { round, standardization }) => ({ weights, numSamples, loss }),
}
```

`createLocalParticipant` keeps the data in a closure. `createDcpParticipant` (in `fedheart/dcp`) runs the same steps on DCP workers. Anything else, such as an HTTP client for a hospital's server, works too if it has this shape.

## DCP (stretch goal)

`src/dcp.js` maps FedHeart onto DCP as follows:

| FedHeart | DCP |
|---|---|
| One hospital | One compute group, whose workers run inside that hospital |
| One hospital's work in one round | A one-slice job deployed to that compute group |
| Hospital data | Remote data. The slice input is a `URL` to the hospital's rows, which the worker fetches itself. |
| Global weights and hyperparameters | Job arguments |
| `localTrain` | Inlined into the work function. It is self-contained, which a test checks, so `.toString()` is enough to ship it. It calls `progress()` every epoch. |
| A hospital dropping out | A slice that doesn't return within `roundTimeoutMs` |

**Setup (one-time, on your own machine):**

1. Create a DCP account on the Portal and save your two keystores in `~/.dcp/`: `id.keystore` (identity) and `default.keystore` (the bank account that pays for jobs). `npx dcp-util mkad` can also generate them. See [Getting set up](https://distributive.network/intro/getting-setup.html).
2. Get credits. New accounts start at 0. Either run a worker from the Portal's Worker tab to earn some, or ask the Distributive PMs to fund the account.
3. `npm install dcp-client dcp-worker`

**Fastest path: `localExec`.** Slices go through the scheduler but run on a worker inside your own Node process. This is good for checking that the work function runs on a real DCP worker.

```bash
node scripts/serve-sites.js &                   # stand-in for each hospital's data endpoint
npm run dcp -- --local --rounds 1
```

**On the network.** Slices run on separate workers, and each worker must be allowed to fetch the data URL.

```bash
node scripts/serve-sites.js &
npx dcp-worker --allowedOrigins http://localhost:8787    # in another terminal
npm run dcp -- --rounds 1 --data-url "http://localhost:8787/{site}.json"
```

To pin a hospital's slices to its own workers, set `FEDHEART_<SITE>_JOIN_KEY` and `FEDHEART_<SITE>_JOIN_SECRET`, for example `FEDHEART_VA_JOIN_KEY`.

> **Status:** the DCP path is tested end to end against a mocked compute API, and its weights match local training exactly. It has not yet run against a real DCP scheduler, so expect to adapt it to the PMs' DCP starter code at camp. Two things to confirm on first run: that a `localExec` worker may fetch `localhost:8787` (if not, check the worker's allowed origins), and how many credits each one-slice job costs.

## Dummy DCP: hospitals as folders

A hands-on simulation of a DCP job without a DCP account. Each hospital is a folder on your machine, and a separate worker process runs inside each folder, the way a DCP worker would run inside a hospital. One page shows everything.

```bash
npm run dummy-dcp
npm run dummy-dcp -- --root ~/Desktop --va "~/Desktop/long beach" --rounds 30 --delay 300 --min-participants 2 --round-timeout 30000 --port 8790 --no-open
npm run dummy-dcp -- --secure --dp --noise 3 --epsilon 5   # start with secure aggregation and DP switched on
```

By default the hospitals are these folders on the Desktop: `cleveland`, `hungarian`, `switzerland` and `long beach`. Change the parent folder with `--root`, or point one hospital somewhere else with `--cleveland`, `--hungarian`, `--switzerland` or `--va`. Each folder needs one `.data`, `.csv` or `.txt` file in either format:
- the 14-column `processed.*.data` files in `data/raw/`
- UCI's original 76-attribute files (`cleveland.data`, `hungarian.data`, …), where each patient spans several lines and `-9` means missing. `cleveland.data` has a known corrupted stretch, so 282 of its patients are readable and the rest are skipped and counted.

How a run works:

1. **Workers start.** The coordinator starts one worker per folder (`scripts/dummy-dcp-worker.js`). The worker reads its file, featurizes it, splits it 70/30 and reports counts only. It writes a `fedheart-worker.log` in its folder.
2. **Choose privacy settings (optional).** The bar under the header switches **Secure aggregation** and **Differential privacy** on or off for the next run, and sets the noise level and an optional ε budget. Settings can change between runs, not during one.
3. **Press Start training.** Each round, the coordinator sends the global weights to every worker over IPC. The worker trains on its own rows and sends back only `{ weights, numSamples, loss }`, or masked numbers under secure aggregation. The coordinator averages them with the unchanged `federatedTrain`, and each worker then scores the new model on its held-out patients.
4. **Watch the page.** It shows:
   - the hospital cards, including each hospital's privacy spent (ε)
   - a live log
   - the results table and AUC by round
   - **What the coordinator received**: each hospital's weights in the clear, or random-looking masked numbers under secure aggregation
   - each finding's weight in the shared model next to each hospital's own update; the update dots are hidden under secure aggregation, because the coordinator never sees them

   Switch a hospital **Offline** to watch it drop out and rejoin.
5. **Personalize.** After the last round, each worker personalizes the shared model on its own data and keeps it. Only the scores come back, shown in **Each hospital, on its own patients**.
6. **Finish.** The final results stay on screen and the model is saved to `results/dummy-dcp-model.json`. **Train again** reruns with the same workers. **Clear & restart** stops any run, clears results and logs, and restarts the workers so they re-read their folders.

How the privacy features run here, all inside the hospital workers:

- **Secure aggregation with real key agreement.** Each worker generates an X25519 key pair when it starts and reports only the public key. Under secure aggregation, the coordinator relays the cohort's public keys with each task. Each pair of workers derives the same secret from them, which the coordinator never sees, and masks its update with it (`src/hospitalWorker.js`).
- **Masks are never reused.** Every run gets a fresh session ID, and a worker refuses to mask a different update under a label it has used before. Without this, two runs on the same long-lived workers would reuse masks, and subtracting them would reveal how a hospital's update changed.
- **Dropouts.** If a hospital goes offline under secure aggregation, the round is re-run with the rest (the log says "after re-running without …"). Secure aggregation needs at least 3 hospitals, and Start refuses with fewer.
- **Differential privacy.** Each worker applies the DP settings itself and tracks ε with its own accountant. Under DP the run uses 10 rounds and public feature ranges, and no feature statistics are requested. ε keeps adding up across **Train again**, because privacy loss accumulates on the same data. It resets only on **Clear & restart**, which starts fresh workers. A hospital that has also trained without DP says so on its card. The holdout scores shown each round are for the demo and are not covered by DP.

`src/hub.js` turns each worker into a participant with the same `stats()` / `train()` shape as `createLocalParticipant`, and a test checks that it produces exactly the same weights. It also produces the same weights under secure aggregation, to within rounding. `src/hospital.js` is the worker's dataset handling, DP and personalization, with no Node imports. `src/hospitalWorker.js` adds the parts that need Node's crypto: key agreement and masking.

The results are each hospital's holdout AUC averaged by patient count, not one AUC over a pooled test set as in `npm run demo`, so they read lower (about 0.83 vs 0.886). While a hospital is offline, its patients also drop out of that average.

## Layout and sub-teams

| Sub-team | Files |
|---|---|
| Data cleaning | `src/features.js`, `src/data.js`, `test/data.test.js` |
| Local training function | `src/localTrain.js`, `test/localTrain.test.js` |
| Coordinator and averaging | `src/coordinator.js`, `src/participant.js`, `src/dcp.js`, `src/secureAggregation.js`, `test/coordinator.test.js`, `test/secureAggregation.test.js` |
| Privacy | `src/privacy.js`, `src/secureAggregation.js`, `scripts/privacy.js`, `test/privacy.test.js`, `test/secureAggregation.test.js` |
| Personalization | `src/personalization.js`, `test/personalization.test.js` |
| Evaluation and pitch | `src/model.js` (metrics), `src/baselines.js`, `src/experiment.js`, `src/chart.js`, `scripts/demo.js` |
| Dummy DCP (hospital folders) | `src/hub.js`, `src/hospital.js`, `scripts/dummy-dcp.js`, `scripts/dummy-dcp-worker.js`, `dummy-dcp/`, `test/hub.test.js` |

`data/raw/` holds the four processed UCI files (303, 294, 123 and 200 patients) and `heart-disease.names`. These are the files from the UCI repository, licensed CC BY 4.0. Creators: Janosi, Steinbrunn, Pfisterer and Detrano.

## Ideas for camp

- Tune `mu` (FedProx) and `localEpochs` and see how much the differences between hospitals hurt plain FedAvg.
- Add calibration (log loss, reliability plot) to the demo. That is where later rounds pay off.
- Plot the privacy-vs-accuracy table (`results/privacy.json`) for the pitch.
- Run secure aggregation on DCP by having each hospital's data endpoint serve its masks.
