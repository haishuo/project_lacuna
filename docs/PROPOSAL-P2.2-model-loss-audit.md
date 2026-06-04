# P2.2 — δ-Prior Model + Loss — Implementation Audit / Spec

*Branch `p2/delta-prior-rearchitecture` (continues from P2.1, commit `f3a5fad`). Governed by
`docs/PROPOSAL-P2-delta-prior-rearchitecture.md`, `docs/NORTH-STAR.md`, and the project
`CLAUDE.md`. **SPEC ONLY — no code, no training, no metrics until approved.***

## 0. Scope (what P2.2 is / is not)

**Is:** the FIRST δ-prior model — a calibrated, ordered distribution over the 7 δ-bins
(P2.1 `delta_bins`) — built by attaching a NEW δ-bin head to the validated `LacunaEncoder`
backbone, trained **from scratch** on P2.1 generator output + answer sheets, with a proper
ordinal scoring loss (RPS), minimal training loop, calibration tracking, and the first
out-of-family calibration number.

**Is not (deferred to P2.3+):** abstention / uncertainty gate, proxy-score, OOD/manifold,
estimand/reporting layer, per-column (fully-missing-table) granularity, multi-idiom δ. None
of those are built or stubbed here.

**Carry-overs from P2.1 review (must honor):**
1. The realized-rate leakage check becomes a **formal diagnostic over the full generated
   corpus** (not a smoke table) — §10. Any systematic δ→rate association is a *blocking* issue.
2. Both `target_rate` and `realized_rate` are already recorded per example (answer sheet) and
   are recorded per run (manifest); a tolerance/rejection policy is considered but not yet
   enforced — §10.

---

## 1. Exact seam after `evidence = encoder(...)`

The codebase audit's clean seam (PROPOSAL §1) is **confirmed in code**. In
`lacuna/models/assembly.py::LacunaModel.forward` the encoder is called at lines 363–369:

```
encoder_output = self.encoder(batch.tokens, batch.row_mask, batch.col_mask,
                              return_intermediates=True)
evidence = encoder_output["evidence"]            # [B, evidence_dim]
```

Everything downstream of `evidence` in `LacunaModel` — reconstruction heads (§2 of assembly),
missingness-feature extractor, MoE gating, `get_class_posterior` (3-class), `BayesOptimalDecision`
— is the **dead 3-class objective** and is OUT of the P2.2 graph.

**Decision: do NOT instantiate `LacunaModel` for P2.2.** Build a new, thin top-level module
`DeltaPriorModel = LacunaEncoder + DeltaBinHead`. The seam is therefore a *composition*
boundary, not a runtime branch: the δ-model never constructs reconstruction/MoE/decision
modules, so they cannot leak into the loss, the gradient, or the parameter count.

- `LacunaEncoder` is self-contained (`lacuna/models/encoder.py`): `forward(tokens, row_mask,
  col_mask, return_intermediates=False) -> evidence [B, evidence_dim]`. We call it WITHOUT
  intermediates (the δ head needs only `evidence`; token-level reps feed reconstruction/proxy,
  not P2.2).
- Param count of the backbone alone (base config hidden=128/evidence=64/4L/4H/max_cols=32):
  **836,226**; mini (64/32/2L/2H): **111,298**. The full `LacunaModel` (v1.0) is ~900k — the
  delta between that and the encoder is exactly the reconstruction/MoE machinery we are dropping.

**Why a new module rather than reusing `feasibility/model_arm.py`:** model_arm trained the FULL
`LacunaModel` and read `posterior.p_class`, then renormalized MAR/MNAR (`binary_q`). That was a
temporary binary probe riding the 3-class head (constraint forbids reusing it as the P2 main).
P2.2 cuts at the seam instead and discards the 3-class head entirely.

## 2. δ-bin head architecture

New module `lacuna/survey/delta_head.py`: `DeltaBinHead(nn.Module)`.

- **Input:** `evidence [B, evidence_dim]`.
- **Body:** small MLP — `Linear(evidence_dim, hidden) → GELU → Dropout(p) → Linear(hidden, NUM_BINS)`
  — mirroring the existing `GeneratorHead`/`ClassHead` idiom (`lacuna/models/heads.py`). Single
  hidden layer for the first run; `hidden` defaults to `evidence_dim`. Output = **raw logits
  `[B, NUM_BINS]`** (NUM_BINS = 7 from `survey.delta_bins`). Xavier-init, zero-bias (as
  `GeneratorHead._init_weights`).
- **No softmax inside the head.** The head returns logits; the loss consumes logits; a separate
  `predict_proba` path applies `softmax(logits / T)` with the calibration temperature T (§7).
- **Ordering is imposed by the LOSS (RPS over the bin CDF), not the architecture.** We deliberately
  do NOT hard-wire a cumulative-link (ordinal-threshold) parameterization in v1 — a plain
  K-logit head + RPS is the simplest order-aware choice and keeps the head swappable. Cumulative-
  link is a recorded future variant (§13), not v1.
- **Temperature:** a single scalar `T` (post-hoc, fit on val — §7). Stored on the model as a
  non-trained buffer set after training; `T=1.0` during training. (Mirrors v1.0 temperature
  scaling, but on δ-bin logits, not gate logits.)
- **Pooling granularity:** v1 = ONE δ-prior per (dataset with one censored target column) — the
  P2.1 example unit. Per-column pooling on fully-missing tables is deferred (PROPOSAL §2 flag);
  the head signature (`evidence → bins`) does not preclude it later.

`DeltaPriorModel(nn.Module)` (in `delta_head.py` or a thin `delta_model.py` if LOC requires the
split) wires `encoder` + `head`:
`forward(batch) -> logits [B, NUM_BINS]` (calls `encoder` at the seam, then `head`).

## 3. From-scratch training for main runs

**Yes — fresh-init, all-layers-trainable, no checkpoint, for every `kind="main"` run.** This is
a hard charter constraint (§4.9) and a forbidden-shortcut (no frozen encoder + stapled head; no
checkpoint loaded for main; no RF/MLP).

Enforcement (reuse the P1 pattern `model_arm.assert_fresh_and_trainable`, generalized into the
survey trainer):
- Construct `DeltaPriorModel` from a factory (fresh random init via an injected `RNGState`-seeded
  torch generator — determinism, Rule 6). Never `load_state_dict` from disk for main.
- Assert `sum(numel for p) == sum(numel for p if p.requires_grad)` → record
  `trainable_param_count` and `all_layers_trainable=True` in the manifest.
- `checkpoint_loaded=False` is asserted and recorded; the manifest validator (§11) rejects
  `kind="main" AND checkpoint_loaded=True` (already enforced in `feasibility.manifest`; the
  survey manifest gains the same guard).
- Best-val weights may be restored **in memory** at the end of training (as model_arm does) —
  that is early-stopping selection, NOT a loaded checkpoint, and is recorded as such.
- `kind ∈ {main, ablation, smoke}`; only `main` is reported as a result. A frozen-encoder probe,
  if ever run, is `kind="ablation"` and never headlined.

## 4. Removing / bypassing the old 3-class & reconstruction heads

By construction (see §1): P2.2 **does not instantiate** `LacunaModel`, so reconstruction heads,
the MoE, `get_class_posterior`, and `BayesOptimalDecision` are never created and cannot
participate. Concretely:

- **No edits to `lacuna/models/*`.** The v1.0 production model stays intact and untouched
  (PROPOSAL §11 "production v1.0 model untouched"). We only *import* `LacunaEncoder` /
  `EncoderConfig` / `create_encoder` from it.
- **No call passes through `forward_classification_only`, the loss matrix, or `p_class`.** The
  δ-model's only forward returns δ-bin logits.
- **Tokenization:** we reuse `tokenize_and_batch` for tokens/row_mask/col_mask only. The δ-bin
  **label is carried OUT-OF-BAND** as a parallel `delta_bin [B]` long tensor alongside the
  `TokenBatch` — it is NOT stuffed into `generator_ids`/`class_ids` (those fields belong to the
  3-class world and stay unused/`None`). This keeps the δ objective free of any MAR/MNAR class
  semantics (no binary, no 3-class — forbidden shortcuts).

## 5. RPS loss definition + tests

New module `lacuna/survey/loss.py`. Primary loss = **Ranked Probability Score (RPS)** — the
discrete CRPS over the ordered δ-bin CDF; proper and order-aware (PROPOSAL §4).

Definitions (per example; mean-reduced over batch):
- `p = softmax(logits)` over K = NUM_BINS bins, `P_k = sum_{j<=k} p_j` (predicted CDF).
- One-hot truth `y` at the true bin `b`; truth CDF `Y_k = 1[k >= b]`.
- `RPS = sum_{k=0..K-1} (P_k - Y_k)^2`  (the standard discrete RPS; equals K·Brier-on-CDF).
  Optionally normalized by `(K-1)` so it lands in `[0,1]` — record which convention is used.
- Differentiable in `logits` (softmax → cumsum → squared error); used directly as the training
  objective. Lower is better; an order-respecting miss (predict bin 5 when truth is bin 4) costs
  less than a far miss (predict bin 0).

Properties to assert (`tests/unit/survey/test_loss.py`):
- **Normal:** perfect confident prediction at the true bin → RPS = 0. Uniform prediction →
  a known constant (computed closed-form for K=7) — regression-pinned.
- **Order-awareness (the defining property):** for a fixed true bin, RPS is monotonically
  non-decreasing as the predicted point mass moves farther (in bin index) from the truth —
  e.g. mass on bin 4 < bin 3 < bin 0 in loss when truth = 5. This is what a plain CE cannot see
  and is the reason RPS is the primary loss.
- **Properness sanity:** expected RPS under a fixed true-distribution is minimized by predicting
  that true distribution (numerical check on a small simplex grid).
- **Gradient:** `loss.backward()` produces finite grads; gradient w.r.t. logits is zero at the
  exact one-hot-matching softmax limit (within tolerance).
- **Edge:** δ=0 truth (bin 0) and δ-tail truth (bin 6) — boundary bins behave (no index error;
  CDF endpoints `P_{K-1}=1`).
- **Failure:** logits/label shape mismatch, label out of `[0,K)`, K<2 → loud `ValueError`
  (Rule 1). Label dtype non-integer → loud.

## 6. Secondary diagnostics (reported, not optimized)

Same module or `lacuna/survey/metrics.py`. All are *reporting* metrics; the optimizer only sees
RPS (§5):
- **Ordinal/cross-entropy log-score** — `−log p_{true bin}` (a second proper score; sanity vs RPS).
- **E[δ] error** — predicted `E[δ] = Σ p_k · center_k` using bin centers (a defined,
  recorded mapping: finite bins → midpoints; bin 0 → 0; open tail bin 6 → a fixed representative,
  e.g. its lower edge + a recorded offset). Report MAE/RMSE vs the answer-sheet δ.
- **δ-bin accuracy** — `argmax p` vs true bin (top-1), plus **adjacent accuracy** (within ±1 bin,
  the order-aware analogue).
- **`P(δ=0)`** — mass on bin 0 (the "plausibly MAR?" quantity), reported but not yet thresholded
  (abstention is P2.3).

These are logged per epoch (val) and at test; accuracy is explicitly **secondary to calibration**
(charter §4.2).

## 7. Calibration tracking

- **Temperature scaling (post-hoc):** after training, fit a single scalar `T` by minimizing
  val NLL (log-score) on the δ-bin logits. Reuse `lacuna/training/calibration.py::find_optimal_temperature`
  **iff** it can be applied to a bare logits/label tensor; if it is too coupled to `LacunaModel`'s
  gate logits, implement a small, tested `fit_temperature(logits, labels) -> T` in `survey/loss.py`
  (golden-section / LBFGS on NLL). Record this decision.
- **Report before/after** T-scaling for every metric below.
- **ECE on bin probabilities** — reuse `lacuna/metrics/calibration.py::compute_ece` (confidence
  = max-prob, correctness = top-1) for a first ECE; additionally report a δ-aware reliability
  using the CDF (see interval coverage). Record `n_bins`.
- **Reliability diagram** — binned confidence-vs-accuracy table emitted to the run dir (data, not
  a plot dependency).
- **Interval coverage** — does the X% credible interval over δ-bins (central mass from the
  predicted bin distribution) contain the true bin X% of the time? Report coverage at e.g.
  50/80/90% nominal. This is the headline calibration check for an ordered prior (charter §4.2).

## 8. Training-data generation plan (uses P2.1)

- **Source X:** the real survey pool already auto-registered in `create_default_catalog()`
  (`survey_bfi, survey_cps1985, survey_cps1988, survey_psid1976, survey_psid7682, survey_hmda,
  survey_workinghours, survey_yrbss, survey_computers, survey_chile, survey_cars93, survey_survey`).
  Each load is complete-case (ingestion drops NaN rows).
- **Mechanism:** `survey.delta_generator.generate_self_censor_example` per example —
  own-value self-censoring on one sampled non-constant target, predictor = most-correlated column,
  matched-β₀ rate solve (no rate cue), answer sheet attached.
- **δ sampling:** sample δ from a recorded **grid that includes 0.0 (MAR)** and spans all 7 bins,
  with deliberate mass on bin 0 so `P(δ=0)` is learnable (e.g.
  `{0.0, 0.15, 0.4, 0.75, 1.25, 1.75, 2.5}` plus jitter — finalized against bin edges; recorded
  in manifest `delta_grid`). The supervised label is `assign_delta_bin(δ)`.
- **β₁ nuisance:** sampled from a recorded non-negative range (e.g. `[0, 2]`) — swept, not a label.
- **Rate:** a fixed `target_rate` (e.g. 0.25–0.30) shared across all δ (the leakage defuse).
  May later randomize rate within a recorded band; v1 fixes it and records it.
- **Determinism:** every example drawn from an injected `RNGState` with a recorded base seed;
  the answer sheet records the per-example seed (Rule 6). Subsample large datasets via the existing
  `subsample_raw` before generation (row-independent, distribution-preserving).
- **Volume:** a recorded number of examples per epoch (e.g. a few thousand), regenerated per epoch
  with fresh spawned seeds (fresh data each epoch, as the v1.0 loader does) — labels stay exact
  because the generator imposes δ.
- **Tokenization:** `tokenize_and_batch(datasets, max_rows, max_cols)` with `max_cols >=` the
  widest survey d (survey_bfi d=28 ⇒ `max_cols=32` base config fits). δ-bin labels carried
  out-of-band (§4).

## 9. Train / validation / test split scheme

The headline metric is **out-of-mechanism-FAMILY calibration** (charter §5, PROPOSAL §8). In
P2.1 there is exactly ONE mechanism family (own-value self-censoring), so a true out-of-family
split is not yet possible — that is a P2.4 concern when more idioms exist. For P2.2 the FIRST
out-of-family number is approximated by the closest available axis: **leave-datasets-out**
(generalization to unseen *survey datasets*, i.e. unseen real-X distributions), which is the
analogue we can run now.

- **Split by source dataset, not by example** (avoid leakage of a dataset's X across splits):
  partition the ~12 survey datasets into disjoint train / val / test pools (e.g. 8 / 2 / 2),
  recorded explicitly in the manifest `split_scheme`.
- **Train pool:** generate examples on the fly (fresh δ/β₁/seed per epoch).
- **Val pool:** held-out datasets; used for early stopping AND temperature fit (§7). A **fixed,
  seeded** val set (not regenerated each epoch) so early-stopping decisions are comparable.
- **Test pool:** held-out datasets never seen in train or val; a **single fixed seeded** set
  used once for the final reported numbers (never for selection) — mirrors model_arm's
  shared-held-out-test discipline.
- δ-bin **stratification** within each split so all bins (esp. bin 0) are represented; recorded.
- **Explicitly record** that this is leave-DATASETS-out (a distribution-shift proxy), and flag
  leave-mechanism-FAMILY-out as the real charter metric pending P2.3+ idioms — no overclaiming.

## 10. Matched-rate leakage diagnostics (formal — P2.1 carry-over #1)

Promote the P2.1 smoke check to a **first-class, blocking diagnostic computed over the whole
generated corpus** and recorded in the manifest:

- **δ→rate association:** across all generated examples, regress / correlate `realized_rate` on
  `delta` (and on `delta_bin`). Report Pearson r, slope, and a per-bin table of mean±sd realized
  rate. **Acceptance gate:** no systematic trend — |r| below a recorded threshold and per-bin
  means statistically indistinguishable from `target_rate` within finite-sample bands
  (`survey.delta_generator.rate_tolerance`). A systematic δ→rate cue is a **blocking issue** (the
  model could read δ off the rate instead of the footprint) — the run is not reported until fixed.
- **δ=0 vs δ>0 rate parity:** mean realized rate for bin-0 (MAR) examples vs bin>0 examples must
  match within tolerance (the core "no rate cue between MAR and MNAR" check).
- **Rate-only baseline (ablation, `kind="ablation"`):** a tiny logistic/threshold on realized
  rate alone *should* be ~chance at predicting δ-bin. If a rate-only baseline can predict δ, the
  matched-rate solve has failed — blocking. (This baseline is a leakage probe, NOT a main model;
  it never headlines — forbidden-shortcut compliance.)
- **Record both rates** (already in answer sheet + manifest). A tolerance/rejection policy
  (resample an example whose realized rate drifts beyond k·`rate_tolerance`) is **specified as
  optional and OFF by default in v1**; if enabled it is recorded in the manifest with its
  threshold and the count of resampled examples (no silent truncation).

## 11. Manifest fields

Extend the P2 manifest discipline. P2.1 shipped `lacuna/survey/manifest.py` (generator/data
config). P2.2 adds a **model/loss/run manifest** — either new fields on the survey manifest or a
sibling `survey/run_manifest.py` (decided at build time by LOC/Rule-3). Required (fail-loud,
no field interpreted unless present & non-null):

- Generator block (reuse P2.1): `generator_family`, `generator_path`, `delta_formula`,
  `beta0_solver`, `delta_bins`, `delta_grid` (must include 0.0), `beta1_range`, `target_rate`,
  `dataset_pool`, `answer_sheet_schema_version`, `answer_sheets_saved`, `seed`.
- Model: `model_arch` (encoder config + head spec), `trainable_param_count`,
  `checkpoint_loaded` (**must be False for main**), `all_layers_trainable` (**True for main**).
- Objective: `loss="RPS"` (+ normalization convention), `num_bins`, secondary metrics list.
- Calibration: `temperature`, `ece` (before/after), `interval_coverage` (before/after),
  `reliability` ref, `log_score`, `rps`.
- Eval: `split_scheme` (leave-datasets-out, pools listed), `e_delta_error`, `bin_accuracy`,
  `adjacent_accuracy`, `p_delta0_summary`.
- Leakage (§10): `delta_rate_corr`, per-bin realized-rate table, `rate_only_baseline_acc`,
  `leakage_pass` (bool).
- Provenance: `run_id`, `git_commit`, `timestamp` (all injected — Rule 6), `wall_clock_seconds`,
  `kind ∈ {main, ablation, smoke}`.
- Validator rejects: missing/null required field; `kind="main"` with `checkpoint_loaded=True`;
  `kind="main"` with `all_layers_trainable!=True`; `delta_grid` lacking 0.0; `leakage_pass=False`
  for a `main` run (a main result may not be reported if leakage failed).

**No metric is interpreted unless the manifest validates** (charter; mirrors
`feasibility.manifest`).

## 12. Smoke-test plan

A `kind="smoke"` end-to-end path (tiny + fast, CPU, seeded) proving the pipeline runs before any
real `main` training:
- mini encoder (64/32/2L/2H) + δ head; a few hundred examples from 2–3 survey datasets; 1–2
  epochs; tiny batch.
- Asserts: forward returns `logits [B, NUM_BINS]`; RPS loss is finite and `backward()` steps;
  loss **decreases** over the smoke run on an easy high-δ-vs-δ=0 split (learnability sanity);
  determinism — same seed reproduces identical logits/loss; temperature fit returns finite `T>0`;
  ECE/coverage/E[δ]-error compute without error; a valid smoke manifest writes and re-validates;
  the leakage diagnostic runs and emits its table.
- Unit tests (`tests/unit/survey/`): `test_delta_head.py` (shapes, init, determinism, failure on
  bad evidence_dim/num_bins), `test_loss.py` (§5), `test_metrics.py` (E[δ], coverage, ECE wiring),
  `test_run_manifest.py` (required/null/guard cases), and a fast trainer smoke
  (`test_train_smoke.py`) behind a small config so the suite stays quick. Every new module gets a
  test file (Rule 7); all files ≤500 LOC (Rule 4); determinism via injected RNG (Rule 6);
  fail-loud boundaries (Rule 1).

## 13. Success / failure criteria for the first minimal run

**Success (a trustworthy first δ-prior):**
- Manifest validates; `kind="main"`, `checkpoint_loaded=False`, `all_layers_trainable=True`,
  `leakage_pass=True`.
- **Leakage clean (gate):** `|corr(δ, realized_rate)|` below threshold; δ=0 vs δ>0 rate parity
  holds; rate-only baseline ≈ chance at δ-bin. (If this fails, nothing else is reported.)
- **Learns signal:** test RPS beats two references — (a) the uniform-prediction RPS constant and
  (b) a "predict the training δ-bin marginal" base-rate RPS — by a margin outside its CI.
  Adjacent-bin accuracy clearly above the base-rate adjacent accuracy.
- **Calibration first:** interval coverage near nominal (e.g. 80% interval covers ~80±band of
  test truths) and ECE small AFTER temperature scaling, reported before/after. Calibration
  quality, not raw accuracy, is the pass criterion (charter §4.2).
- **δ=0 behaves:** on MAR (δ=0) test examples the model puts substantial mass on bin 0
  (`P(δ=0)` high on average) rather than confidently asserting a high bin — the null-behavior
  sanity (PROPOSAL §8). A confident high-δ claim on MAR data is a red flag.
- **Out-of-(dataset)-family generalization:** the above hold on the held-out *test* dataset pool,
  not just val.

**Failure / red flags (do NOT paper over — charter §4.5):**
- Any leakage-gate failure (systematic δ→rate, rate-only baseline predicts δ) → blocking.
- Coverage far from nominal or ECE large after T-scaling → overconfident prior; report as a
  failure of the first run, not a tuning footnote.
- Suspiciously *perfect* discrimination → check for a leaked cue (rate, column-selection
  artifact, a degenerate dataset) before celebrating.
- Model collapses to predicting one bin (e.g. always bin 0 or always the modal bin) → degenerate;
  report.
- Train/val/test dataset overlap or label-in-band leakage → invalidates the number.

## 14. Constraints honored (restated)

No old 3-class objective; no binary MAR-vs-MNAR objective; no frozen encoder + stapled head as
main; no checkpoint loaded for main; no RF/MLP baseline as main; no abstention/OOD/proxy/reporting
layer (P2.3+). v1.0 production model untouched. All metrics gated behind a validating manifest.

## 15. Proposed module plan (P2.2 — new code only; production model untouched)

- `lacuna/survey/delta_head.py` — `DeltaBinHead` + `DeltaPriorModel` (encoder@seam + head).
- `lacuna/survey/loss.py` — RPS (+ log-score) + `fit_temperature` (or documented reuse of
  `training.calibration`).
- `lacuna/survey/metrics.py` — E[δ] error, bin/adjacent accuracy, ECE wiring, interval coverage,
  reliability table, `P(δ=0)`.
- `lacuna/survey/train.py` — from-scratch loop (fresh init, all-trainable assert, early stop on
  val RPS, in-memory best-val restore, injected RNG); emits the run manifest.
- `lacuna/survey/run_manifest.py` (or extended `survey/manifest.py`) — model/loss/eval/leakage
  fields + validator (§11).
- `lacuna/survey/leakage.py` — the §10 corpus-level δ→rate diagnostic + rate-only baseline probe.
- Tests for every module (Rule 7); ≤500 LOC each (Rule 4); RNG-injected determinism (Rule 6);
  fail-loud (Rule 1).

## Approval gate

This is the P2.2 audit. **No code until approved.** On approval I will implement the modules
above on `p2/delta-prior-rearchitecture`, keep the suite green, produce a validated `main`
manifest + the first out-of-(dataset)-family calibration number with the leakage gate passing,
and report — then proceed to P2.3 (abstention/governance) with review between.
