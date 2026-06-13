# P2.2b — Rung 3b — High-Row Real-X Diagnostic — Implementation Note

*Short pre-run note (not a full audit). Separates the two surviving P2.2 bottleneck candidates —
per-example evidence SCALE vs real-X GEOMETRY — left ambiguous after rung 3.
No new model/loss/tokenization code: rung 3b is a RUN configuration of the existing
target-conditioned path (`train_delta_prior` + `SurveyExampleSource` + `TargetConditionedDeltaModel`).*

## Design rationale (one paragraph)

Rung 1 (synthetic 2-col, **n=1024**) learned strongly; rung 3 (real-X, target-conditioned,
**n=160**) floored. The two differ in (i) rows/example and (ii) synthetic-vs-real X. Rung 3b holds
the protocol like rung 1 — **in-distribution, high rows, target-conditioned** — but on REAL survey
X, so the only thing separating rung 3b from the rung-1 *pass* is real-X geometry. Clean read:
**learns ⇒ the P2.2/rung-3 floor was SCALE; floors ⇒ real-X GEOMETRY / signal expression.**

## 1. Selected datasets and row counts

Large, NARROW real survey datasets (narrow keeps high-row training tractable on CPU and removes
table width as a confound — width was already cleared by rung 1 at d=2):

| dataset | n | d |
|---|---|---|
| survey_cps1988 | 28,155 | 3 |
| survey_yrbss | 11,522 | 5 |
| survey_computers | 6,259 | 7 |
| survey_psid7682 | 4,165 | 6 |
| survey_chile | 2,590 | 4 |
| survey_hmda | 2,380 | 6 |

All n ≥ 2,380 ≥ max_rows (full 1024-row evidence per example); all d ≤ 7.

## 2. Selected max_rows

`max_rows = 1024` (matches rung 1's per-example evidence), `max_cols = 8` (fits the widest d=7;
keeps the encoder cheap at 1024 rows so the run finishes on CPU).

## 3. Train/val/test split

**In-distribution** (same protocol as the rung-1 pass, so the comparison is clean): the 6-dataset
pool is shared across train/val/test; val and test are FIXED, separately-seeded fresh draws
(stratified over the δ-grid). This is a LEARNABILITY diagnostic, not a generalization test — using
leave-datasets-out here would re-introduce a generalization burden and reconfound the geometry-vs-
scale question. (Leave-datasets-out at high rows is a sensible follow-up only if rung 3b learns.)

## 4. Expected runtime

~15–25 min CPU. Config: hidden 96 / evidence 48 / 2 layers, batch 16, 40 train batches/epoch,
≤20 epochs (patience 5), val/test 105. Per-step cost ≈ rung 3 (more rows but far fewer columns →
less attention); ~800 steps.

## 5. Leakage diagnostics

Unchanged blocking gate (`survey.leakage`): δ→realized-rate Pearson, per-bin realized-rate table
vs the matched target, rate-only baseline vs base rate. `leakage_pass` must be True for the number
to be interpreted; recorded in the manifest.

## 6. Manifest fields

Standard `survey.run_manifest` (`kind="main"`): generator block (family, δ-grid incl. 0.0, β₁
range, target_rate, answer-sheet schema), model block (`model_path`, `model_arch` incl.
`target_conditioned=True` + `conditioning=head_side_target_token_pool`, `trainable_param_count`,
`checkpoint_loaded=False`, `all_layers_trainable=True`, `num_bins=7`), `loss="RPS"`, temperature,
`split_scheme` (records in-distribution + the dataset pool + max_rows), metrics (before/after T),
calibration (ECE/coverage), leakage block + `leakage_pass`, wall-clock. Validated before any metric
is read.

## 7. Success criterion

**RPS must beat BOTH the uniform-prediction RPS and the base-rate ("predict the δ-bin marginal")
RPS by ≥ 2 × the test SE of RPS, AND show nontrivial adjacent accuracy** (clearly above the
chance/marginal ±1-bin level). The runner computes the per-example RPS (reduction="none") to get
the test SE and reports `(uniform − rps)/SE` and `(base_rate − rps)/SE`.

## Interpretation (the user's rules, restated)

- **Beats both by ≥2 SE with nontrivial adj-acc ⇒ SCALE** was the P2.2/rung-3 bottleneck (real-X is
  learnable given rung-1-like row evidence) → next is rung 5 (scale curve) / a leave-datasets-out
  high-row run.
- **Still at the uniform floor ⇒ real-X GEOMETRY / generator-signal expression** is the bottleneck
  (rows, width, localization, generalization all controlled) → redesign the signal/representation,
  not the scale.
- **Partial ⇒ report how RPS/adj-acc scale with rows; whether it looks sample-size limited.**

## Scope

Implement/run ONLY this diagnostic (`scripts/run_p2p2b_rung3b.py`), then stop for review. No model/
tokenization changes, no binary/3-class objective, no abstention/OOD/reporting, no P2.3.
