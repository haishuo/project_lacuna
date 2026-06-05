# P2.2c — LOD / Top-Coding Idiom — Detectability-Spectrum Audit / Spec

*Branch `p2/delta-prior-rearchitecture`. Governed by `NORTH-STAR.md` (§2 identification line, §3½
manifold, §5 oracle, §8 validation ladder) and `PHASE-SUMMARY-P2.2-P2.2b.md`. **SPEC ONLY — no code
until approved.** Tests ONE structurally detectable idiom on the SAME semi-synthetic, held-out
real-survey ladder, to answer: does the detectability spectrum exist on real survey X?*

## 0. Purpose (one question, pre-registered)

P2.2b established that *own-value smooth self-censoring, matched rate, single column* is a
**flat-likelihood idiom** on real survey X — after ruling out objective, localization, scale, proxy,
cardinality, and resolution. NORTH-STAR §2 predicts detectability is **mechanism-specific**, with
"sharp truncation / limit-of-detection MNAR" at the **detectable** end. P2.2c tests exactly that
prediction with a **structurally detectable** idiom (LOD / top-coding), changing **only the
mechanism** — same datasets, same matched-rate protocol, same δ-bins / RPS / calibration / leakage
gate / manifest / held-out ladder. **Not a search for a win** — the cleanest remaining test of whether
the footprint channel is alive for *any* region of mechanism space.

## 1. The idiom: LOD / top-coding (sharp, value-localized)

Limit-of-detection / top-coding: values beyond a threshold are disproportionately missing (a sensor
floor, a top-coded income, "≥ \$X" collapsed to missing). The structural difference from own-value
smooth self-censoring is **sharpness + value-localization**: the missingness concentrates in a hard
value-region, so the **observed marginal of the target is truncated with a hard edge** — a strong,
low-entropy "consequence feature" (NORTH-STAR §6) that survives matched rate.

**δ parameterization (nests the spectrum; reuses the δ-bin scale).** On the z-scored view, with a
fixed threshold τ (a recorded quantile, e.g. the 70th percentile of z_target):

    P(missing_target_i) = sigmoid( β₀ + β₁·z_pred_i + δ · 1[ z_target_i > τ ] )

- **δ ≡ the log-odds JUMP in missingness at the threshold** (a logit coefficient, same units as the
  own-value β₂ — so the existing 7 δ-bins apply unchanged; MAR ⇔ δ=0).
- δ=0 → MAR (missingness depends only on the observed predictor; no value-localization).
- δ large → sharp top-coding (cells above τ almost always missing) → observed target hard-truncated
  above τ.
- **β₀ solved for matched marginal rate** (reuse the bisection), for ANY δ — so the *amount* missing
  is never a cue; only the *value-localization* (the edge) carries δ. **β₁** a swept nuisance.

**Why this should be readable where smooth self-censoring was not** (and why we must verify, not
assume): own-value self-censoring also distorts the observed marginal, but *smoothly* and
confoundably with the predictor-MAR term and the unknown baseline conditional. A hard truncation edge
at a consistent location is a distinctive, low-entropy signature. **Whether that intuition holds on
the real-survey manifold is precisely what the §3 oracle adjudicates before any model is trained.**

## 2. Generator design (reuse, do not duplicate)

- New idiom generator = the existing `apply_self_censor` structure with the value term swapped from
  linear `δ·z_target` to step `δ·1[z_target > τ]`. Same matched-β₀ solve, same RNG injection, same
  single-target / others-observed contract, same original-scale observed hand-back.
- **Threshold τ:** a fixed recorded quantile of z_target (e.g. q=0.70 → top 30% region). Held
  CONSTANT across δ and across the δ-grid, so "fraction of the value-axis above τ" is never a δ cue
  (the matched-rate analogue for the location axis — see §5).
- **Answer sheet:** identical schema; `generator_family` becomes an enum admitting the new family
  (`"lod_top_coding"`), and records τ (and its quantile). δ-bin via the existing `assign_delta_bin`.
- **Sources:** the LOD idiom plugs into the existing `ExampleSource` seam (a `SyntheticTwoColSource`
  variant for the oracle/sanity and a `SurveyExampleSource`/`StratifiedRealXSource` variant for the
  real-X ladder), so the model/loss/metrics/leakage/manifest harness is byte-for-byte the same.

## 3. ORACLE-FIRST pre-check (the methodological lesson from P2.2b — NON-OPTIONAL)

NORTH-STAR §5: *"run the oracle before blaming the model for missing a signal that isn't there."*
P2.2b went model-first; P2.2c goes **oracle-first**. Before training any model:

- Adapt the profiled oracle (`feasibility.oracle.bayes_error_nsample` /
  `profiled_oracle.compute_profiled_cell`) to the **step mechanism**: compute the theoretical Bayes
  error / separability of H0 = profiled-MAR vs H1 = LOD(δ) at **matched rate**, on (a) the synthetic
  Gaussian X-model and (b) X-models **fit to the real survey target/predictor columns** we will use.
- **Gate:** the oracle must show the matched-rate LOD footprint carries **substantial** δ information
  (Bayes error well below chance / separability clearly positive) on real-fitted X **before** we
  train. This converts a later model floor into an *attributable* result and a later model success
  into an *expected* one.
- **Decision branch on the oracle alone:**
  - Oracle says LOD is **separable** on real-fitted X → proceed to the model ladder (§4); a model
    floor would then be a real modeling/representation gap, not a missing signal.
  - Oracle says LOD is **also near-flat** at matched rate on real-fitted X → a **major finding**: even
    a sharp idiom is non-identifiable on this manifold at matched rate. We would **not** train-and-blame;
    we'd record that the footprint channel is broadly weak under matched-rate single-column conditions
    and escalate the strategic question — without ever having mis-attributed it to the model.

This oracle step is the single most important addition versus P2.2b.

## 4. Validation ladder (identical to NORTH-STAR §8.1)

Only run if §3's oracle gate passes. Train on real survey X + semi-synthetic LOD holes (known δ);
**validate/test on HELD-OUT real survey datasets** (leave-datasets-out), δ known. Reuse the exact
P2.2 stack: `DeltaPriorModel` / target-conditioned variant, RPS loss, temperature calibration,
δ-bins, leakage gate, run manifest, from-scratch (no checkpoint, all layers trainable). Natural
missingness is **not** in this ladder (face-validity only, later).

## 5. Leakage guards (idiom-specific, on top of the matched-rate gate)

- **Matched marginal rate** across all δ (existing gate): no "how much" cue.
- **Fixed threshold quantile τ** across all δ and the δ-grid: the *fraction of the value-axis* above
  τ is constant, so the value-localization is present only via the *missingness concentration*, not
  via where τ sits. Record τ; assert it is δ-independent.
- **Same δ→rate leakage diagnostic** (`survey.leakage`): δ→realized-rate corr, per-bin rate table,
  rate-only baseline ≈ base rate. `leakage_pass` required.
- **New watch:** a trivial "is the observed target's max/upper-quantile depressed?" feature *should*
  carry δ here — that is the legitimate signal, not leakage. Leakage would be a cue tied to δ that is
  NOT the mechanism's intended footprint (e.g. τ or fraction-above varying with δ). The fixed-τ +
  matched-rate construction closes that.

## 6. Metrics & success criteria (pre-registered)

Per held-out test (and, for the A/B, per dataset): **RPS vs uniform and base-rate** with the **2-SE**
pass bar (per-example RPS for SE), **bin accuracy**, **adjacent accuracy**, **mean predictive entropy
(bits)**, **P(δ=0)**, **ECE**, **interval coverage**, **leakage_pass**, validated manifest.

**Headline success:** on **held-out** real survey datasets, the LOD δ-prior **beats uniform AND
base-rate by ≥2 SE with nontrivial adjacent accuracy** — i.e. it learns out-of-family where own-value
self-censoring did not. **Direct A/B:** run own-value self-censoring and LOD on the *same* held-out
datasets at the same rate/δ-grid; the contrast (LOD discriminates, own-value floors) is the empirical
demonstration of the detectability spectrum on real survey X.

## 7. Outcome interpretation (both branches pre-registered)

- **LOD recovers δ out-of-family** → the footprint channel is **ALIVE**; own-value self-censoring is
  one mapped flat boundary, and the manifold/detectability-spectrum thesis (§3½, §2) gains direct
  empirical support. Lacuna proceeds on its own terms — next would be mapping more idioms and the
  abstention/OOD boundary, not a metadata pivot.
- **LOD also floors on real survey X** (despite the oracle saying signal exists) → a deeper modeling/
  representation question about the footprint channel on the real-survey manifold; *that* — not the
  P2.2b own-value result — would be the trigger to seriously weigh the metadata channel (built only
  with its §8.3 semi-synthetic evaluation protocol). If instead the **oracle** says even LOD is flat
  (§3), the question is about the manifold/rate regime, not the model.

No strategic judgment on the footprint channel is made before this test (and its oracle gate) reports.

## 8. Non-goals

Not a win-hunt; not abandoning the semi-synthetic held-out ladder; no metadata/LLM-prior channel in
this phase; no claim beyond the LOD idiom tested; no natural-data accuracy claim; no architecture or
objective change (same δ-bins/RPS); no other idioms in this pass (skip-logic/attrition later, if the
spectrum is confirmed).

## 9. Reuse vs new code (scope for the approved cycle)

**Reuse unchanged:** `DeltaPriorModel`/conditioned variant, `loss` (RPS/temperature), `metrics`
(+entropy), `leakage`, `run_manifest`, `delta_bins`, the `ExampleSource`/`train` harness, the
profiled-oracle framework. **New (small):** a LOD generator (`survey/lod_generator.py` — step
mechanism + matched-β₀ + answer sheet, reusing the bisection), a one-line `generator_family` enum
extension in `answer_sheet`, a LOD `ExampleSource` (synthetic + real), an oracle adaptation for the
step mechanism, and two runners (oracle pre-check; held-out ladder + A/B). Tests for each
(normal/edge/failure), ≤500 LOC/file, RNG-injected determinism, fail-loud, leakage-gated,
manifest-validated.

## 10. Open decisions for review

1. **τ quantile** — propose q=0.70 (top 30% region); confirm or sweep {0.6, 0.7, 0.8}.
2. **δ-bin reuse** — propose reusing the 7-bin edges (δ is a logit coefficient in both idioms);
   confirm against the realized LOD footprint strength, or recalibrate edges to the LOD δ-grid.
3. **Oracle X-model on real columns** — fit `ConditionalGaussian` to (target, top-predictor) per
   dataset (propose) vs a richer fit; the Gaussian fit is the documented assumption.
4. **A/B scope** — same 8/2/2 leave-datasets-out split as the P2.2 pilot (propose) for a clean
   own-value-vs-LOD contrast.

## Approval gate

This is the P2.2c spec. **No code until approved.** On approval I will implement the LOD generator +
oracle adaptation first, **run the oracle pre-check (§3) and report it before training any model** —
and only proceed to the held-out model ladder if the oracle gate passes, then report the A/B against
own-value self-censoring. Stop for review at the oracle result, and again at the ladder result.
