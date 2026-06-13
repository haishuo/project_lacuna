# Implementation Audit / Spec — Distributional-Consequence Stream

*Spec only. **No code in this step.** Status: **for PI review.** Operationalizes Proposals A + B of
[`ARCHITECTURE-FITNESS-delta-estimation.md`](ARCHITECTURE-FITNESS-delta-estimation.md) (PI-accepted,
verdict "partially valid"). Governed by `NORTH-STAR.md` (§2 identification, §3 generalization axis, §5
metrics, §6 coverage, §8 ladder) and `DECISION-MEMO-P2.2c-consolidation.md` (the three detectability
levels; the phase-locking-bug correction).*

---

## 0. The corrected diagnosis this spec acts on

Not: "the pooling primitive cannot compute order statistics" (false — the §3 probe shows the project's
own `AttentionPooling` fits p90/max R²≈0.97 in isolation). Instead:

> The current architecture does not **naturally expose** within-column, across-row distributional
> consequences to the δ head **under the actual training objective and held-out transfer regime**.

The fix is a **general-purpose distributional-consequence inductive bias** — a stream that answers,
for the supplied target column:

> *What does the observed distribution of this column look like, and how does it differ from the
> distribution expected under a no-mechanism / MAR reference?*

This is mechanism-**general** (useful for LOD, top-coding, self-censoring, and plausibly skip/block
effects), **not** an LOD detector. It is "spatial locality for tabular missingness," in the report's
analogy. The goal is **not** to solve δ; it is to move the **learned channel** off base-rate up to the
**feature-level transfer ceiling** for the *detectable* idiom (LOD), while the *flat* idiom (own-value)
correctly stays flat — i.e. make the detectability spectrum appear **in the held-out learned channel**,
the one of the three levels P2.2c never reached.

### Standing constraints (binding, from the PI)
- No mechanism-specific LOD detector. No metadata channel. No new missingness families in this step.
- No tokenization rewrite unless this audit proves it necessary (it does **not** — see §A1).
- Same LOD-vs-own-value A/B; same semi-synthetic held-out survey split; same leakage gate; same
  manifest; same LR-on-features baseline and oracle framing.

---

## 1. Exact distributional statistics / ECDF representation

The stream has **two sub-representations**, ranked by risk. The first is the headline; the second is
gated behind it and carries an explicit prior-failure warning.

### 1A. Marginal order-statistic representation — PRIMARY (low risk)
Two parallel pieces, both over the **within-column, across-row** axis of the **supplied** target:

- **(i) Fixed value-ECDF — KEEP AS-IS.** The existing 17-dim `consequence_features.compute_consequence_features`
  (`consequence_features.py`: `missing_rate`, z-quantiles q05…q99, z_min/z_max, skew, frac(z>1/1.5/2),
  inter-quantile gaps; z-scored within observed). This is a deterministic ECDF/tail summary of the raw
  observed values and is *already* the feature map whose LR hits the ~0.68 OOF ceiling (DECISION-MEMO
  §4). We keep it because it de-risks the change: it is the proven content, and the experiment isolates
  whether the **learned** path can match/exceed it.
- **(ii) Learned rep-ECDF — NEW, the actual architectural change.** A **differentiable multi-quantile
  pooling** of the encoder's per-row representation of the target column, replacing the mean in
  `_pool_target`. Concretely: project each valid row's target token-rep to a small `m`-dim "shape probe"
  with a learned `Linear(hidden_dim, m)`; for each probe dimension take a **fixed grid of `Q` quantiles
  across rows** (plus max); LayerNorm the result. This is Proposal B done on the right axis: gradients
  flow from the δ loss through the quantiles into the encoder, so the encoder is **taught to expose the
  tail shape** instead of averaging it away.

`torch.quantile` over the row axis is differentiable (piecewise-linear interpolation; well-defined
subgradient) and deterministic (sorting). `max` is sub-differentiable. No RNG enters the stream.

### 1B. Conditional-reference deviation — SECONDARY (gated, HIGH RISK — read the warning)
The "how does it differ from a MAR reference" half. Principle: under MAR (δ=0), missingness depends
only on observed predictors, so **within a stratum of similar predictor values the observed target
distribution is the true conditional distribution — undistorted in shape** (only its count changes,
and count is rate-matched away). Under own-value/LOD MNAR, the observed target is shape-distorted
**even within a predictor stratum**. So: bucket rows by a 1-D observed-predictor summary (e.g. a learned
linear projection of the observed predictors, or the most-correlated observed predictor), compute the
target's tail statistics within each bucket, and represent the **within-bucket tail deficit relative to
the pooled reference**. This is mechanism-general and, in principle, removes the proxy-absorption
confound (P2.2b).

> **⚠ Prior-failure warning (mandatory context).** Predictor-referencing has been tried once and
> **failed**: `transfer_features.py` (predictor-conditional residual asymmetry + predictor-as-ruler)
> gave no-training OOF AUC **0.523 < 0.65**, *worse* than the simple 17 marginal features (0.734)
> (DECISION-MEMO §3). Predictor-referencing **added** column-specificity rather than removing it.
> Therefore 1B is **OFF by default**, is **not** part of the headline run, and may be attempted only
> as a labelled ablation **after** 1A is evaluated, pre-registered against the same no-training LR gate
> that killed `transfer_features`. If 1B cannot beat the simple-marginal LR ceiling **in that gate**,
> it does not reach the neural A/B. This is the §4.9 "no proxy weaker than deployment for a kill" plus
> the pre-registered-gate discipline that caught the last regression.

---

## 2. How it attaches to the existing encoder/head

No new top-level model. Extend the existing `TargetConditionedDeltaModel` (`conditioned_head.py`)
behind config flags, so the A/B harness, manifest, leakage gate, and `train_delta_prior` seam are
reused verbatim.

- **Attachment point:** the head input is currently `cat([evidence, pooled_target, (LayerNorm(consequence))])`
  (`conditioned_head.py:78–88`). The stream **augments** this list with the rep-ECDF vector (1A-ii) and
  **replaces** the `pooled_target` mean with the multi-quantile pooling (1A-ii is literally the new
  `_pool_target`). `evidence` is **kept** (it still carries the cross-column mask structure the encoder
  is genuinely good at — the MCAR/identifiable anchor, §2). The fixed value-ECDF (1A-i) remains the
  existing `consequence` concat path.
- **New deterministic module:** `lacuna/survey/distributional_stream.py` — ONE job (UNIX rule): given
  the encoder's target-column per-row reps + row mask (and, for 1B, the observed predictor block),
  return the fixed-width distributional vector. Pure function of its inputs; no RNG; ≤ ~120 LOC.
- **Plumbing already exists:** `collate` builds `DeltaBatch.consequence` and `target_idx`
  (`batching.py:126–135`); `model(tokens, target_idx, consequence)` is the uniform call
  (`train.py:101,192`). The stream needs only the encoder's `token_representations` (already exposed via
  `return_intermediates=True`, `encoder.py:639`) and the existing `target_idx` — **no tokenizer change,
  no new batch field for 1A.** (1B, if ever run, needs the observed predictor columns, which are already
  in the batch tokens — still no tokenizer change.)

---

## 3. Replace or augment current target pooling

**Both, deliberately:**
- **Replace** the across-row **mean** in `_pool_target` (`conditioned_head.py:67`) with multi-quantile
  pooling (1A-ii). The mean is the specific defect (§2C of the report; mean-pool max R²=0.70 vs
  attention 0.99 in the §3 probe). It goes.
- **Augment** the head input with the rep-ECDF and keep `evidence` and the fixed value-ECDF. We do not
  remove the encoder evidence — it is the cross-column/identifiable signal and is cheap to keep.

A config flag must allow the **pure ablation** (mean-pool, no stream) to reproduce the current floored
result, so the A/B attributes any movement to the stream alone (ladder discipline — the rung-to-rung
attributability that `example_source`/`train.py` were built to preserve).

---

## 4. Dimensionality added

Budget the head input to avoid blow-up (current head is `MLP(cond_dim → hidden → num_bins)`):

| piece | dim | note |
|---|---|---|
| `evidence` (kept) | `evidence_dim` (48–64) | unchanged |
| `pooled_target` → rep-ECDF (1A-ii) | `m·(Q+1)` | proposal: `m=4` probes, `Q=6` quantiles + max ⇒ **28** |
| fixed value-ECDF (1A-i) | 17 | unchanged `consequence_features` |
| **1B (secondary, off)** | ~16 | `n_buckets` deficits + spread; only if gated-in |

Headline added dimensionality over the current model: **+28** (the rep-ECDF replaces a `hidden_dim`-wide
mean, so net head-input change is modest). `m` and `Q` are manifest-recorded knobs; start small (the LR
ceiling says headroom is limited — §6 — so over-parameterizing the stream is wasteful and risks
in-distribution overfit that won't transfer).

---

## 5. Differentiable or fixed

**Both, by design — that is the experiment:**
- **Fixed** (1A-i): deterministic ECDF of raw values; no params; the de-risking control whose LR ceiling
  we are trying to *reach with the learned path*.
- **Differentiable** (1A-ii): learned projection + differentiable quantile pooling, gradients reaching
  the encoder. This is the *mechanism* by which the architecture learns to expose shape — the whole
  point of the change. Determinism note (Rule 6): quantile/sort are deterministic; the only
  non-determinism remains training-time dropout/global-RNG, already documented in `train.py:181–188`.

Rationale: if **fixed-only** already hits the LR ceiling (it does, in-dist) but the learned channel
floors, the question is whether **gradients into the encoder** let the learned channel reach that
ceiling out-of-family. You cannot answer that with a fixed front-end alone; you need the differentiable
path. Keeping both in one run lets us read whether the learned path adds anything over the fixed one
(the pre-registered "features-only must not exceed the learned model, else the learned model adds no
value" rule from the coarse A/B).

---

## 6. How it stays mechanism-general

- The stream computes **only** order statistics / ECDF of the observed target and (1B) MAR-reference
  deviations. **No τ, no step indicator, no LOD parameter, no idiom label** enters it. It is computed by
  the **identical code path** for own-value and LOD examples — the A/B feeds both idioms through the
  same module. Generality is *tested*, not assumed: the success criterion requires own-value to stay
  **flat** through the same stream that lets LOD rise. A stream that lifted both equally (or lifted
  own-value) would be reading an artifact, not a consequence.
- The representation describes "distribution shape and its distortion," a property shared by every
  value-localized or self-censoring idiom — so it is reusable for future families (skip/block, attrition)
  without modification, satisfying the "general distributional stream, not one special case" mandate.
- It respects identification (§2): the stream can only surface **observable-footprint** statistics; it
  never beats the oracle. Where the observed-data law is flat (own-value at matched rate), the stream is
  flat too — the correct high-entropy output (§3, §4.2).

---

## 7. Tests proving no δ/rate leakage (Rule 7 + §4.5/§4.9)

The corpus-level **leakage gate is unchanged and still blocking** (`leakage.py`: |corr(δ,rate)|≤0.2,
per-bin rate within 0.03 of target, rate-only ≤ base-rate+0.1). On top of it, the new module needs:

1. **Rate-orthogonality of every stream feature.** Over a generated corpus, `|corr(stream_feature_j,
   realized_rate)|` must be small for all `j` except the one intentional `missing_rate` slot; and adding
   the stream must **not** raise the rate-only baseline accuracy. (The matched-rate solve already makes
   `missing_rate` δ-uninformative — `consequence_features.py:58` — verify it holds for the new pieces.)
2. **δ-shuffle null.** A no-training LR on the stream features with **δ labels permuted** must collapse
   to chance, in-distribution and OOF. Guards against a dataset-identity artifact masquerading as δ.
3. **Phase-locking guard (regression test).** Reproduce the DECISION-MEMO §3 bug deliberately in a unit
   test and assert the gate runner uses **decoupled / coprime** dataset-vs-δ cycling (the corrected
   `run_p2p2c_coarse_ab._lr_build` pattern). No new diagnostic may re-introduce `i%2 / i%2` locking.
4. **Determinism (Rule 6):** identical inputs ⇒ identical stream vector (no RNG); gradient-flow test
   (loss.backward reaches the encoder through the quantile pooling).
5. **Contract / edge (Rule 1, 7):** degenerate columns (constant observed, < `_MIN_OBS` observed,
   all-missing target) return the documented safe vector and never NaN; out-of-range `target_idx` fails
   loud; scale-invariance (z-scored) preserved.
6. **1B-only gate:** the conditional-reference features must **pass the same no-training LR OOF gate that
   `transfer_features` failed** (OOF AUC ≥ 0.65 **and** ≥ the simple-marginal LR) **before** any neural
   run. Pre-registered; a fail means 1B stops at the gate (no train-and-blame).

---

## 8. What success / failure means (pre-registered)

Evaluation is the **unchanged** held-out LOD-vs-own-value A/B (`run_p2p2c_coarse_ab.py` split:
train cps1988/yrbss/computers/psid7682, val chile/hmda, **test cps1985/workinghours**), binary
(δ0-vs-δ2.5) AUC + 3-bin RPS vs **base-rate** and vs the **LR-on-features** ceiling, leakage-gated,
manifested. Reference numbers to beat: LR binary OOF AUC ~0.68; learned channel currently at base-rate
(DECISION-MEMO §6).

- **SUCCESS (architecture thesis confirmed):** on the held-out test, the **learned** LOD channel beats
  base-rate by **> 2 SE** *and* reaches/exceeds the LR binary OOF ceiling (≈0.68), **while own-value
  stays flat** (within 2 SE of base-rate). ⇒ the detectability spectrum appears **in the held-out
  learned channel** — the missing level. Story: *Lacuna needs a distributional-consequence inductive
  bias, as vision needs spatial locality.*
- **PARTIAL:** LOD **binary** reaches the ceiling but **3-bin** stays ≈ base-rate. ⇒ the learned channel
  now recovers *presence/absence of strong truncation* but not calibrated δ-magnitude — consistent with
  the intrinsically weak, low-resolution OOF transfer ceiling (DECISION-MEMO §4–5). Architecture fix
  *worked* to the extent the ceiling allows; report it as such, do not over-claim.
- **FAILURE:** the learned LOD channel **still floors at base-rate** even with order statistics wired on
  the correct (within-column across-row) axis and gradients reaching the encoder. ⇒ strong evidence the
  binding constraint is the **low OOF transfer ceiling / dataset regime**, *not* the architecture —
  i.e. architecture was *necessary-but-not-even-sufficient*, and the §6-caveat dominates. Redirect to
  the DECISION-MEMO §9 path (domain-randomization to lift the ceiling; the detectability-map +
  sensitivity-reporting reframe). This outcome **partially falsifies** the architecture hypothesis and
  is a legitimate, publishable negative under the validation ladder.

Either way the result is attributable, because the pure-ablation flag (§3) reproduces the current floor
in the same harness, isolating the stream as the only moving part.

---

## Appendix A — audit findings on prerequisites

- **A1. Tokenizer rewrite NOT necessary.** The stream reads (i) raw observed target values — already
  available in `collate`/`consequence_features` — and (ii) the encoder's `token_representations`,
  already exposed (`encoder.py:639`, used by `conditioned_head`). 1B needs observed predictor columns,
  already in the batch tokens. `TOKEN_DIM=4` is untouched. The "no tokenization rewrite unless proven
  necessary" constraint is satisfied: it is **not** necessary.
- **A2. LOC / module discipline.** New `distributional_stream.py` (~120 LOC, one job). `conditioned_head.py`
  (currently 127 LOC) gains the pooling swap + concat — stays well under 500; if it crowds 400, split the
  pooling into the stream module. No god-file growth.
- **A3. Reuse.** `consequence_features` (value-ECDF), `column_stats`/`proxy_score` (predictor selection
  for 1B), `coarse_bins`, `metrics`, `loss`, `leakage`, `run_manifest`, `run_p2p2c_coarse_ab` harness,
  `lod_oracle` framing — all reused unchanged. New surface is the stream module + two `TrainConfig`
  flags (`rep_ecdf_pooling: bool`, `n_shape_probes`/`n_quantiles`) + manifest fields recording them.
- **A4. Scope (Rule 8).** Entirely within `lacuna/survey/` + `scripts/`. No sibling project, no charter,
  no `lacuna/models/encoder.py` change (the encoder is consumed via its existing intermediates, not
  modified). If a future step wants order-stat pooling *inside* the encoder, that is a separate proposal.

---

## Appendix B — open decisions for the PI (before any implementation)

1. **`m` (shape probes) and `Q` (quantiles)** — proposal `m=4, Q=6`(+max) ⇒ +28 dims. Smaller/larger?
2. **Keep the fixed value-ECDF (1A-i) in the headline run, or learned-only?** Proposal: keep both (de-risk
   + the "does the learned path add value over fixed" read).
3. **1B conditional-reference** — authorize as a *gated, post-1A ablation only* (proposal), or exclude
   entirely from this step given its prior failure?
4. **Predictor summary for 1B** (if authorized) — most-correlated observed predictor vs learned linear
   projection. Proposal: defer until 1A's result is known.
5. **Compute** — same CPU envelope as the coarse A/B (~minutes/arm) or GPU? Proposal: CPU first; the LR
   ceiling caps the headroom, so a big-scale run is not yet justified.

**No code, no run until these are settled and the spec is approved.**
