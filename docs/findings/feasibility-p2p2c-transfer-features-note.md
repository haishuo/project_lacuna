# P2.2c — Transfer-Robust Consequence Features — Implementation Note

*Short pre-implementation note (feature list to be FROZEN before the run). One bounded iteration on
the consequence-feature representation, framed as a specific falsifiable hypothesis, gated by a
no-training transfer test BEFORE any model training. Same A/B, same held-out validation, same
leakage gate / manifest / loss. No encoder / tokenization / metadata / mechanism change.*

## Hypothesis (specific, falsifiable)

> The current 17 consequence features fail to transfer leave-datasets-out because they are
> **within-column summaries** (z-scored-within-observed shape stats) whose δ→feature relationship —
> both its DIRECTION and its baseline offset — depends on the column's own marginal distribution.
> Transfer requires expressing the observed target **relative to a per-column reference that is
> untruncated under the mechanism** (the fully-observed predictor / the predictor-conditional
> expectation), so that (a) the δ effect points the SAME way across columns and (b) the MAR (δ=0)
> baseline is approximately column-agnostic.

The diagnostic that motivates this: current features give in-distribution binary AUC 0.94 but
out-of-family AUC 0.43 (direction *flips* across columns) — a column-specific map.

## Design principle

LOD top-codes the upper tail of the target *given its predictors*. The **predictor is fully
observed and untruncated** — so it is a transfer-robust "ruler" for where the target's upper tail
*should* reach and for the conditional expectation the residual is measured against. Features are
therefore built from (i) **predictor-conditional residuals** and (ii) **target-vs-predictor
upper-reach comparisons**, and where possible expressed as **departures that are ≈0 under MAR for
any column** (asymmetry/deficit measures), not absolute shape values.

## Proposed FROZEN feature set (~12 features; finalize before the run)

Per example: observed-target rows `t` (where target observed), predictor column(s) `p` at those
rows (fully observed). Ridge-fit `t ~ p` → residual `e`; robustly standardize `e` by its IQR
(`e* = (e − median)/IQR`). Both `t` and `p` z-scored within observed for the ruler features.

**Group A — predictor-conditional residual asymmetry** (upper-truncation ⇒ consistent direction):
- A1 `skew(e*)` — upper-tail truncation skews residuals negative.
- A2 `(q95(e*) − q50) / (q50 − q05(e*))` — upper/lower residual spread ratio; < 1 under upper cut.
- A3 `q975(e*) − q50` minus `q50 − q025(e*)` — signed upper-vs-lower reach **deficit** (≈ 0 under
  symmetric MAR residuals; negative under top-coding).
- A4 `frac(e* > +1) − frac(e* < −1)` — tail-count asymmetry (≈ 0 under MAR).

**Group B — predictor as the transfer-robust ruler** (target's upper reach vs the untruncated
predictor's):
- B1 `zq95(t) − zq95(p)` ; B2 `zmax(t) − zmax(p)` — target reaches *less far* up than the predictor
  under top-coding.
- B3 `(zq99(t) − zq50(t)) − (zq99(p) − zq50(p))` — upper-spread deficit vs predictor.
- B4 `corr(t, p)` measured in the **upper half** of `p` minus in the **lower half** — top-coding
  attenuates the target–predictor association where the target is high.

**Group C — anchors / context:**
- C1 `missing_rate` (matched across δ ⇒ no δ cue; anti-leakage anchor).
- C2 `corr(t, p)` overall.

All quantities are scale-free (ratios, rank/IQR-standardized, z-difference, or correlations). The
**asymmetry/deficit features (A3, A4, B-differences) are designed to be ≈0 under MAR for ANY
column**, which is the property that makes the δ-direction transfer. (Optional later: ECDF/rank
features and a MAR-resampled reference — excluded this pass to keep the set small and frozen.)

`N_TRANSFER_FEATURES` fixed once finalized; recorded in the manifest with the schema.

## NO-TRAINING TRANSFER GATE (run BEFORE any model training — the GO/NO-GO)

Compute the frozen features on semi-synthetic LOD examples and fit a logistic regression:
- **in-distribution** (train+test on the same datasets) — confirm the features still carry the
  signal (expect high AUC, like the 0.94 we saw);
- **leave-datasets-out** (train on the train pool, test on the held-out pool) — the decisive number.

**Gate:** proceed to model training ONLY if the LR **transfers out-of-family** (OOF binary δ0-vs-δ2.5
AUC clearly > 0.5 by a recorded margin, e.g. ≥ 0.65, and OOF 7-bin clearly above chance). If the LR
does **not** transfer, STOP before training and report — *"the feature representation is still not
transferable"* — and do **not** blame the encoder (per the decision rules). This makes the cheap LR
the arbiter of the hypothesis before spending a training run.

## Then: the same A/B (only if the gate passes)

Exact same held-out leave-datasets-out A/B as before — own-value vs LOD, target-conditioned, same
δ-bins / RPS / calibration / leakage gate / manifest / split — with the new frozen transfer features
replacing the 17 (a `TrainConfig` flag selecting the feature set; the old set stays available for
provenance). The 95th-pct / external diagnostics remain external, not training targets.

## Decision rules (pre-registered, from approval)

- **LOD learns out-of-family, own-value flat** → detectability spectrum demonstrated **in the held-out
  learned channel**; transfer-robust consequence features validated. (Headline target.)
- **LR transfers but the model does not** → model-integration issue (e.g. features swamped by
  encoder evidence) — investigate integration, not features.
- **LR itself does not transfer (gate fails)** → representation still not transferable; do NOT blame
  the encoder; pause and reassess whether LOD transfer needs a different modeling formulation.
- **own-value also learns** → audit leakage / whether a feature encodes δ or rate directly.

## Constraints (hard, per approval)

No encoder redesign; no tokenization change; no metadata channel; no new mechanisms; no loss/objective
change. Same A/B (own-value vs LOD), same held-out leave-datasets-out validation, same leakage gate
and manifest. Feature list FROZEN before running. Include the no-training LR baseline (in-dist + OOF).
Unit-test feature computation / missing handling / determinism / shape / scale-invariance before the
run. This is a bounded representation iteration — **if it fails, pause and reassess the modeling
formulation rather than piling on more features.**

## Order

Note (this) → finalize+freeze the feature list → implement module + wiring + tests (suite green) →
run the **no-training LR transfer gate** and report → ONLY if it passes, run the same A/B → report →
stop.
