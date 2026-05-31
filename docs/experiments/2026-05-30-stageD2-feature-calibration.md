# Stage D-v2 — feature-conditional calibration: a clean null (per-instance error isn't footprint-predictable)

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007 (composition-posterior estimand), Stage D follow-up.
- **Predecessor:** Stage D (global temperature calibration). This tests whether the per-instance
  uncertainty *ranking* — which a global temperature cannot touch — can be fixed.
- **Module:** `lacuna/training/composition_recalibration.py` (+ `composition_loss.dirichlet_nll`); tests.
  **Script:** `scripts/stageD2_feature_calibration.py`.
- **Question:** Stage D made the posterior's region statements reliable on average (ECE 0.072→0.044) but
  the can't-tell mass did not track per-instance error (vacuity-error corr ≈ −0.15 → 0). Can a
  temperature that DEPENDS on the observable footprint fix the per-instance ranking — and can it at all?

## Method (attribution-first)

1. **Feasibility probe.** Regress per-dataset composition error `|mean − realised|₁` on the 20-D footprint
   (RandomForest, fit on a calibration split, scored on a disjoint test split). Held-out R² > 0 ⇒ a
   feature-conditional temperature has signal to exploit; R² ≤ 0 ⇒ the error is non-identifiable noise
   and a global temperature is the best attainable (a documented null).
2. **Feature-conditional temperature.** A small net maps footprint → per-dataset temperature
   (`FeatureTemperature`), fit on the calibration split by minimising the Dirichlet NLL (a proper score
   — the Bayes-risk expected CE does NOT work, it over-rewards sharpening-when-right and collapses to a
   near-global temperature; verified). A temperature preserves the mean's argmax, so it can only change
   per-dataset CONFIDENCE, never the predicted composition — any change is pure calibration.

The module is validated on SYNTHETIC feasible data (tests): when hardness IS a footprint feature, the
fit learns to spread the confidently-wrong datasets (τ_hard 25 vs τ_easy 0.5) and lifts the
vacuity-error correlation to +0.82. So the tool works; the question is whether REAL data is feasible.

## Result — NULL: the per-instance error is not predictable from the footprint

Calibration split n=960, test n=640 (held out from Stage-C training and from each other):

```
FEASIBILITY PROBE: |error| predictable from footprint?  held-out R² = -0.308, corr(pred,actual) = 0.117
```

| method | ECE ↓ | Brier ↓ | can't-tell | corr(vacuity, error) |
|---|--:|--:|--:|--:|
| uncalibrated | 0.0715 | **0.1473** | 0.256 | −0.006 |
| **global temperature (Stage D)** | **0.0442** | 0.1505 | 0.594 | −0.145 |
| feature-conditional (Stage D-v2) | 0.0651 | 0.1615 | 0.576 | +0.011 |

- **The probe is decisive: held-out R² = −0.308** (the footprint→error regressor does *worse* than
  predicting the mean error; corr a weak 0.117). The per-dataset composition error is **not predictable
  from the observable footprint.**
- **The feature-conditional temperature is therefore dominated.** It nudges the vacuity-error
  correlation from −0.145 to ≈0 (+0.011) — no useful ranking — while making BOTH the reliability (ECE
  0.044→0.065) and the proper score (Brier 0.151→0.162) WORSE: with no signal to fit, it overfits the
  calibration split's noise. The **global temperature (Stage D) remains the best calibration.**

## Interpretation

1. **Why the error is unpredictable: it is the footprint's UNEXPLAINED residual, and it is ~homoscedastic
   in the footprint.** The head's mean is already the footprint's best guess at the composition, so the
   error is the part the footprint cannot explain. The probe (R² < 0) shows this residual has roughly
   CONSTANT magnitude across the footprint space — there is no "this footprint looks ambiguous" signal to
   key on. So no feature-conditional temperature can rank datasets by reliability. (This is not
   tautological — heteroscedastic residuals *would* be exploitable; they simply are not present here.)
2. **The footprint carries the COMPOSITION signal but not the per-instance ERROR signal.** Stage C: a
   RandomForest on the same 20-D footprint reaches composition L1 0.385 (strong). Stage D-v2: the same
   footprint predicts |error| at R² < 0 (none). Honest uncertainty therefore lives at the **axis level**
   (the seam — Stage D/E: confident on random-vs-structured, a wide band on MAR-vs-MNAR) and the **global
   level** (calibrated reliability — Stage D's temperature), **not per-instance.**
3. **This was the right call to defer.** When asked whether to run D-v2 before Stage E, the prediction was
   that the per-instance error might be irreducibly unpredictable; Stage E confirmed the axis-level seam
   is the operative deliverable, and Stage D-v2 now confirms the per-instance refinement is a null. A
   clean negative result, attributable to non-identifiability rather than a modelling shortfall.

## "Tanked" framing

Stage D-v2 does NOT beat the Stage-D global temperature (worse ECE and Brier). This is the intended,
attributable outcome of a feasibility-gated experiment: the gate (R² < 0) said no signal, and the
recalibrator confirmed it. The global temperature stands as the calibration of record.

## Caveats and leads

- **Bounded by the 20-D footprint.** Richer deployable observables (raw per-column distributional
  features beyond the footprint, or the encoder evidence) are an untested lead — but the footprint is
  precisely what carried the *composition* signal, and it does not carry the *error* signal, so the
  ceiling is likely low.
- The module is a working, tested tool; it is a null **on this manifold**, not a broken implementation —
  it lifts the vacuity-error correlation to +0.82 on synthetic feasible data.

## Reproduce

```
python scripts/stageD2_feature_calibration.py --cal-batches 60 --test-batches 40 --batch-size 16
```

## Files

- Report: `/mnt/artifacts/project_lacuna/runs/stage0_general_baseline/stageD2_feature_calibration.json`.
- Module: `lacuna/training/composition_recalibration.py` (+ `composition_loss.dirichlet_nll`);
  script `scripts/stageD2_feature_calibration.py`; tests `tests/unit/training/test_composition_recalibration.py`.
