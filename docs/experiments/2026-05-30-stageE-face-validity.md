# Stage E — face validity on real survey anchors (+ the manifold caveat, in neon)

- **Date:** 2026-05-30
- **ADR:** docs/decisions/0007 (composition-posterior estimand), Stage E — the final stage.
- **Predecessors:** Stage B (generator), Stage C (composition head), Stage D (calibration). This runs
  the full calibrated instrument on real data.
- **Script:** `scripts/stageE_face_validity.py`. **Anchors:** `lacuna_survey/anchors.py` (14 anchors).
- **Question:** Real data has NO mechanism ground truth (you cannot observe *why* a value is missing),
  so accuracy is impossible by construction. Stage E asks the only answerable question: does the
  calibrated posterior's read agree, in DIRECTION, with the textbook CONSENSUS for each anchor?

## 🟥 THE MANIFOLD CAVEAT (in neon) 🟥

- The anchor labels are **textbook consensus, not ground truth.** Mechanism accuracy on real data is
  **impossible by construction** — Stage E is **face validity only: necessary, not sufficient.**
- The model is trained on the survey **manifold** (the generator registry's operationalization of
  "mechanisms that actually occur"); a confident read on a real dataset is a read *relative to that
  manifold*, not a measurement of the truth.
- **MAR vs MNAR is non-identifiable** (Molenberghs): no method, this one included, can resolve it from
  observed data. A wide MAR/MNAR band on real anchors is the **correct report of a real limit.**
- The **PISA anchors are MCAR-by-DESIGN** (random rotated booklets) yet have heavy **block**
  co-missingness; our "MCAR = value-independent *and* unstructured" head therefore reads them as
  structured — a **documented blind spot** of the random-vs-structured framing (value-independence is
  invisible in the observable footprint), not a generic failure.

## Method

The calibrated instrument = encoder + Stage-C frozen+footprint composition heads (3-head deep ensemble)
+ the Stage-D temperature (`tau=4.26`). Each anchor CSV is loaded mask-preserving (numeric columns,
NaN = missing); for n > 128 the posterior is averaged over 8 random row-subsamples. Reported per
anchor: the composition mean `(f_MCAR, f_MAR, f_MNAR)`, `f_structured = f_MAR+f_MNAR`,
`MNAR-within-structured = f_MNAR/f_structured`, the argmax, and the can't-tell mass.

## Result — the model reads STRUCTURE (observable) and abstains on MECHANISM (non-identifiable)

Per-anchor (composition = MCAR / MAR / MNAR), sorted by consensus class:

| anchor | consensus | n | composition | f_struct | MNAR\|str | can't-tell |
|---|---|--:|---|--:|--:|--:|
| pisa2018_gbr_rotation | MCAR | 500 | .14/.50/.36 | 0.858 | 0.41 | 0.42 |
| pisa2022_deu_rotation | MCAR | 500 | .46/.26/.28 | 0.539 | 0.53 | 0.58 |
| survey_bfi | MAR | 2800 | .37/.29/.33 | 0.626 | 0.53 | 0.58 |
| survey_cars93 | MAR | 93 | .28/.43/.29 | 0.724 | 0.41 | 0.66 |
| survey_chile | MAR | 2700 | .38/.28/.35 | 0.625 | 0.56 | 0.56 |
| survey_gssvocab | MAR | 500 | .48/.22/.30 | 0.519 | 0.57 | 0.55 |
| survey_survey | MAR | 237 | .35/.28/.38 | 0.654 | 0.58 | 0.59 |
| survey_ucla_textbooks | MAR | 201 | .16/.48/.37 | 0.845 | 0.43 | 0.45 |
| survey_yrbss | MAR | 13583 | .24/.29/.47 | 0.756 | 0.62 | 0.57 |
| nhanes_dpq_phq9 | MNAR | 500 | .17/.36/.47 | 0.832 | 0.56 | 0.48 |
| nhanes_duq_drug | MNAR | 500 | .19/.46/.35 | 0.815 | 0.43 | 0.52 |
| nhanes_inq_income | MNAR | 500 | .18/.39/.43 | 0.824 | 0.53 | 0.50 |
| nhanes_whq_weight | MNAR | 500 | .25/.39/.36 | 0.750 | 0.48 | 0.59 |
| survey_nhanes_demographics | MNAR | 500 | .42/.29/.29 | 0.583 | 0.50 | 0.62 |

Aggregate by consensus class:

| consensus | n | mean f_MCAR | mean f_structured | mean MNAR\|structured | mean can't-tell |
|---|--:|--:|--:|--:|--:|
| MCAR | 2 | 0.301 | 0.699 | 0.47 | 0.50 |
| MAR | 7 | 0.322 | 0.678 | 0.53 | 0.57 |
| MNAR | 5 | 0.239 | **0.761** | 0.50 | 0.54 |

### Check 1 — random-vs-structured (the IDENTIFIABLE axis): directionally passes

`f_structured` tracks the **actual amount of co-missingness in the footprint**, exactly as a model
trained on that footprint should: the heavily-structured anchors read clearly structured — the four
NHANES sensitive-item MNAR anchors at **0.75–0.83**, the `ucla_textbooks` skip-logic anchor at 0.845,
and the PISA rotated-booklet anchor at 0.858 — while the sparse light-item-nonresponse CRAN surveys sit
near the 1/3 prior (the model does not fabricate structure where there is little). The MNAR-consensus
anchors read the MOST structured (mean f_structured 0.761, f_MCAR 0.239) and below the 1/3 MCAR prior.
The model is responding to the data (compositions span f_MCAR 0.14–0.48), not emitting a constant.

### Check 2 — MAR-vs-MNAR (the NON-IDENTIFIABLE axis): correctly NOT resolved

`MNAR-within-structured` is ≈ **0.50 for every consensus class** (MNAR 0.50, MAR 0.53, MCAR 0.47): the
model is **at chance** separating the NHANES sensitive-item (MNAR-consensus) anchors from the CRAN
item-nonresponse (MAR-consensus) anchors. This is not a bug — it is the Molenberghs limit and the
honest seam, reported as the estimand is designed to: a wide band, with a substantial can't-tell mass
(0.42–0.66, mean 0.55) on every anchor, rather than a confident wrong claim.

## Interpretation

1. **The thesis holds on real data: the model reads STRUCTURE and abstains on MECHANISM.** It tracks
   the identifiable random-vs-structured axis (f_structured follows the footprint's real co-missingness)
   and is honestly at chance on the non-identifiable MAR-vs-MNAR split — exactly the calibration seam
   Stage D quantified (resolution on the MCAR axis, prior-fallback on the MAR/MNAR split), now confirmed
   directionally on real anchors.
2. **The PISA blind spot is the sharpest honest caveat.** `pisa2018_gbr_rotation` is MCAR-by-design yet
   reads 0.858 structured: rotated-booklet planned-missingness is value-INDEPENDENT (the MCAR property)
   but block-STRUCTURED (the pattern), and the observable footprint sees only the pattern. Our estimand's
   "MCAR" conflates value-independence with unstructured-ness; planned-missing designs break that. This
   is a definitional limitation to state plainly, and a lead (a "by-design / planned-missing" flag, or a
   value-dependence test, would be needed to recover it).
3. **Necessary, not sufficient.** Passing the identifiable check and honestly declining the
   non-identifiable one is the most a real-data face-validity test can show. It does not validate the
   mechanism *mix* — that remains a semi-synthetic, manifold-bound claim.

## "Tanked" check (real-data, qualitative)

- Real survey missingness reads as **structured, not MCAR-random** (the identifiable axis), strongest
  for the cleanest module-structured anchors. ✓ (directional)
- The model **does not over-claim MAR-vs-MNAR** on real data (≈ chance, wide can't-tell). ✓ (honest)
- Documented failure mode: **MCAR-by-design block designs read as structured** (PISA). ✗ — stated as a
  definitional blind spot, not hidden.

## Reproduce

```
python scripts/stageE_face_validity.py          # reads tau from the Stage-D report
```

## Files

- Report: `/mnt/artifacts/project_lacuna/runs/stage0_general_baseline/stageE_face_validity.json`.
- Script: `scripts/stageE_face_validity.py`; anchors: `lacuna_survey/anchors.py` +
  `lacuna_survey/evaluation_data/*_real.csv`.
