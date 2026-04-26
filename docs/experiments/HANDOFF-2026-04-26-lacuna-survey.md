# Session handoff — 2026-04-26 (Lacuna-Survey deployment stack)

## TL;DR

Project Lacuna's mission is to encode missingness-mechanism domain
knowledge as a tool — replacing or augmenting expert review of the
MCAR / MAR / MNAR distinction. Over an eight-version iteration arc
this session, we established that **specialisation by collection
process is structural, not a feature** (a generic Lacuna can't transfer
to real-world data because mechanism plausibility is intrinsically
domain-bound), built **Lacuna-Survey** as the first specialised
variant, and added a **deployment-layer stack** (real-data calibration
+ OOD detection) that wraps the trained model without touching the
base architecture.

The current state is "best so far but not yet at the user's
correctness bar": **4/5 within-domain real-survey MAR detection**,
**3/4 cross-domain confidently-wrong cases flagged via OOD**, all
1071 unit tests passing.

The next work is **data acquisition** — adding real-survey anchors to
the calibration corpus so the calibration is no longer structurally
biased. The user is pursuing this manually (free public sources +
possible university grant for restricted-use data).

**Critical: do NOT make architectural changes to base Lacuna without
explicit user authorisation.** Variant-specific work belongs in
`lacuna_survey/`; base framework changes go in `lacuna/`.

## Read this first (15 minutes)

In order:

1. **`docs/decisions/0005-lacuna-survey-iteration-arc.md`** — the
   v1–v8 iteration record + post-v8 calibration/OOD work + open
   directions. This is the work spec.
2. **`lacuna_survey/README.md`** — variant package overview, where
   each asset lives, typical workflow.
3. **`lacuna_survey/DATA_ACQUISITION.md`** — concrete URLs and
   acquisition instructions for additional MNAR / MCAR anchors.
4. **`docs/decisions/0004-remove-mle-slot-keep-rest.md`** — the
   prior decision that explicitly removed Little's test from the
   feature pipeline. Do NOT propose re-adding any classical MCAR
   test family in any combinator (feature, ensemble, deployment
   wrapper); ADR 0004 is settled and the 6-family bakeoff arc
   ruled them out empirically.
5. **`CLAUDE.md` Rule 8** (cross-project scope boundary) — do not
   modify pystatistics, pystatsbio, or any sibling project on your
   own initiative. They've been touched together with Lacuna in
   prior sessions; they are not yours to touch in this one without
   explicit user authorisation.

## State to verify before starting

```bash
# Tests should be green
cd /mnt/projects/project_lacuna && \
  /home/haishuo/miniconda3/envs/lacuna/bin/python -m pytest -q --ignore=tests/slow tests/
# expected: 1071 passed, 1 skipped

# Diagnostic should give the documented numbers (within-domain MAR
# 4/5; OOD flags Pima/Pima_tr2/hitters)
/home/haishuo/miniconda3/envs/lacuna/bin/python -m lacuna_survey.diagnostic --calibrated --ood
# expected per ADR 0005: agreement 5/8 MAR-consensus, OOD flags on
# pima_uci, pima_tr2, hitters, pbc

# These files should exist:
ls demo/model.pt                                              # v8 checkpoint
ls demo/model.json                                            # sidecar metadata
ls lacuna_survey/deployment/calibration.json                  # fitted calibration
ls lacuna_survey/deployment/ood_detector.json                 # fitted OOD
ls /mnt/data/lacuna/nhanes/demographics_clean.csv             # 122 MB raw NHANES
ls lacuna_survey/evaluation_data/                             # 8 anchor CSVs
```

If any of those don't match, something has drifted since this handoff
was written. Inspect ADR 0005 and figure out which version diverged.

## What's open

The user is most likely to want one of these in their next session,
in rough order of likelihood:

### A. Ingest newly-downloaded real data (most likely)

The user said they would manually download data per
`DATA_ACQUISITION.md`. They will probably arrive with one or more
of:

- **NHANES sensitive-item subset** (drug use, sexual behavior — MNAR)
- **PISA 2018 rotated-booklet data** (MCAR by design)
- **ESS rotating-module data** (MCAR by design)
- **NLSY income items** (MNAR)
- **HRS wealth items** (MNAR)

For each one they bring, the workflow is:

```bash
# 1. Stage and validate (this prints a template Anchor row)
python -m lacuna_survey.import_anchor /path/to/downloaded.csv \
    --slug <descriptive_slug> --max-rows 500

# 2. Edit lacuna_survey/anchors.py to add the row with the
#    consensus mechanism + citation. The label is a JUDGEMENT
#    CALL — verify against the missing-data literature; do not
#    just trust the README of the source dataset.

# 3. Refit calibration
python -m lacuna_survey.calibrate

# 4. Re-run diagnostic to see the new state
python -m lacuna_survey.diagnostic --calibrated --ood

# 5. Update ADR 0005 with the new diagnostic numbers + the
#    expanded anchor count (1 MNAR + 7 MAR → ?)
```

If the source is in a non-CSV format (SAS XPT, SPSS .sav, Stata .dta),
write a per-source parser at `lacuna_survey/parsers/<source>.py` that
takes the raw file and emits a clean numeric CSV ready for
`import_anchor`. Document the parser. Use `pyreadstat` for
SAS/SPSS/Stata (already installed in the lacuna conda env).

**Critical for labelling**: do not casually label things "MNAR" without
checking. Many datasets the user might bring have MIXED missingness
sources — e.g. NHANES demographics has cycle-coverage admin-NaN mixed
with income-refusal MNAR. Filter / preprocess to isolate the
mechanism-of-interest before labeling. ADR 0005 has the NHANES
example as a worked case.

### B. The user is dissatisfied with within-domain detection (4/5) and wants more architectural work

If so, the explicit options ranked in ADR 0005 are:

3. **Encoder-side regularization** — base-Lacuna change. This is
   the only remaining direction that touches base Lacuna. Discuss
   carefully with the user before starting; they've been explicit
   about not wanting drift in base Lacuna defaults.
4. **Active learning on real data** — would require model
   retraining (not just calibration refit). Big effort — only do
   this if user explicitly authorises retraining and accepts the
   cost. Not yet justified by the data we have.

### C. The user wants to start a new variant (Lacuna-Records, -Instrument, etc.)

Per ADR 0005's organisational pattern, mirror the `lacuna_survey/`
structure for the new variant:

```
lacuna_<variant>/
├── __init__.py
├── README.md
├── anchors.py
├── calibrate.py
├── ood.py
├── import_anchor.py
├── DATA_ACQUISITION.md
├── evaluation_data/
└── deployment/
```

Plus YAML configs at `configs/generators/lacuna_<variant>.yaml` and
`configs/training/<variant>.yaml`.

Branch the codebase ONLY if the variant requires fundamentally
different model computation (e.g. temporal attention for
Lacuna-Longitudinal). Don't branch as a precaution.

## Critical constraints

1. **Rule 8 — cross-project boundary**. Do not modify pystatistics,
   pystatsbio, or any other sibling project under `/mnt/projects/`
   without explicit per-session user authorisation. Things you might
   want to touch (e.g. `pystatistics.mvnmle.little_mcar_test`) — flag
   to the user, do not edit.

2. **No re-introducing classical MCAR tests**. ADR 0004 + the
   `mcar-alternatives-bakeoff` arc tested six families (Little MLE,
   MoM, propensity, HSIC, MissMech, median-split) and found none
   added signal. Don't propose ensemble/feature/wrapper re-additions.

3. **No silent base-Lacuna architecture drift**. The single Survey-
   specific architectural choice is `model.learn_evidence_attenuation:
   true` in `configs/training/survey.yaml`. Base Lacuna defaults to
   `False`. Don't flip the base default, and don't add new
   variant-specific architectural choices to base Lacuna without
   discussing.

4. **Calibration label must come from literature, not heuristic**.
   When the user brings a new dataset, do NOT label its mechanism by
   inspecting the data; LOOK UP the consensus reading in the
   missing-data literature (Allison 2001, van Buuren 2018, Schafer
   1997, NCHS / OECD / NORC methodology reports). The dataset's
   README is not authoritative for mechanism — those are usually
   convenience assumptions, not verified properties.

5. **Don't celebrate or recommend "shipping"**. The user has been
   explicit that this isn't a demo build. The bar is correctness on
   real-world data. Report results honestly without framing them as
   victories. Negative results are valuable; partial fixes are
   partial.

## When done

- Update `docs/decisions/0005-lacuna-survey-iteration-arc.md` with
  the new state (anchor count, latest diagnostic numbers, any new
  candidate directions surfaced).
- Run unit tests one more time: should still be 1071 passed.
- Delete this handoff file (`docs/experiments/HANDOFF-2026-04-26-lacuna-survey.md`).
- Report back with: (a) what changed, (b) new diagnostic numbers,
  (c) test count.

## File / directory map

```
lacuna/                              # base framework — touch with care
├── data/
│   ├── missingness_features.py     # 10-dim feature extractor (added value-conditional 2026-04-26)
│   └── semisynthetic.py            # apply_missingness with z-score normalisation
├── models/
│   ├── moe.py                      # learnable evidence-attenuation (default off in base)
│   └── assembly.py                 # threads model config through
├── training/
│   └── trainer.py                  # per-class weights option (default None = unweighted)
└── ...

lacuna_survey/                       # variant package — Survey-specific work goes here
├── __init__.py
├── README.md                        # variant overview
├── DATA_ACQUISITION.md              # acquisition guide for next anchors
├── anchors.py                       # declarative anchor registry
├── calibrate.py                     # vector-scaling calibration fitter
├── ood.py                           # supervised OOD classifier
├── import_anchor.py                 # validate + stage user-provided CSV
├── diagnostic.py                    # --calibrated and --ood flags
├── evaluation_data/                 # 8 real-survey CSVs (gitignored)
└── deployment/
    ├── calibration.json
    └── ood_detector.json

configs/
├── generators/
│   ├── lacuna_tabular_110.yaml     # generic registry
│   └── lacuna_survey.yaml          # survey-specialised registry (60 generators)
└── training/
    ├── semisynthetic_full.yaml     # generic training config
    └── survey.yaml                 # Survey training config (with evidence-attn opt-in)

docs/
├── decisions/
│   ├── 0004-remove-mle-slot-keep-rest.md  # the Little's-test ban
│   └── 0005-lacuna-survey-iteration-arc.md # the v1–v8 + deployment record
└── experiments/
    └── HANDOFF-2026-04-26-lacuna-survey.md # THIS FILE — delete when done

/mnt/data/lacuna/                    # gitignored data — Forge-only
├── raw/survey_*.csv                 # 12 survey training X-bases
└── nhanes/demographics_clean.csv    # 122 MB raw NHANES (subset extracted to evaluation_data)
```

## Numerical reference (for sanity check)

Diagnostic state at session end (calibrated + OOD):

```
                     consensus  pred         conf    P(OOD)
airquality           MAR        MNAR ✗        46%    0.00 (in)  ← contested textbook
pima_uci             MNAR       MAR  ✗        49%    0.55 (OOD) ← OOD-flagged
pima_tr2             MNAR       MAR  ✗        92%    0.60 (OOD)
hitters              MNAR       MAR  ✗        94%    0.99 (OOD)
mass_survey          MAR        MAR  ✓        88%    0.00 (in)
pbc                  MAR        MCAR ✗        38%    0.62 (OOD) ← clinical, debatable
survey_bfi           MAR        MAR  ✓        81%    0.08 (in)
survey_yrbss         MAR        MAR  ✓        40%    0.12 (in)
survey_chile         MAR        MAR  ✓        86%    0.00 (in)
survey_cars93        MAR        MAR  ✓        94%    0.02 (in)
survey_survey        MAR        MNAR ✗        56%    0.00 (in)  ← unambiguous miss
```

Within-domain real surveys (the test that matters): 4/5 MAR-consensus
correct (bfi, yrbss, chile, cars93; survey_survey misses).
Cross-domain MNAR: 0/3 correct but 3/3 flagged OOD.
Anchor corpus: 8 datasets (1 MNAR / 7 MAR / 0 MCAR).
Learned calibration: T=2.20, bias=[+2.84, −2.76, −0.10].
Learned evidence attenuation in v8: α = 0.241.
