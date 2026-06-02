# Stage Q — the per-column SUBTYPE layer (ADR-0008 commitment 6, the restored Q3)

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md, **commitment 6** (Q3 returns as a per-column
  *subtype detector + prior*, NOT the Stage-5 per-column 3-way classifier that collapsed).
- **Branch:** `experiment/subtype-layer` (off `experiment/metadata-prior`).
- **Modules:** `lacuna/priors/{dirichlet_evidence,subtype_ontology,subtype_prior}.py`,
  `lacuna/data/subtype_targets.py`, `lacuna/models/subtype_likelihood.py` (+124 tests).
  **Scripts:** `scripts/stageQ_subtype_layer.py`, `scripts/stageQ_feature_attribution.py`.
  **Reports:** `runs/stage0_general_baseline/stageQ_{subtype_layer,feature_attribution}.json`.

## What Q builds

The whole ADR-0008 prior × likelihood arc (P0–P6) ran at MECHANISM granularity (MCAR/MAR/MNAR). Q
extends the *same* machinery to SUBTYPE granularity — where property 3's literal output lives ("fairly
certain 20% is THRESHOLD MNAR … unable to determine for the rest"). Two separable channels, fused per
column, aggregated to a dataset subtype-composition by missing cell, with a calibrated abstain mass.

- **Ontology O (the fused Dirichlet support, K=5):** `threshold_mnar`, `detection_mnar` (the LOUD,
  data-detectable MNAR subtypes), `self_censoring_mnar` (the quiet/non-identifiable MNAR catch-all),
  `mcar`, `mar` (data-silent per column). Parent mechanisms roll the subtype-composition up to the
  ADR-0007 mechanism-composition.
- **Likelihood (data):** a per-column readout over `L = {threshold, detection, reject}`. It detects
  the loud fingerprints and folds the *entire* non-identifiable region (quiet self-censoring MNAR +
  MAR + MCAR) into one **reject** class. This is the design move that avoids the Stage-5 trap: the
  readout never makes the per-column MAR-vs-MNAR call that collapsed; it makes the *identifiable*
  loud-vs-reject call. The L→O embedding sends reject mass to **zero evidence** (the data DEFERS to the
  prior on the non-detectable classes), so a column the data rejects does not pull the posterior off a
  prior's self-censoring lean.
- **Prior (metadata):** the subtype prior (`subtype_prior.py`): `lab_lod → detection_mnar`,
  `planned_random → mcar`, `skip_gated → mar`, `sensitive_disclosure → self_censoring_mnar`,
  `indeterminate → flat`; tiered + confidence-gated as the mechanism prior. `threshold_mnar` has **no**
  semantic class — a sharp cutoff is a DATA fingerprint, not a codebook fact — so the prior is flat on
  it and the data leads there (the honest seam at subtype granularity).
- **Fuse → calibrate → abstain:** combine in raw evidence (P2), aggregate by missing cell (ADR-0007),
  temperature-calibrate the combined subtype-composition (P5), per-column selective decision with a
  risk-coverage commit threshold (P6). On semi-synthetic the prior reliability is *simulated* at a
  known ρ (P5/P6 method) to separate calibration from correctness; the real frozen spec is checked for
  consistency against the grounded benchmark.

Regime: matched miss rate, full diversity (`mnar_diverse + mar_diverse`; diverse pools ⇒
`compensate_rate` is a no-op). 5 seeds, mean ± sd. **Rate caveat (a correction — see Result 1):**
the loud (threshold/detection) columns realise a *lower* miss rate (~0.197) than the reject region
(~0.248) even here — a residual generator under-compensation, NOT a real-world fact. A *lower* rate is
just as exploitable a cue as a higher one; the production subtype detector therefore EXCLUDES
`missing_rate` (the rate-free, transferable readout), and the rate-included number is reported only as
a contaminated upper bound.

## Result 1 — the data channel: stable but WEAK detection (rate-confound CORRECTED), honest abstain (5 seeds)

The headline detector is **rate-free** (excludes `missing_rate`); the rate-included readout is shown
only as a contaminated upper bound, with the audit that motivates excluding it:

| likelihood metric (5 seeds) | rate-FREE (honest) | rate-INCLUDED (contaminated bound) |
|---|---|---|
| loud-vs-reject ROC-AUC | **0.773 ± 0.040** | 0.819 ± 0.018 |
| loud-vs-reject AP (base rate 0.112) | 0.333 ± 0.032 | 0.421 ± 0.043 |
| loud recall @ precision 0.6 | 0.108 ± 0.040 | 0.214 ± 0.082 |
| reject-correct on the silent region | **0.979 ± 0.003** | 0.989 ± 0.006 |
| realised miss rate (loud / reject) | 0.197 / 0.248 | — |

- **The rate audit (a correction to the first Stage-Q headline).** Even at "matched" rate the loud
  columns realise **0.197** vs the reject region's **0.248** — a residual generator under-compensation
  (the threshold/detection pools undershoot the 0.25 target). `missing_rate` is a deployable feature, so
  the detector exploited it: a feature ablation gives **AUC 0.819 with rate → 0.771 without**, and
  **miss-rate ALONE scores 0.759** — i.e. *most* of the original 0.82 was a NON-TRANSFERABLE rate cue.
  On real data a threshold/detection column has no characteristic miss rate (it is just where the cutoff
  sits), so a learned "lower rate → loud" rule would not transfer. This is the **Stage-5 confound
  recapitulated** (and the first Stage-Q write-up mis-reasoned that "MNAR misses *less* ⇒ no cue" — a
  systematic gap in *either* direction is exploitable). The honest, transferable detector excludes rate:
  **AUC 0.773 ± 0.040**.
- **The Stage-5 collapse is still solved by the reframe.** The per-column 3-way MCAR/MAR/MNAR
  classifier was winner-take-all unstable at matched rate (Stage-5 per-class recall sd **0.30–0.43**,
  acc ~0.33). The loud-vs-reject detector is **not** that: AUC 0.773 ± 0.040 — an order of magnitude
  more stable, even rate-free. Folding the non-identifiable region into one reject class and asking the
  data only the *identifiable* question is what removes the collapse (the rate cue was a separate,
  now-removed inflation).
- **Honest abstain holds.** The data correctly rejects **98%** of the silent/MAR/MCAR region.
- **But the detector is WEAK — weaker than first reported.** Rate-free AUC 0.77 / AP 0.33; usable recall
  is only ~11% of loud columns at 60% precision. **The Stage-4 single-mechanism "threshold/detection
  detectable at 0.9–1.0" does NOT transfer** to matched-rate mixtures — self-censoring MNAR's truncation
  footprint overlaps the loud cliff (loud-vs-self-censoring RF AUC ~0.73). The data confidently flags a
  small minority of loud columns and abstains on the rest.

## Result 2 — attribution: the signal is in the features; the encoder DILUTES it (3 seeds)

`scripts/stageQ_feature_attribution.py`, loud-vs-reject ceiling (RF), matched rate, loud base 0.106:

| input to the RF | ROC-AUC | AP | recall @P.5 |
|---|---|---|---|
| deployable distributional features (5-D) | **0.808 ± 0.021** | 0.366 | 0.212 |
| frozen-encoder per-column token reps (64-D) | 0.649 ± 0.020 | 0.205 | 0.050 |
| features + encoder reps | 0.756 ± 0.021 | 0.298 | 0.070 |

The loud signal lives in the deployable observed-value features; the frozen dataset-task encoder reps
are barely above chance and **dilute** the features when concatenated (0.81 → 0.76). This is why the
canonical `ColumnReadoutHead` (encoder reps + features) collapsed to all-reject in early runs, and why
the Q likelihood is a deployable-feature detector with **no encoder, no oracle**. It echoes Stage C's
"the encoder is the bottleneck / under-represents the footprint", now at subtype granularity. (Among
feature models: RF/GBM ~0.85–0.86 ≫ histogram-GBM ~0.79 ≫ a small MLP ~0.78 — the RF is the lens.)
(These absolute AUCs *include* `missing_rate`, so they carry the same rate inflation Result 1 corrects;
the *relative* conclusion — encoder reps dilute, features carry the signal — is unaffected, and the
rate-free features-only ceiling is ~0.77.)

## Result 3 — fusion: prior-led, calibrated, safe; the override is weak (5 seeds)

Per-column posterior → dataset subtype-composition (by missing cell) + per-column abstention. Three
conditions: data-only, a *reliable* simulated prior (ρ=0.85), a *near-chance* one (ρ=0.45); favoured
prob 0.70 (the commitment-1 overridable cap).

(Fusion uses the rate-FREE detector — the honest one.)

| condition | comp L1 | query-ECE | Brier (prior-only 0.119) | coverage @acc.8 | committed-acc | abstain mass |
|---|---|---|---|---|---|---|
| data-only | 0.916 | 0.090 | 0.118 | **~0.00** | **0.0** (never commits) | **1.00** |
| reliable prior | **0.700** | **0.040** | **0.085** | **0.997** | 0.822 | 0.003 |
| near-chance prior | 0.858 | 0.037 | 0.110 | ~0.00 | 0.0 | 1.00 |

committed subtype-accuracy split (at the target, reliable prior): loud 0.824 / silent 0.822.

- **Data-only abstains on EVERYTHING at subtype granularity** ("unable to determine" is the honest
  default): rate-free, the data commits on ~0% of columns at an 80%-accuracy target and *never* reaches
  the target on its confident set — it cannot reliably name *which* subtype (threshold↔detection
  unresolved). Its subtype-composition equals the prior-only base-rate predictor (Brier 0.118 vs 0.119).
- **A reliable prior supplies exactly the resolution the data lacks** — coverage@80% lifts ~0 → 1.0,
  committed-accuracy 0.82, calibrated (ECE 0.040), beats prior-only on the proper score (Brier 0.085 vs
  0.119). The prior × likelihood payoff at subtype granularity (the P5/P6 result transfers).
- **A near-chance prior manufactures no false confidence** (coverage stays ~0) — the calibration +
  abstention prevent a bad prior from inventing commitment.
- **The override (commitment 1) is very weak (~0.06).** With the rate confound removed the data is
  weaker still, so it rarely drags the posterior off a wrong subtype prior. **The subtype layer is
  PRIOR-DOMINATED**: rate-free, the data essentially only sharpens the prior, it does not override it.
  (The earlier "data protects the loud axis at 0.95 under a bad prior" was itself partly the rate cue;
  rate-free, the data does not commit confidently enough to protect that axis — an honest downgrade.)
- The real frozen subtype-prior spec is **1.0 consistent** with the benchmark's `gold_subtype` (n=43) —
  a P1-style consistency check (the spec faithfully encodes the grounded semantic→subtype map), not an
  independent accuracy claim.

## Verdict (pre-registered success criteria)

**PARTIAL — an honest seam at subtype granularity.** Against the pre-registered bar:

- *Detect the loud subtypes at HIGH recall with honest abstain on the quiet ones* — **partial.**
  Detection is **stable** (no Stage-5 collapse; rate-free AUC 0.773 ± 0.040 vs the 3-way's recall sd
  0.30–0.43) and the abstain is **honest** (98% reject on the silent region), but recall is **low**
  (~11% @60% precision, rate-free): the single-mechanism detectability does not survive matched-rate
  mixtures, and the first headline (0.82) was partly a now-removed miss-rate artifact.
- *The prior adds calibrated resolution where the data is silent, and the data overrides where
  informative* — **yes on resolution + calibration** (reliable prior: coverage 0→1.0, Brier 0.085 beats
  prior-only 0.119, ECE 0.040), **no on override** (~0.06 rate-free): the data is too weak to override
  a wrong subtype prior — the layer is prior-dominated.
- *The dataset subtype-composition + abstain mass is calibrated* — **yes** (combined ECE ~0.04; the
  abstain mass is the property-3 "unable to determine" fraction, ~1.0 data-only and ~0 with a reliable
  prior).

So the layer is **buildable, calibrated, and stable**, and the P-arc machinery (tiered/gated prior,
raw-evidence combine, calibration, selective abstention, by-cell aggregation) transfers cleanly to
subtype granularity — but the per-column DATA channel is too weak at matched rate (rate-free AUC ~0.77)
to detect the loud subtypes at high recall or to override a wrong prior. The honest output is a
**prior-led** subtype-composition with a large, calibrated "unable to determine" mass when the prior is
absent — fully consistent with Stage 5 (the matched-rate non-identifiability wall, whose miss-rate
confound we re-encountered and removed here) and the P-arc (the prior's value is real,
reliability-contingent, and uncheckable on the non-identifiable subtypes).

## Follow-up — richer per-column features do NOT lift the deployable ceiling (NULL)

The open lead was that richer per-column distributional features might lift the ~0.82 detector
ceiling. `scripts/stageQ_richer_features_probe.py` tests this directly (one variable = the feature set;
RF loud-vs-reject ceiling; 5 seeds, matched rate), adding three theory-motivated families to the
base-5, each targeting a different way LOUD differs from REJECT:

| feature set | loud-vs-reject AUC | AP |
|---|---|---|
| base5 (current deployable) | **0.819 ± 0.020** | 0.421 |
| + shape9 (own-value: quantile-edge ratios + L-moments) | 0.789 ± 0.032 | 0.380 |
| + mar4 (richer MAR-axis: max / top-3 / frac-coupled / sd of per-other-column SMD) | 0.819 ± 0.034 | 0.429 |
| + disc1 (max adjacent-bin density jump — a discontinuity signature) | 0.825 ± 0.015 | 0.428 |
| + all | 0.822 ± 0.031 | 0.434 |

**Clean NULL.** No richer family lifts the ceiling. Specifically:
- **Own-value shape features (the sharp-cutoff/pileup signature) HURT** (−0.03 AUC, −0.04 AP): on real
  catalog columns the arbitrary base-distribution shape swamps the censoring edge, so quantile-edge /
  L-moment features add noise the RF splits on. The base-5 `robust_skew`/`excess_kurtosis` already
  extract the available own-value signal.
- **Richer MAR-axis coupling is flat** (0.819 → 0.819, *higher* variance). A 3-seed pilot showed a
  spurious +0.016; at 5 seeds the mean is identical and the per-seed lift is seed-dependent
  (+0.07 on one seed, −0.03 on others) — noise, not signal. The single `smd_to_others` already
  captures the loud-vs-MAR separation; more coupling statistics do not help.
- **The density-discontinuity feature is within noise** (+0.006 AUC, AP flat) — a density jump is not
  cleanly recoverable from a 20-bin histogram of ~75%-observed real-catalog values.

So no richer family lifts the ceiling. **Note:** this probe's `base5` baseline still *includes*
`missing_rate`, so its ~0.82 is the rate-contaminated upper bound (Result 1); the honest rate-free
ceiling is ~0.77, and since every richer family is flat-or-worse vs the contaminated baseline, none
lifts the rate-free one either — the conclusion is robust to the rate correction. The binding
constraint is loud-vs-self-censoring (a *sharp* vs *graded* truncation distinction, RF AUC ~0.73),
which none of own-value shape, MAR-axis coupling, or density-discontinuity features crack on real
catalog data — consistent with Stage 5's matched-rate non-identifiability bound. The data channel stays
weak, and the Stage-Q verdict is unchanged: the subtype layer is prior-led with calibrated abstention.
(The production detector uses the 4 rate-free deployable features — `missing_rate` excluded as a
non-transferable confound, Result 1; no richer family justified additional surface.)

## On using miss-rate as an *open* signal (the question that prompted the correction)

Would using per-column miss-rate as a signal — rather than excluding it — be statistically justifiable?
**Not for subtype detection.** It is justifiable only where the rate→subtype relationship is *real and
transferable*; for the loud subtypes it is neither, because a threshold/detection column's miss rate is
simply *where its cutoff sits* (a threshold at p5 misses 5%, at p50 misses 50% — same subtype). Any
rate-based detection therefore encodes the generator's (arbitrary) cutoff-placement distribution, not a
fact that transfers — precisely the Stage-5 confound, which the audit shows had already crept back in
(loud realises 0.197 vs reject 0.248; rate-only AUC 0.759). ADR-0007's "use rate openly, report when a
conclusion rests on it" is honoured by *reporting* the rate-included bound and excluding it from the
honest detector. If a specific deployment genuinely knows a rate→mechanism relationship, that belongs
in the **auditable prior channel** (like the metadata prior — separate, overridable, documented), not
baked into the data channel as a general capability. Real per-column rate variation that is
*independent* of mechanism (the realistic case) carries no subtype information by construction, so it
cannot lift the detector either.

## Caveats

- **Semi-synthetic; simulated prior reliability** (ρ injected, as P5/P6) — the real prior's subtype
  accuracy is a face-validity question (P1/P3), not measured here. The real spec is only checked for
  *consistency* against the grounded benchmark.
- **Small val pool** (7 source datasets → ~800 subsample/mixture instances): enough for the cal/test
  split and the 5-seed sd, but the dataset *diversity* is limited.
- **`kappa_like` is a fixed operating point** (20). It sets the prior:data evidence ratio; calibration
  absorbs the scale, but the override rate is sensitive to it (reported at the honest, non-inflated value).
- **Detector AUC ~0.82** is the RF ceiling on the 5 deployable features; richer per-column
  distributional / MAR-axis / discontinuity features were tested (see the follow-up section) and do
  **not** lift it — ~0.82 is the deployable ceiling at matched rate.

## Reproduce

```
python scripts/stageQ_subtype_layer.py --seeds 5            # the layer (likelihood + fusion + abstention)
python scripts/stageQ_feature_attribution.py --seeds 3      # the encoder-dilution attribution
python scripts/stageQ_richer_features_probe.py --seeds 5    # richer-feature ceiling probe (NULL)
```

## Files

- Ontology + generic ops: `lacuna/priors/{subtype_ontology,dirichlet_evidence}.py`.
- Prior + targets + detector: `lacuna/priors/subtype_prior.py`, `lacuna/data/subtype_targets.py`,
  `lacuna/models/subtype_likelihood.py` (+ tests under `tests/unit/{priors,data,models}/`).
- Scripts + reports: `scripts/stageQ_{subtype_layer,feature_attribution,richer_features_probe}.py`;
  `runs/stage0_general_baseline/stageQ_{subtype_layer,feature_attribution,richer_features_probe}.json`.
