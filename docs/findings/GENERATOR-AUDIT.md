# Generator Audit — Existing Family vs the Reformulation's Needs

*Charter §9 step 1 (`docs/proposals/GENERATOR-DESIGN-charter.md`). Audit of `lacuna/generators/`
(114 generator classes, ~7,800 LOC) against (a) the pyampute field standard and (b) the L1–L4
realism decomposition. Method: framework read (`base.py`, `semisynthetic.py`, `registry_builder.py`)
+ exhaustive per-module mechanism extraction. Pure inventory + gap analysis; no code changed.*

---

## 0. Headline findings

1. **PLASMODE-READY, UNIVERSALLY (major salvage win).** Every one of the 114 generators implements
   `apply_to(X, rng) -> R` — computes the mask from *provided real X*, not self-synthesized X. This
   is exactly the plasmode interface the reformulation needs; it already exists (the 2026-01-10
   fix). The `sample(rng, n, d)` path (synthesizes its own Gaussian X via `base_data.py`) is
   vestigial for our purposes — we use `apply_to` with real corpus X.
2. **We vastly SUBSUME pyampute.** pyampute = weighted-sum-score MAR (weights on observed) + MNAR
   (weight on own value): two mechanisms. We have 114, including exact supersets of both
   (`MARLogistic`/`MARMultiPredictor` ⊇ pyampute-MAR; `MNARLogistic`/`MNARSelfCensorHigh` ⊇
   pyampute-MNAR). Subsumption is trivial; continuity with the standard is free.
3. **δ IS parameterized — but only in one family.** The `mnar/self_censoring.py` family (12 classes)
   has explicit, graded value-dependence knobs (`beta1`, `beta2`, `beta_quadratic`, plus
   Weak/Moderate/Strong/ValueDependentStrength variants). This is the template for "δ as an explicit
   swept axis." The other MNAR families (censoring, detection, social, strategic) use **hard
   percentile thresholds with no smooth δ knob** — they are fixed idioms, not graded.
4. **L1 realism is the big gap, and it's already documented.** Most generators are parameterized for
   standard-normal X and **saturate on real scales** (the 2026-04-25 audit: 41/116 saturated on
   `wine`). A `_zscore_columns` mitigation exists in `semisynthetic.py` and the `mar/realistic.py`
   family — but it is not universal, and *no generator's output has been tested against real
   missingness footprints* (no C2ST, no fidelity panel). This is the central re-objectiving target.
5. **Rate is not a unified controlled axis.** Each family controls rate idiosyncratically
   (`miss_rate`, `miss_prob`, `alpha0` intercept, percentile). There is no single "target overall
   rate = 0.3" knob across the family — matched-rate is currently enforced *outside* the generators
   (retry-on-draw in the showdown harness). The reformulation needs rate as an explicit, uniform,
   controlled axis.

## 1. Inventory (114 classes; all `apply_to`-capable)

| family | n | mechanism core | δ-graded? | L-layer home |
|---|---|---|---|---|
| **MCAR** bernoulli/blocks/column/row/conditional/distributional/multilevel | 36 | rate-only: Bernoulli, blocks, per-row/col/group rates, heavy-tailed rate draws | n/a (no value dep) | L1 (footprint) + the R0 ignorable control |
| **MAR** simple/multiple/complex/predictor_types/structural/survey | 30 | `sigmoid(α₀+Σαₖ·X_observedₖ)` + nonlinear (poly/spline/kernel/tree/distance) + block/skip/branch structure | n/a (depends on observed) | L2 (conditional structure) — the MAR competitor class |
| **MAR** realistic (single/partial-response/demographic-gated) | 3 | z-scored predictor, target-rate-parameterized, correlated within-row drop | n/a | **L2 + L1** (only family built for real-scale realism) |
| **MAR** strength (weak/moderate/strong) | 3 | logistic with αₖ drawn from strength bands | n/a | L2 strength sweep |
| **MNAR** self_censoring | 12 | `sigmoid(β₀+β₂·X_own)` — value-dependent, **β as explicit δ knob**; Weak/Strong/regime variants | **YES (graded)** | **L4 (the δ axis) — the core for the reformulation** |
| **MNAR** censoring/detection | 12 | hard percentile threshold on own value (left/right/two-sided/LOD); one `SoftThreshold` (steepness knob) | mostly NO (hard) | L4 fixed idioms (top-coding/LOD) |
| **MNAR** social/strategic | 6 | own-value threshold (under/over-report, gaming, privacy, competitive) | NO (hard) | L4 fixed idioms (survey social-desirability) |
| **MNAR** informative/selection | 9 | inverse-MNAR (abnormal→observed), Heckman selection, attrition, competing events | partial (some knobs) | L4 + structural |
| **MNAR** latent | 6 | unobserved Z drives both X and R; **`apply_to` approximates Z as row-mean** | implicit | L4 — honesty caveat (see §4) |

## 2. pyampute subsumption map

| pyampute mechanism | our exact superset | continuity action |
|---|---|---|
| MCAR (uniform) | `MCARBernoulli` | register as baseline |
| MAR (weighted sum on observed) | `MARMultiPredictor` / `MARWeightedPredictor` | register as baseline; our nonlinear MAR extends it |
| MNAR (weight includes own value) | `MNARLogistic` (β₂) / `MNARSelfCensorHigh` (β₁) | register as baseline; our graded-δ extends it |

Action: expose a thin `pyampute_compat` config that instantiates exactly these three as the
labeled baseline vocabulary — provably ≥ the field standard, and reviewers see the lineage.

## 3. L1–L4 status against the charter's validation protocol

- **L1 footprint realism — NOT VALIDATED (the gap).** No generator output has faced a C2ST or
  fidelity panel vs real masks. Known saturation on real scales; z-score fix only partial. **This
  is the first build target** (the realism-gate harness).
- **L2 conditional structure — PRESENT IN FORM, NOT CALIBRATED.** The MAR family produces
  conditional-on-observed missingness (the right *form*), and `mar/realistic.py` is closest to real
  structure — but parameters are hand-chosen, **not fit to ESS/NHANES/GSS**. Re-objectiving =
  fit L2 to the corpora (Radosavljević method) + accept via the L2 gate.
- **L3 span — BROAD (likely sufficient), UNTESTED vs real.** 114 mechanisms is a wide vocabulary;
  span against *real* missingness patterns (does real fall inside the manifold?) is unmeasured.
- **L4 residual δ — PARAMETERIZED IN ONE FAMILY.** `self_censoring` gives the graded δ knob the
  reformulation sweeps. Hard-threshold idioms (top-coding/LOD/social) are fixed-δ — keep as named
  idioms, but add graded variants where a sweep is needed. δ is currently *not* tied to the
  literature anchors (Jabkowski etc.) — that wiring is new.

## 4. Risks / honesty flags found in the code

1. **Latent family `apply_to` approximates Z as the row mean** — the unobserved confounder can't be
   reconstructed from X (that's the point of MNAR), so this is a known approximation. Flag it; do
   not treat latent-family plasmode masks as faithful to a true latent process.
2. **Saturation on real scales** (documented 2026-04-25) — must be fixed universally before any L1
   gate can pass; z-score-the-predictor is the existing mitigation, not yet applied family-wide.
3. **Rate not matched uniformly** — matched-0.3 currently lives in harness retry logic, not the
   generators; certification needs rate as a first-class controlled axis.
4. **`sample()` self-synthesizes Gaussian X** — irrelevant to plasmode but a foot-gun if a future
   caller uses `sample` instead of `apply_to` and silently trains on synthetic-X footprints.

## 5. Re-objectiving worklist (feeds charter §9 steps 2–4)

1. **Realism-gate harness** (charter §6 / §9.2): C2ST + conformal C2ST + fidelity panel (Dankar
   axes) + authenticity (Alaa), run `apply_to` output vs ESS/NHANES/GSS masks, block-aware. *First
   code.* Reuse/extend `analysis/generator_validation.py`.
2. **Universal anti-saturation**: lift `_zscore_columns` from a per-family fix to a generator-base
   guarantee; re-run the saturation audit to confirm 0/114 saturate on the real corpora.
3. **Uniform rate axis**: add a `target_rate` calibration wrapper so every generator hits a
   specified overall rate on given X (bisection on the intercept/threshold), making matched-rate a
   generator property not a harness hack.
4. **δ as explicit swept axis**: standardize the `self_censoring` β-knob interface; add graded
   variants of the top-coding/LOD/social idioms where a sweep (not a fixed threshold) is needed;
   wire the δ-grid to the literature anchors at certification time (not inside the generator).
5. **L2 calibration from real data**: fit the conditional/block structure (co-missingness,
   inter-variable correlation) to the corpora per Radosavljević; gate via L2.
6. **pyampute_compat baseline config** (§2).

## 6. Bottom line

The generator layer is **far more salvageable than the charter assumed**: the plasmode interface is
universal, the vocabulary subsumes the field standard, and graded-δ exists where it matters most.
What's missing is exactly the reformulation's load-bearing addition — *measured realism*: no
generator output has been tested against real missingness, saturation is a known unfixed hazard, and
rate/δ are not yet clean controlled axes. The work is re-objectiving (validate + calibrate + expose
axes), not greenfield. Next concrete build: the realism-gate harness (step 1 above).
