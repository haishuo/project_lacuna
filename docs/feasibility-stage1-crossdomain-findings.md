# Stage-1 φ-spine + role-B projection + cross-domain curve — findings

*Implementation findings (PI greenlit Stage-1 + role-B in parallel, 2026-06-06). Honest results, not
spun. Governed by `MASTER-lacuna-survey-architecture.md`. Code: `lacuna/survey/{survey_catalog,
column_batching,column_phi,level1_model,level1_train,role_b_projection}.py`; scripts `run_level1_stage1_m1`,
`build_role_b_bases`, `run_cross_domain_curve`. 244 survey tests green.*

## 1. Stage-1 φ-spine — built, architecture validated, trainer bug found+fixed

- Built the full Level-1 column-primary spine (φ → δ-bin head) on the governed pipeline; no BERT encoder,
  no v1.0 heads; `named_prior` recorded; leakage-gated.
- **Trainer bug found + fixed:** the first M1 run inverted the spectrum (top_coding 0.57 < own_value 0.68).
  Root cause (isolated): `train_level1` generated **fresh examples every batch** (online) → underfits the
  sharp footprint → spectrum collapse. Fixed to a **fixed pre-generated corpus re-used across epochs**.
- **Architecture validated:** the bare Stage-0 probe still gives a crisp spectrum (φ top_coding 0.735 vs
  own_value 0.591); `Level1Model` under the fixed-corpus protocol on Stage-0's 8-dataset set reaches ~0.71.
  So the φ-spine carries the column-primary signal the BERT backbone loses (0.548).
- **M1 PARTIAL (honest):** on the genuine-5 training corpus the governed pipeline gives top_coding ~0.59 >
  own_value ~0.54 — spectrum DIRECTION holds and ≫ backbone, but the absolute level is **data-limited**
  (2-dataset test, 5 small training surveys, ±0.04 noise), well below the favorable Stage-0 0.735. Dropping
  the 3 continuous-rich contaminants (cars93/computers) removed datasets that had propped up 0.735. **Not
  an architecture failure — a data limitation.** Did NOT chase the metric (no bs/early-stop tuning).

## 2. Role-B projection — raw-source auto-curation fails; continuous survey targets are scarce

- **Raw NHANES/ESS auto-curation FAILED, honestly:** the raw files are dominated by survey-weight/design
  columns (NHANES `WTMREP1..52`, `SDMV*`; ESS `dweight`/`pspwght`/design) and **per-variable refuse codes**
  that automated heuristics cannot separate from valid values without codebooks (age 77 vs refuse 77;
  nominal country codes pass the continuous filter; NHANES demographics yielded **zero** clean targets).
- **Recorded finding (important):** clean **continuous** δ-targets are **scarce** in survey data — most
  items are **ordinal/Likert/binary** (NHANES DPQ depression 0–3, PISA Likert 1–5, DUQ yes/no). The
  top-coding / self-censoring idioms (which need continuous targets) apply to the **continuous minority**.
  This sharpens the survey-realism point: the *most common* survey missingness is item-nonresponse on
  ordinal items — a different idiom than continuous top-coding/self-censoring.
- **Pivoted to the v1.0 CURATED extracts** (`lacuna_survey/evaluation_data/*_real`, which already did the
  codebook work) → **4 CLEAN role-B bases** (spike 0.000; `projected_from_naturally_missing` flag; role-A
  preserved): `rb_nhanes_demographics` (demographics: poverty `INDFMPIR` card 338, age),
  `rb_gssvocab` (social: age/educ/vocab), `rb_nhanes_income` (income: `INDFMMPI`), `rb_nhanes_weight`
  (health: self-reported weights `WHD020/050/...` — social-desirability targets). **Small (277–478 rows).**

## 3. Cross-domain curve — FLAT at this scale (inconclusive)

Top-coding OOF AUC vs #domains, fixed labor test, 3 seeds:

| point | domains | datasets added | AUC |
|---|---|---|---|
| P1 | 1 (labor) | cps1988, psid1976, psid7682 | 0.626 ± 0.020 |
| P2 | 3 | + bfi (psych), yrbss (health) | 0.627 ± ? |
| P3 | 5 | + rb_nhanes_demographics, rb_gssvocab | 0.634 ± 0.017 |
| P4 | 7 | + rb_nhanes_income, rb_nhanes_weight | 0.636 ± 0.008 |

**Slope = +0.0017 AUC/domain; edge delta P4−P1 = +0.009 — FLAT (within ±0.017 seed noise).**

Honest reading (per cross-domain plan §3 + design-spec §9.3 caveat):
- Adding the 4 **small** role-B domains did **not** materially move labor-test transfer. This is **not** a
  positive scaling result — but **not** a clean "data-limited/architecture" verdict either: the added data
  is too small/thin to move the needle (≈4 domains × ~300–480 rows × ~1–5 continuous targets vs cps1988's
  28k rows). **Saturation on small/thin bases is not an architecture verdict** (§9.3).
- The test is fixed **labor** (top-coding on wage); the cross-domain benefit, if any, may show on a
  held-out **non-labor** domain (the leave-one-domain-out reading) rather than on labor — untested here
  because the role-B domains are too small for a low-noise held-out-domain test.

## 4. The binding constraint is DATA, and we now know its shape

The scaling hypothesis is **not yet answerable** with the on-disk assets, because:
1. The φ-spine works (≫ backbone) but at a modest, noisy ~0.62–0.64 level on the genuine corpus.
2. Role-B projection of on-disk assets yields only **small, thin** bases — clean continuous survey targets
   are scarce (most survey data is ordinal).
3. The cross-domain curve is therefore **flat/inconclusive** — too little added signal.

This is itself a result for the dissertation/grant framing: **to test whether domain diversity improves
the δ-prior, we need larger role-B bases with continuous targets** — either codebook-curated large NHANES
modules (income/weight at full cycle size, not 500-row extracts) or genuine acquisition (CPS/ACS income
with top-coding, NHANES labs are out-of-scope instrument-LOD). The data work is the bottleneck, and its
*shape* is now known: continuous-target curation, at scale, per source.

## 5. Recommended next (for PI — no auto-proceed)

Options, ranked:
1. **Codebook-curate larger continuous-target modules** (NHANES INQ income / WHQ weight at full cycle
   size via the raw `.xpt`, with per-variable sentinel handling) → bigger role-B bases → re-run the curve.
2. **Leave-one-domain-out** evaluation on the existing small bases (test transfer to a held-out domain,
   not labor) — cheap, but noisy given base sizes.
3. **Acquire** a large continuous-target survey (CPS/ACS income with real top-coding) — the cleanest test
   of scaling, and the grant-justifying move.
4. **Accept** the φ-spine result as the architecture validation and write up the data-bottleneck finding
   as the scaling result (honest negative/inconclusive at current scale).

No further runs without PI direction.
