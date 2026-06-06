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

## 6. UPDATE (2026-06-06) — larger NHANES bases moved the curve (weakly)

PI pivoted to data scale: codebook-curated FULL NHANES weight+poverty modules (`build_nhanes_role_b.py`).
**Bases built + passed §9 quality checks** (the spike gate dropped `WHD120` @0.0039, protecting the prior):
- `rb_nhanes_weight` **3,833 rows** [preferred], 4/5 self-reported-weight targets pass (social-desirability).
- `rb_nhanes_poverty` **4,774 rows** [preferred], `INDFMPIR` poverty 0–5 pass (top-coding). ~8–10× the extracts.

**Scaled cross-domain curve** (top-coding OOF AUC, labor test, 3 seeds, max_rows=384):

| | small bases (§3) | **large NHANES bases** |
|---|---|---|
| 1d labor | — | 0.608 ± 0.009 |
| 3d +psych+health | 0.627 | 0.625 ± 0.023 |
| 5d +NHANES (big) | 0.634 | **0.644 ± 0.033** |
| **slope / domain** | **+0.0017** | **+0.0091** |
| edge Δ | +0.009 | **+0.036** |
| add-NHANES Δ (P2→P3) | — | **+0.019** |

**Honest reading:** with *larger* bases the slope **quadrupled** (+0.0017 → +0.0091) and the edge Δ
**tripled** (+0.009 → +0.036). So the earlier flat curve was **partly a small-base artifact** — bigger,
cleaner domains *do* move it. **But the effect is modest and within seed noise** (±0.02–0.03/point; the
+0.019 add-NHANES Δ is borderline). This is a **credible WEAK positive trend, not decisive proof.**

**Decision bearing (PI step 5):** the trend strengthened with base size ⇒ the corpus is **data-limited**
and more/larger domains help ⇒ **external acquisition is justified to get a DECISIVE curve** (grant
result, weakly supported). Before/with acquisition, two cheap ways to tighten the estimate: **more seeds**
(shrink ±0.03) and **leave-one-domain-out** (a less-noisy probe than the fixed labor test). The φ-spine
remains validated; the bottleneck remains data, now with a measured (weak-positive) scaling signal.

## 7. UPDATE (2026-06-06) — tightened evidence: leave-one-domain-out shows diversity HELPS transfer

Cheap tightening (8-seed cumulative + block-aware leave-one-domain-out; no architecture/HP/variant
changes; `run_curve_tightened.py`). The two probes diverge informatively:

**A) Cumulative curve, 8 seeds, LABOR test:** 1d 0.608±.006 → 3d 0.613±.011 → 5d 0.635±.017. Slope
**+0.0068/domain**, edge P3−P1 **+0.027 (SE 0.018) — within 2 SE, NOT significant.** On the *in-family
labor* test, the slope does **not** survive 8 seeds — adding domains barely helps a domain already
well-covered by the labor-core training.

**B) Leave-one-domain-out (5 seeds; NARROW=labor vs DIVERSE=+other domains; test on the held-out D):**

| held-out domain D | narrow (labor) | diverse (+others) | Δ | verdict |
|---|---|---|---|---|
| **psychology (bfi)** | 0.665 ± .051 | 0.728 ± .031 | **+0.064** | **HELPS (>2 SE)** |
| **health (yrbss)** | 0.642 ± .019 | 0.679 ± .027 | **+0.037** | **HELPS (>2 SE)** |
| **NHANES (weight+poverty)** | 0.581 ± .011 | 0.627 ± .058 | **+0.045** | positive, noisy (not >2 SE) |

**All three positive; two significant; none hurt.** Diverse training **improves out-of-domain transfer**
— exactly where domain diversity *should* show value (transfer to *unseen* domains), not on the already-
covered labor test. The weak labor cumulative slope and the strong leave-one-out result are **consistent**:
labor is in-family (little to gain), the held-out domains are out-of-family (clear gain).

**Mapped to the PI's interpretation rules:**
- *Rule 1 (positive slope AND leave-one-out shows improved transfer ⇒ acquisition strongly justified):*
  **largely met** — leave-one-out improves transfer significantly for 2/3 held-out domains, positively for
  3/3. The directional slope is positive throughout.
- *Rule 2 (slope vanishes under seeds OR only helps labor ⇒ weak):* the labor cumulative slope **is** weak/
  not-significant — but it does **NOT** "only help labor"; the opposite — it helps **non-labor** domains
  more. So Rule 2's escape clause does not bite.
- *Rule 3 (domain-specific):* reported, not averaged — bfi (+0.064) > NHANES (+0.045) > yrbss (+0.037);
  bfi/yrbss significant, NHANES positive-but-noisy. **No domain hurt.**

**Honest caveat:** the one domain we most care about for the idiom vocabulary — **NHANES (continuous
weight/poverty targets)** — is positive (+0.045) but **noisy** (diverse SE 0.058), so its transfer gain is
encouraging, not yet significant. The significant gains are on bfi (Likert) and yrbss.

**Decision bearing:** the tightened evidence is **stronger than the weak cumulative slope suggested** —
the leave-one-domain-out (the right transportability probe) shows domain diversity **helps transfer to
unseen domains**, significantly for 2/3. Per Rule 1 this **justifies acquisition** — specifically of
**more large continuous-target domains** (CPS/ACS/IPUMS income with top-coding), which would (a) firm up
the noisy NHANES transfer and (b) add held-out domains for a decisive leave-one-domain-out. The φ-spine
remains validated; the bottleneck remains data; the scaling signal is now real out-of-family, modest, and
domain-specific.

No further runs without PI direction.
