# Cross-Domain Corpus Plan — exploit on-disk survey assets, then scale by domain (no code)

*Plan + spec only. **No code, no experiments, no acquisition.** Status: **for PI approval.** Operationalizes
the PI's 5-point sequence (2026-06-06) after `DATA-ROLE-B-PROJECTION-feasibility.md`. Preserves the
role-A/role-B discipline (design-spec §9.0). Reframes the learning curve from **file-count** to
**genuinely-distinct domains** (PI: correlated slices are not independent evidence).*

## 0.5 Framing updates folded (2026-06-06 — `MASTER-lacuna-survey-architecture.md` §12 governs)

- **`lod_top_coding → top_coding`** (survey-realistic top-coding, **not** assay-LOD). Idiom assignment:
  **income = top-coding**; **weight = social-desirability self-censoring** (smooth).
- **NHANES projection = questionnaire + demographic items only; EXCLUDE lab/exam assays** (instrument-LOD
  is out of Lacuna-Survey scope).
- **Survey-idiom vocabulary only** (item nonresponse, skip logic, top-coding/bracketing, social-
  desirability, attrition, DK-vs-refuse).
- The cross-domain curve reports **detectability comparison-class-relative** (named class) and the
  **"lab-coat fraction"** (data-alone vs metadata, oracle-gated) as headline; **UNKNOWN** (off-manifold)
  is a first-class label; the **MCAR-vs-not gate (Stage A)** precedes the δ-prior (master §5).

## 1. Confirmed sequence (plan of record)

1. **Build the Level-1 φ-spine** (Stage-1 spec; still pending its approval + 4 sub-decisions).
2. **Project on-disk role-A assets → role-B bases** — NHANES demographics, selected NHANES modules, ESS
   (PISA optional/deferred). *We impose the holes; we keep ground truth.*
3. **Run the cross-DOMAIN learning curve** — labor → +demographics → +health → +attitudes (→ +education).
4. **Decide acquisition from the slope** — positive ⇒ acquisition justified (grant result); flat ⇒ not yet.
5. **Revisit broader acquisition / synthetic generation only then.**

Dissertation question (binding): **"does performance improve as the diversity of survey *worlds* (domains)
increases?"** — not "is it deployable?" A positive cross-domain slope is itself the scientific result.

## 2. Role-B projection task (spec — the new data-engineering unit)

**One job:** turn an on-disk **role-A** survey source into a **role-B** complete-case base (we impose the
δ-holes), **without discarding the role-A archive.** Per CLAUDE.md (one-job modules, ≤500 LOC, fail-loud,
determinism, tests).

**Per-source pipeline:**
1. **Recode refusal/admin sentinels → NaN** (e.g. ESS 77/88/99/777/888/999; NHANES 7/9/777/999 per
   codebook). *Mandatory before projection* — sentinels look "observed," corrupt the value distribution,
   and inflate cardinality (feasibility §4: ESS 90/399 complete cols carry them).
2. **Complete-case projection:** select the column set with observed-rate ≥ τ_col (post-recode), keep rows
   complete over it. Record (τ_col, cols kept, rows retained).
3. **Continuous-target filter:** mark targetable columns as those with cardinality ≥ 30 post-recode (the δ
   idioms need continuous targets; card 10–29 = likely coded ordinal, kept as predictors not targets).
4. **Preserve role A:** archive the original natural-missingness table (role A) untouched alongside the
   role-B projection — for manifold/OOD/face-validity (§9.0). **Two outputs, one source.**
5. **Provenance manifest** per base: `{source, cycle, domain, source_block, sentinels_recoded,
   tau_col, cols_kept, rows_retained, n_continuous_targets, survey_weights: present-but-unused,
   projection: "complete_case_from_naturally_missing"}`.

**Honesty flag (binding):** projected bases are tagged **`projected_from_naturally_missing`** and kept
**distinct** from native-complete `survey_*` bases — because complete-case projection is a mechanism-laden
*selection* (feasibility §4.1): the role-B base is the complete-case subpopulation, not the population. The
flag travels into the named-prior manifest so any result is auditable for this bias.

**Initial targets (cheap, on-disk):**
| base | domain | source_block | notes |
|---|---|---|---|
| `nhanes_demographics` | demographics | NHANES-2017–18 | ~3 continuous targets, ~126k rows |
| `nhanes_inq_income` | health/income | NHANES-2017–18 | income → ideal for **top-coding** idiom |
| `nhanes_whq_weight` | health | NHANES-2017–18 | weight → ideal for **social-desirability self-censoring** |
| `nhanes_dpq_phq9` | health | NHANES-2017–18 | depression (ordinal-heavy; predictors) |
| `ess_pooled` | attitudes | ESS-R11 | pooled (1 world); ~38 continuous targets post-recode |
| *(PISA 2018/2022)* | education | PISA | **deferred** — multi-GB SAS extraction |

## 3. Cross-domain learning-curve design (spec — replaces the file-count curve)

**X-axis = cumulative DOMAINS, not file count.** The curve adds *domains*, and correlated slices within a
domain/source are **one block**, not independent points.

- **Curve:** train on cumulative domain sets — `{labor}` → `{labor, demographics}` → `{+health}` →
  `{+attitudes}` (→ `{+education}`) — held-out test fixed.
- **Block-aware, leave-DOMAIN-out split (binding):** all NHANES-2017–18-derived bases = **one block** (same
  respondents → leakage if split across train/test); ESS = one block. The split holds out **whole
  domains/blocks**, never same-respondent slices. This is *both* the leakage guard *and* the actual
  cross-domain transportability test (§5/§8).
- **Two complementary readings:**
  1. **Add-domain-improves-OOF:** does adding a new domain to training improve held-out calibration/sharpness
     on the *other* domains? (the scaling slope — the grant result).
  2. **Leave-one-domain-out transfer:** train on all-but-one domain, test on the held-out domain — the
     §5 out-of-family transportability number (the headline calibration metric).
- **Report per domain-set, with variance:** δ-prior calibration/coverage; top-coding sharpness;
  own-value prior-dominated behavior; seed/split variance. *(Detectability-vs-oracle and OOD curves arrive
  with Stages 2–3.)*
- **Interpretation discipline:** the slope is read over **domains**; a rise from adding correlated slices
  *within* a block is **not** counted as scaling evidence. Saturation is judged only after genuinely new
  *domains* are added (design-spec §9.3 caveat).

## 4. Gating & dependencies

- Step 1 (build φ-spine) needs the **Stage-1 spec + 4 sub-decisions** approved (still pending).
- Step 2 (role-B projection) needs **this §2 spec** approved (a separate data-engineering deliverable;
  its own modules + tests; does **not** touch the model).
- Step 3 (curve) needs **both** done. Steps 1 and 2 are **independent** and can be specced/built in
  parallel once approved.

## 5. Open decisions for the PI

1. **NHANES modules to project** — proposal: `demographics`, `inq_income` (top-coding), `whq_weight` (social-desirability)
   first (continuous targets); `dpq_phq9`/`duq_drug` as predictor-rich health context. PISA **deferred**.
2. **ESS unit** — **pooled (1 world)** [proposal, avoids country-slice inflation] vs a few large single-
   country tables (a correlated block). 
3. **Held-out design** — **leave-one-domain-out** [proposal, the real transportability test] vs a fixed
   diverse held-out set.
4. **Catalog placement** — projected bases in a **separate, flagged registry** (`projected_*` / a
   `role_b_projected` tag) [proposal] vs mixed into the existing catalog. (They must stay distinguishable
   from native-complete, §2.)
5. **Sentinel/threshold parameters** — `τ_col` (proposal 0.95), continuous-target card ≥ 30, per-source
   sentinel code lists (from codebooks). Confirm.
6. **Sequencing** — spec & build the **role-B projection (step 2) in parallel** with the Stage-1 φ-spine
   (step 1) [proposal], so the corpus is ready when the spine is.

**No code until the Stage-1 spec, this projection spec, and these decisions are approved.**

## 6. Assessment of the PI's plan (requested)

**Endorsed.** It is the scientifically correct order: exploit the real assets we already have (which add
*domains*, the diversity that matters), test the cross-domain scaling relationship cheaply, and let the
slope — not a deployment standard — justify any acquisition. The one place to hold the line is the
**counting discipline**: NHANES modules and ESS countries are *blocks*, not independent worlds, both for
the split (leakage) and for reading the curve (don't credit correlated slices as scaling evidence). With
that guard, "we have not yet exhausted the real survey data we already possess" is **correct**, and the
cross-domain curve is the right next scientific result.
