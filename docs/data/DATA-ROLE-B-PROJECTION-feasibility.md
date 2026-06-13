# Role-B Projection Feasibility — how many training worlds from assets we already have?

*Read-only analysis. **No code, no acquisition, no implementation.** Status: **for PI review.** Answers
"how many role-B (complete-case, we-impose-the-holes) training worlds can we derive from survey assets
already on disk?" — before deciding on new acquisition. Measured 2026-06-06 (complete-case projection at
column-observed-rate thresholds; cardinality + refusal-sentinel profiling). Companion to
`DATA-INVENTORY-ground-truth.md`. Preserves the role-A/role-B discipline (design-spec §9.0).*

## 0. Verdict up front

Existing on-disk assets can realistically take the role-B corpus from **9 genuine worlds → ~13–15
genuinely/weakly-distinct worlds across ~6–8 domains** *without any new downloads* — the **real win is
domain coverage** (adds demographics, health, education, social attitudes), not raw count. A naive count
that slices NHANES into modules and ESS into countries could claim "30–50 datasets," but those slices are
**same-respondent / same-questionnaire correlated** — the *same redundancy trap* as the labor-econ cluster
(§4). So: **exploit what we have first, run the cross-domain learning curve, then decide acquisition** —
which is exactly the PI's plan (§5). The corpus does *not* trivially jump to "30–50 independent worlds."

## 1. Role-B projection feasibility table (measured)

| source | project to role-B? | targetable cols (card≥10 / **continuous** card≥30) | rows retained | domain (new?) | honest independent worlds |
|---|---|---|---|---|---|
| **9 genuine `survey_*`** (already B) | already role B | 65 total | 82–28,155 | labor×5, psych, political, finance, health | 9 files → **~3–4 effective** (labor-redundant) |
| **NHANES demographics_clean** (135k×280, 80% NA) | **YES** — 15 cols @95% obs, 126k complete rows | 5 / **3** | ~110k–126k | **demographics (NEW)** | **1** |
| **NHANES questionnaire (pooled)** (134k×1444, 94% NA) | **NO** — only 3 cols observed for all → collapses | 2 / 2 | 40k | health | — (pooled is not a world) |
| **NHANES questionnaire (per-module: DPQ/DUQ/INQ/WHQ/diet…)** | **YES, but per-module extraction** (keep module-responders) | ~2–6 each / **1–3** | ~500–5k each | **health (NEW-expand)** | ~4–8 module-worlds — **same respondents ⇒ ONE held-out block** |
| **ESS R11** (50k×667, 36% NA pooled) | **YES** — **396 cols 100%-observed**, 38k complete rows (*after sentinel recode*) | 137 / **38** (90 cols carry 77/88/99 refusal codes → recode first) | ~38k | **social attitudes (NEW)** | **1 pooled** (or ~30 country = correlated) |
| **PISA 2018 / 2022** (raw multi-GB SAS on disk) | **YES but needs SAS extraction** (only 500-row eval extracts exist now) | tens / ? | up to ~600k students | **education (NEW)** | **2** (cohort/year) |
| **GSS** | **NO on disk** beyond a 500-row `gssvocab` extract — full GSS = acquisition | 4 / — | 500 | attitudes | needs download |

*Method: per-column observed rate on the full/ sampled table; "complete rows" = rows complete over the
chosen column set, scaled to full n; "targetable" = cardinality ≥ 10; "continuous" = card ≥ 30 (a stricter
proxy for genuine continuous targets, since the δ idioms need continuous values).*

## 2. Re-estimated corpus size & diversity (honest counting)

- **Genuinely-INDEPENDENT new worlds** (counting one survey/cohort as one block, *not* slicing):
  NHANES-2017–18 (**1** block: demographics + health modules) + ESS R11 (**1**) + PISA 2018/2022 (**2**,
  need extraction) ≈ **+3–4 independent worlds**, adding **4 new domains** (demographics, health, education,
  attitudes).
- **From 9 → ~13–15** genuinely/weakly-distinct training worlds across **~6–8 domains**, *no downloads*
  (NHANES is cheap; ESS is cheap-after-recode; PISA is medium-effort SAS extraction).
- **The misleading large number:** slicing NHANES into ~8 modules + ESS into ~30 countries would *report*
  "30–50 datasets," but those are within-block correlated (§4) — inflated diversity, not independent worlds.

**Diversity, the metric that matters:** the 9 genuine cover ~5 named domains but are **labor-dominated**
(5 of 9 → ~3–4 effective). The existing assets' value is that they **break the labor monoculture** —
demographics, health, education, attitudes — which is exactly the axis the cross-domain learning curve
needs to test. **Domain coverage roughly doubles; independent-world count grows modestly.**

## 3. Assessment of the PI's 6-point plan

**I agree with all six**, and the inventory supports them:
1. **Preserve role-A/role-B** — correct; it is the source of ground-truth integrity (§9.0).
2. **Do not train on natural missingness** — correct; complete-case projection (we impose the holes) keeps
   the answer sheet.
3. **Inventory role-B-derivable from on-disk assets first** — this document; the answer is "meaningful new
   *domain* coverage, modest new *world* count."
4. **Re-estimate corpus after projection** — done (§2): 9 → ~13–15 worlds, ~6–8 domains.
5. **Decide acquisition only then** — correct; and the learning curve over the *expanded-by-projection*
   corpus is the deciding evidence (does adding NHANES/ESS/PISA domains improve OOF?).
6. **Synthetic generation later** — correct; exploit real assets first.

The plan is sound and dissertation-appropriate: it tests a **scaling relationship** (does new *domain*
diversity improve OOF?) cheaply, before spending on acquisition.

## 4. Hidden assumptions I'd flag (you asked)

1. **Complete-case projection is a mechanism-laden SELECTION, not a neutral cleanup.** Dropping rows/cols to
   get complete X yields the *complete-case subpopulation*, which differs from the true survey population
   *because of* the missingness we study. Role-B-from-naturally-missing ≠ role-B-from-truly-complete: the
   X-manifold we train on is the **complete-case manifold** (esp. severe for per-module extraction, where we
   keep only module-responders). Tolerable for semi-synthetic supervision (we just need realistic survey-
   like X), but the M2/M3 manifold claims then concern complete-case X, not the population. **Name it.**
2. **Module/country slicing inflates diversity with correlated data.** NHANES modules share respondents;
   ESS countries share the questionnaire. They are **not** independent worlds. Counting them as such repeats
   the labor-redundancy error. **Count independent worlds/domains, not files.**
3. **Same-respondent slices must be ONE held-out block.** If NHANES-demographics is in train and
   NHANES-income is the held-out test, they share *people* → leakage in leave-datasets-out. All
   NHANES-2017–18-derived worlds (and all ESS countries) must move together across the train/test split.
4. **`targetable` (card≥10) ≠ usable-δ-target.** The δ idioms (LOD, top-coding, smooth self-censoring) need
   **continuous** targets. ESS shows 137 "targetable" but only **38 continuous (card≥30)**, and **90 columns
   carry refusal sentinels (77/88/99)** that look "observed," corrupt the value distribution, and inflate
   cardinality — **they must be recoded to NaN before projection.** Usable-target growth < column-count
   growth.
5. **The learning curve tests *new-domain* generalization, not file count.** Its scientific value comes from
   adding *independent domains*; correlated slices won't move OOF in an interpretable way. So the assets help
   chiefly by **adding domains** (high value), not by padding the dataset count.
6. **"Exploit what we have" ≠ all-free.** NHANES (CSV on disk) and ESS (CSV, after recode) are cheap;
   **PISA requires multi-GB SAS extraction**; **full GSS is not on disk** (only a 500-row vocab extract) — so
   "GSS as a role-B world" is actually acquisition, not exploitation.
7. **The `lacuna_survey/evaluation_data/*_real` anchors are v1.0-paradigm.** They were built for the old
   consensus MCAR/MAR/MNAR calibration, not δ. Reusing them as **role-A validation** (manifold/OOD/face-
   validity) for the δ-prior is fine; their **consensus labels are not δ ground truth** — don't conflate.
8. **Survey weights.** Surveys are weighted samples; ignoring weights makes even a "complete" projection the
   *unweighted-respondent* manifold — a further (usually minor) distortion from the population. Minor, but it
   exists.

## 5. What this implies for sequencing (assessment only — no recommendation to act)

The evidence says the **cheap, high-value first move** is to project the **already-on-disk** NHANES
(demographics + a few health modules) and **ESS** (post-recode) into role-B bases, add them to the corpus,
and run the **cross-domain learning curve** (does OOF improve as we go from labor-only → +demographics
→ +health → +attitudes?). That curve — not a deployment-readiness check — is the dissertation/grant result
(design-spec §9.A). PISA is a medium-effort extension; GSS-beyond-vocab and anything else is genuine
acquisition, to be decided *after* the curve. **All of this is deferred pending your decision; this
document is feasibility analysis only.**
