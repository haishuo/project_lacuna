# NHANES Full-Module Role-B Projection — spec (no code/runs until approved)

*Projection spec only. **No model runs; bases produced only after this spec is approved and they pass the
§9 quality checks.** Status: **for PI approval.** Codebook-aware role-B projection of the FULL on-disk
NHANES 2017–18 income/weight modules (not the 500-row curated extracts), to test whether the flat
cross-domain curve was caused by small/thin bases. Governed by `MASTER-lacuna-survey-architecture.md`
(§6 survey idioms, §7 data roles) and `cross-domain-corpus-plan.md`. Grounded in a read-only inspection
of the actual `.xpt` files (variable labels, ranges, refuse-code frequencies).*

## 1. Source files / modules (on disk; preserved in place)
`/mnt/data/lacuna/incoming/` — NHANES 2017–18 ("_J" cycle), read via `pyreadstat.read_xport`:
- **`DEMO_J.xpt`** — 9,254 respondents × 46 cols (demographics + the poverty ratio).
- **`WHQ_J.xpt`** — weight-history module (self-reported weights/heights).
- **`INQ_J.xpt`** — income module (income brackets + monthly poverty index).
Merged on **`SEQN`** (respondent id). Same respondents across all three ⇒ **one block** (§10).

## 2. Target variables (continuous; survey-realistic idioms)
| base | continuous target(s) | idiom (MASTER §6) | predictors (context) |
|---|---|---|---|
| **`rb_nhanes_weight`** | `WHD020` current wt, `WHD050` wt 1yr ago, `WHD110` wt 10yr ago, `WHD120` wt age 25, `WHD140` greatest wt (lbs) | **social-desirability self-censoring** (weight underreporting) | `RIDAGEYR` age, `RIAGENDR` gender, `WHD010` height, `DMDEDUC2` education |
| **`rb_nhanes_poverty`** | `INDFMPIR` family-income-to-poverty ratio (0–5; naturally **top-coded at 5**) | **top-coding** | `RIDAGEYR`, `RIAGENDR`, `DMDEDUC2`, `DMDFMSIZ` family size |

Income brackets (`INDHHIN2`/`INDFMIN2`, values 1–15) are **categorical top-coded** (top bracket = $100k+),
kept as **ordinal predictors only**, never continuous targets. `INDFMPIR` is the continuous income target.

## 3. Codebook / sentinel handling (PER-VARIABLE; the core of "codebook-aware")
Refuse/DK codes are **per-variable** and must NOT be applied blindly (age 7/9/77 are valid ages — the raw
inspection false-flagged them). Rules:
- **Self-reported weights/heights** (`WHD0*`, `WHD1*`): NHANES uses **`7777`=Refused, `9999`=Don't know**.
  Recode `{7777, 9999}` → NaN. (Plausible range clip §9: lbs ∈ [50, 700]; inches ∈ [40, 90].)
- **Age-at-event** (`WHQ150`, `WHQ200`): `77777`/`99999` → NaN.
- **Poverty ratio** `INDFMPIR`: range [0, 5], **no refuse code** (NCHS imputes); keep as-is.
- **Ordinal predictors** (income brackets, education): refuse `77`/`99`, DK `7`/`9` **only where the valid
  code set excludes them** (e.g. income bracket 1–15 ⇒ recode `{77,99}`; education 1–5 ⇒ recode `{7,9}`).
- **Continuous age** `RIDAGEYR` (0–80): **no recode** (valid integers; 7/9/77 are real ages).
The per-variable sentinel map is **explicit in the build config** and **recorded in provenance** (§8).

## 4. ID / design / survey-weight exclusions (dropped before projection)
`SEQN` (used for the merge, then dropped), `SDDSRVYR`, `RIDSTATR`, `WTINT2YR`, `WTMEC2YR`, `SDMVPSU`,
`SDMVSTRA`, and all `WT*`/`SDMV*` columns. Survey weights/PSU/strata are **design** variables, never
substantive targets or predictors.

## 5. Natural-missingness preservation (role A — binding, §7)
The merged module table (DEMO⋈WHQ⋈INQ on SEQN), **with natural missingness AND recoded sentinels left as
NaN**, is archived **unmodified** as the **role-A** artifact (`/mnt/data/lacuna/role_a/nhanes_2017_18_merged.csv`
+ a mask). It is **never** used for supervised δ-training — only for manifold/OOD/detectability validation
(§9.0). The complete-case projection (§6) is a **separate** role-B output. **Two artifacts, one source.**

## 6. Role-B complete-case projection rules
1. Merge the three modules on `SEQN` (left-join on DEMO, the universe of respondents).
2. Apply the §3 per-variable sentinel recode and §4 exclusions.
3. **Select** the target + predictor columns for each base (§2) — *not* all columns (avoids re-introducing
   sparse/ordinal noise).
4. **Complete-case** over the selected column set: keep rows observed on all selected columns
   (`τ_col` is implicit — we keep the chosen columns and drop incomplete rows).
5. Standardize is **not** done here (φ standardizes within observed at batch time).

## 7. Minimum rows / columns thresholds (else flag FRAGILE, do not use)
A role-B base is **usable** only if, after projection: **rows ≥ 1500**, **≥ 1 continuous target**
(card ≥ 30, plausible-range), **≥ 2 predictors**. Below these ⇒ tag `fragile=true` in provenance and
**exclude** from training (a fragile base is the trigger for external acquisition, §11). Expectation
(from the 9,254-respondent modules): weight base ~6–8k complete rows; poverty base ~8k+ rows.

## 8. Provenance manifest (per base; auditable)
`{source_modules, merge_key: SEQN, base_name, domain, source_block: "NHANES-2017-18", flag:
projected_from_naturally_missing, target_vars, predictor_vars, sentinel_map (per-variable codes recoded),
design_cols_excluded, rows_retained, cols_kept, n_continuous_targets, plausible_ranges, spike_check (per
target), role_a_archive_path, fragile}`. Written alongside the base CSV; indexed in `role_b/INDEX.json`.

## 9. Sanity checks (must pass before any model use)
Per continuous target, after recode:
- **Sentinel-spike check:** fraction of values at any NHANES refuse code (`7777/9999/77/99/...`) ≤ **0.002**.
- **Plausible-range check:** all values within the documented range (weights [50,700] lbs; heights
  [40,90] in; poverty [0,5]; age [0,100]). Any out-of-range value after recode ⇒ FAIL (uncaught sentinel).
- **Distribution sanity:** print min/median/max/card + a histogram summary; the analyst eyeballs for a
  residual spike at a round code. A target failing any check is **dropped** (logged), not used.
The build prints these; the bases are **approved for model use only after the checks pass** (PI gate).

## 10. Block-aware split rules (no leakage)
All NHANES-2017–18-derived bases (`rb_nhanes_weight`, `rb_nhanes_poverty`, and any future NHANES module)
**share respondents (SEQN)** ⇒ they are **ONE block**, not independent worlds. The cross-domain split must
keep the **entire NHANES block on one side** of train/val/test — never train on `rb_nhanes_weight` and
test on `rb_nhanes_poverty` (same people → leakage). Recorded as `source_block: NHANES-2017-18`; the
runner groups by `source_block` for leave-DOMAIN/leave-BLOCK-out (cross-domain plan §3).

## 11. Decision gate
- If the full-module projections yield **usable** bases (§7) → re-run the cross-domain curve with the
  larger NHANES weight/poverty domains (does the flat curve move with *bigger* bases?). **Model runs only
  after the bases pass §9.**
- If they **still** produce only tiny/fragile bases → that is the **trigger for external acquisition**
  (CPS/ACS/IPUMS income-with-top-coding; BRFSS/GSS/ANES by target type) — to be proposed separately.

## 12. Build plan (on approval; data-engineering only, no model)
New `scripts/build_nhanes_role_b.py` (uses `pyreadstat` + `role_b_projection` primitives + the per-variable
sentinel map). Produces: the role-A archive (§5), the role-B bases (§6), provenance (§8), and the §9 check
report. New module code (if any) stays ≤500 LOC, fail-loud, tested. **No `level1_train` / curve runs until
the bases pass §9 and the PI approves.**

**Open decisions for the PI:** (a) the two bases above, or add an `INDFMIN2`-bracket top-coding base?
(proposal: the two above first); (b) `rows ≥ 1500` threshold ok? (c) confirm the per-variable sentinel map
(§3). No code until approved.
