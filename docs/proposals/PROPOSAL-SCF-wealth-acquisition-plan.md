# Acquisition Plan — Survey of Consumer Finances (SCF) → role-B wealth domain

*Acquisition planning document — **specification & analysis only. No download, no code, no runs, no
architecture change** (binding). Status: **for PI review / go-decision before any download.** Governed by
`NORTH-STAR.md`, `MASTER-lacuna-survey-architecture.md`, and the
`ACQUISITION-FRAMEWORK-lacuna-survey.md` (admissibility §1, ledger §2, ingestion §3, rubric §4, evidence
§5). Target selected by PI 2026-06-06 (wealth, SCF preferred).*

---

## 0. Why this dataset, in one paragraph

The current corpus is **labor/income-skewed** (5 of 9 genuine surveys are `labor`; diversity lives in
single datasets — bfi/psychology, chile/political, hmda/finance, yrbss/health). The tightened
leave-one-domain-out evidence (`feasibility-stage1-crossdomain-findings.md` §7) showed domain diversity
**improves out-of-family transfer** — significantly for bfi/yrbss, **positively but noisily for the
idiom-relevant continuous-target NHANES domain (weight/poverty, +0.045 ± 0.058)**. The acquisition rubric
(§4) ranks **wealth / net worth** HIGH on *both* axes: continuous targets matching the current idioms
**and** a genuinely **new domain** distinct from the labor/income core. SCF is the canonical public wealth
survey. It is the single acquisition that can (a) add continuous targets to firm the idiom transfer and
(b) supply a **brand-new held-out `wealth` domain** for a *decisive* leave-one-domain-out — the exact test
the noisy NHANES result could not deliver.

---

## 1. Honest scope correction — what SCF contributes (and what it does NOT)

A correction that governs the rest of this plan, stated up front so the framing is not oversold:

- **Role-B training imposes *our own* holes** (the `top_coding` / `own_value` generators) on **clean
  continuous targets**. Therefore the **only** training requirement on the source is *clean continuous
  targets in a new domain* — **NOT** that the source already exhibits top-coding.
- **SCF does little classic top-coding.** Its design *oversamples the wealthy* (a list sample from
  tax records) precisely to **capture the high-wealth tail** — the opposite of CPS/ACS statutory
  top-coding. So SCF's value is **not** "real top-coding for face validity."
- **What SCF genuinely contributes:**
  1. **A new `wealth` domain** (diversity axis (b)) — the decisive held-out-domain test.
  2. **Clean, high-cardinality continuous targets** (net worth, income, asset components) for the
     **role-B** supervised δ corpus under *our* imposed idioms.
  3. Real-world face validity for the **self-censoring / wealth-refusal** idiom (the *flat* idiom — the
     sensitive-item case the "lab-coat fraction" framing cares about) — but this is a **role-A /
     interpretive** point, weakened by §4 below.
- **What SCF does NOT contribute:** a strong **role-A natural-missingness archive** (see §4 — the public
  file is multiply imputed, so item nonresponse is already filled). **This is a role-B-first acquisition.**

This keeps us honest against the rubric's nominal "top-coded + wealth-refusal" label: the **wealth-refusal
MNAR** characterization is real and substantively important, but **for training we use clean targets +
imposed holes**, and SCF's top-coding contribution is minimal. Recorded so no later claim overstates it.

---

## 2. Admissibility check (framework §1)

| criterion | SCF | verdict |
|---|---|---|
| 1. Survey-administered | interviewer-administered triennial financial survey (Federal Reserve) | ✅ |
| 2. Respondent-level tabular, numeric-codable | one row per family (per implicate; see §4) | ✅ |
| 3. ≥1 codebook CONTINUOUS target (card≥30, range-ok, sentinel-clean) | net worth, income, asset components — all continuous, card ≫ 30 | ✅ |
| 4. ≥2 columns (≥1 continuous target + ≥1 predictor) | many continuous + demographic predictors | ✅ |
| 5. Codebook available (type, range, refuse/DK/NA codes) | SCF codebook + Summary Extract documentation (public) | ✅ |
| 6. Legally usable | **public domain**, no registration, no license gate | ✅ |

**Inclusion criteria (all required):** survey-administered ✅ · ≥1 continuous target ✅ · ≥750 projectable
complete-case rows ✅ (expected — see §5) · codebook present ✅ · **adds a distinct domain** ✅ (`wealth`,
new). **Exclusion criteria:** none triggered (not instrument/assay, not administrative-only, not a
contaminant, not aggregate/derived). **→ ADMISSIBLE.**

---

## 3. Source specifics

- **Program / wave:** Survey of Consumer Finances, **2022 wave** (latest; triennial 1989…2019, 2022).
- **Publisher:** U.S. Federal Reserve Board, `federalreserve.gov/econres/scfindex.htm` (public-use).
- **File to use — the SCF *Summary Extract* public dataset** (readable variable names:
  `networth`, `income`, `asset`, `fin`, `nfin`, `houses`, `debt`, `age`, `edcl`, …). **Prefer this over
  the full public-use file**, whose variables are cryptic X-codes (e.g. `X42001` = weight) requiring the
  full codebook to decode. This mirrors the lesson already recorded in the findings: **use the curated /
  documented extract** (as we pivoted to the v1.0 `*_real` extracts) rather than auto-curating cryptic raw
  columns — the raw-NHANES/ESS auto-curation failure (`feasibility…` §2) is the warning.
- **Codebook:** the SCF codebook (full file) + the Summary Extract documentation (Kennickell extract
  programs). Both public; both document per-variable type and the SCF missing/special-value conventions.
- **Format:** Stata `.dta` / SAS / CSV available. Numeric-codable throughout.
- **Access cost:** direct public download; **no registration, no license** (contrast HRS).

*(No URL is fetched and no file is downloaded by this document. The above is for the go-decision.)*

---

## 4. The multiple-imputation wrinkle (the load-bearing ingestion decision)

**SCF public files are *multiply imputed*.** Missing values are imputed **5 times**, producing **5
"implicates" per family** — every household appears as **5 rows** (implicate id, e.g. `Y1`/`YY1`). Correct
SCF analysis uses repeated-imputation inference (RII) combining across implicates; the survey weight `WGT`
is divided by 5 in the stacked file.

**Consequences for our pipeline — two binding rules:**

1. **The 5 implicates are NOT 5× data.** They are imputed copies of the **same families**. Treating them
   as 5× rows is exactly the **correlated-slice inflation trap** the framework warns against (§1
   "counting waves/modules"). **Rule: project role-B from a SINGLE implicate (implicate 1 only)** → one
   row per family. (Alternative: keep all 5 but assign them **one `block_id`** so they never split across
   train/test — but single-implicate is cleaner and avoids leakage entirely. **Recommend implicate 1.**)
2. **Role-A is compromised.** Because the public file's item nonresponse is **already imputed away**, SCF
   provides **no clean natural-missingness archive**. Therefore SCF is ingested as **role-B-only**; the
   role-A archive field is `role_A_available = no (imputed)`. We do **not** train on it (we never train on
   role-A anyway), and we do **not** claim SCF as a natural-missingness validation source. This is the one
   real cost of choosing SCF over HRS/ACS, and it is acceptable because the **decisive test is the
   role-B held-out wealth domain**, which does not need role-A.

---

## 5. Targets, predictors, projection design

**Candidate continuous targets** (codebook-continuous, expect card ≫ 30, wide range; Summary Extract
names): `networth` (net worth — **can be NEGATIVE**, debt > assets), `income` (total family income),
`asset` (total assets), `fin` (financial assets), `nfin` (non-financial assets), `houses` (primary
residence value), `debt` (total debt). All dollar-valued, continuous, heavily right-skewed.

- **Plausible-range check must admit negatives for `networth`/`debt`-derived nets** — the framework's
  range gate (§3) must not flag legitimate negative net worth as a sentinel. Recorded as a
  per-variable codebook range, not a generic "≥0" rule.
- **Skew:** φ standardizes within observed and pools order-statistics/quantiles at batch time — robust to
  skew on raw values. **No transform** (architecture frozen; no preprocessing changes).
- **Disclosure rounding:** SCF rounds dollar values for disclosure; cardinality remains ≫ 30 for
  net worth/income. A small risk for narrow asset components (`houses` among non-owners = many exact 0s →
  spike) — the **spike ≤ 0.002 gate (§3) will catch and DROP** any target whose zero-spike is excessive
  (e.g. `houses` for renters). Expected drops are a feature, not a failure (as the NHANES `WHD120` drop
  demonstrated the gate works).

**Predictors:** demographic / structural columns the codebook types as predictors — `age`, `edcl`
(education class), `married`, `kids`, `famstruct`, `racecl`, `occat1` (occupation class), `lf` (labor
force). Ordinal/categorical → **predictors only** (current vocabulary has no ordinal idiom).

**Complete-case projection (framework §3):** select {1–N continuous targets + predictors} → drop the
weight/design/id columns (§6) → keep rows complete over the selected set → **no standardization at
projection**. Usability tier by row count: **preferred ≥1500 / acceptable ≥750 / fragile <750.** SCF 2022
≈ **4,500+ families** (single implicate) → expected **preferred** tier even after complete-case attrition.

---

## 6. Exclusions — SCF-specific design/weight/id columns (framework §3)

Exclude by **codebook TYPE**, never as targets or predictors:
- **Survey weight:** `wgt` (Summary Extract) / `X42001` (full file) and any replicate weights.
- **Implicate / id:** the implicate index (`Y1`/`YY1`), case id (`YY1`/`y1` household id), `id`.
- **Design / variance:** any PSU/strata/bootstrap-replicate-weight columns if present in the full file.
- **Derived flags** the Summary Extract carries for convenience that are not respondent answers.

---

## 7. Corpus-ledger entry (framework §2 — filled, pending build)

| field | value (planned) |
|---|---|
| `dataset_id` | `rb_scf2022_wealth` |
| `program` / `wave` / `country` | SCF / 2022 / US |
| `domain` | **`wealth`** (NEW) |
| `world_id` / `block_id` | `scf` (one world; future waves → same block, weakly-independent discount) |
| `correlation_discount` | 1.0 (independent world; **single implicate** so no within-implicate inflation) |
| `n_respondents` | ≈ 4,500+ families (single implicate) |
| `n_complete_rows` | TBD at build (expect preferred tier ≥1500) |
| `d_columns` / `n_continuous_targets` | TBD (≥1 continuous target required to pass) |
| `continuous_target_vars` | from {`networth`,`income`,`asset`,`fin`,`nfin`} that pass the gates |
| `idiom_relevance` | top-coding (imposed) + **self-censoring/wealth-refusal (face-valid)** |
| `natural_missingness_available` | **no (multiply imputed)** |
| `role_A_available` | **no (imputed)** — role-B-only acquisition |
| `role_B_feasible` | expected **preferred** |
| `codebook_available` | SCF codebook + Summary Extract documentation (public) |
| `provenance` | source URL, license (public domain), acquisition date, file hash, sentinel map, exclusions, quality checks, **implicate-1 rule**, role-A=imputed-none |

**Diversity contribution (binding rule §2):** **+1 independent world, +1 new domain (`wealth`),
+`n` continuous targets** — discounted for the single-implicate decision (no slice inflation). Future SCF
waves would add **weakly-independent** worlds in the **same `scf` block** (cross-sectional, near-identical
instrument, different families) — counted with a discount, never split across train/test.

---

## 8. Integration plan (what the build would touch — NOT done here)

When approved to build (separate task), the work mirrors the NHANES template:
1. **`survey_catalog.py`:** add `wealth` to the `DOMAIN` vocabulary; register `rb_scf2022_wealth` with
   `block_id = "scf"`. (Role-B bases are registered via the projection registry, as the NHANES bases were.)
2. **Build script** analogous to `scripts/build_nhanes_role_b.py` (`build_scf_role_b.py`): read the
   Summary Extract → apply the per-variable codebook sentinel map (conservative; SCF's special codes only) →
   **select implicate 1** → exclude weight/id/design (§6) → complete-case project over selected
   targets+predictors → run the §3 quality gates (spike ≤ 0.002, range incl. negatives, tier) → write the
   provenance manifest + append the §7 ledger entry → **archive role-A = none (imputed)**.
3. **Re-run `scripts/run_curve_tightened.py`** with `wealth` as a **held-out domain** in the block-aware
   leave-one-domain-out (NARROW=labor vs DIVERSE=+wealth+others; test on held-out `wealth`). The SCF block
   moves together (single implicate, so this is automatic).
4. Tests for the new build path (fail-loud, injected-RNG, ≤500 LOC per file), suite stays green.

**No file in this plan is created or modified. Steps 1–4 are the *next* task, gated on the go-decision.**

---

## 9. The experiment this enables (the point)

A **decisive leave-one-domain-out with a new continuous-target wealth domain held out:**
- **Reading 1 (transfer):** does DIVERSE training (labor + psychology + health + finance + NHANES + **wealth**)
  transfer to a held-out **`wealth`** test better than NARROW (labor-only)? This is the **out-of-family**
  probe that gave the *significant* bfi/yrbss gains — now run on the **idiom-relevant continuous** domain
  the NHANES result was too noisy to settle.
- **Reading 2 (firming NHANES):** with a second large continuous-target domain in training, re-run the
  NHANES-held-out leave-one-out — does its noisy +0.045 firm up (tighter SE)?
- **Evidence rule it feeds (framework §5):** if the wealth-held-out transfer gain is **positive and
  significant** for an idiom-relevant continuous domain ⇒ **continue acquisition** (and strengthens the
  grant case: diversity helps transfer for the domains we most care about, not just Likert/bfi). If it
  **saturates** ⇒ approaches the **stop** condition. Either way it is a **clean reading**, because wealth
  is genuinely out-of-family (unlike CPS/ACS income, which would be in-family with the labor core).

---

## 10. Open decisions for PI (before any download)

1. **Wave:** SCF **2022 only** (recommended — latest, one clean world), or also 2019 (a second
   weakly-independent world in the `scf` block, for a within-block robustness read)?
2. **Implicate rule:** **implicate 1 only** (recommended — clean, no inflation), or all 5 under one
   `block_id`?
3. **Target set:** lead with **`networth` + `income`** (two strongest continuous targets), or include
   asset components (`fin`/`nfin`/`asset`) subject to the spike gate?
4. **Role-A acceptance:** confirm we accept SCF as **role-B-only** (no natural-missingness archive) — i.e.
   the wrinkle in §4 is an acceptable cost for the new wealth domain. (If role-A is required, **HRS** is
   the alternative — registration-gated but role-A intact.)
5. **Go/no-go on download:** this plan recommends a go for SCF 2022 Summary Extract; **no download or build
   happens until PI approves**, per the binding constraint.

---

*No download, no code, no runs, no architecture change are performed by this document. It specifies a
single admissible acquisition (SCF 2022 → role-B `wealth` domain), its ingestion design, its ledger entry,
the experiment it enables, and the open decisions to resolve before the build task begins.*
