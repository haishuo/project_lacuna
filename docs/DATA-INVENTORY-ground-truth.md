# Data Inventory — ground truth (read-only)

*Read-only inventory. **No architecture, no implementation, no acquisition recommendations** (per PI
instruction — establish ground truth first). Generated 2026-06-06 by inspecting the on-disk data stores
and `lacuna_survey/DATA_ACQUISITION.md`. Scope note: `/mnt/projects/pystatistics-validation/` is a
**sibling project** (out of scope) and is excluded; everything below is within project_lacuna or its
data store `/mnt/data/lacuna/`.*

## 0. Headline correction

**The Stage-1 "12 survey datasets" is option (c): a filtered subset.** The default catalog
(`create_default_catalog`) scans **only** `/mnt/data/lacuna/raw/`, so it never sees a substantial pool of
survey data that is physically present elsewhere — **NHANES, PISA, GSS, ESS** — much of which **preserves
natural missingness (role A)**. The "we only have 12 surveys" picture is an artifact of *what the catalog
is wired to scan*, not of what we possess. **However**, almost all the additional data is **role A
(natural missingness) or raw source files** — which, per the project's own role discipline (design-spec
§9.0), are **validation/source material, not supervised δ-training data.** So the **training-eligible
(role B) corpus is genuinely still 12 → 9 genuine**, while the **validation/face-validity diversity is
already much richer** than the 12-view implied.

---

## 1. Full inventory

### 1a. The default catalog (`cat.list_datasets()`) — 43 datasets, all complete-case (role B)
Source = 4 sklearn built-ins + a scan of `/mnt/data/lacuna/raw/` (39 CSVs). **All have 0.000 natural
missingness** — the `survey_*` CSVs are pre-cleaned complete-case extracts (the `dropna()` in `load_csv`
is a no-op on them). **12 are `survey_*`; 31 are generic ML benchmarks** (abalone, iris, spambase,
superconductor, …) — not surveys.

| survey dataset (role B, in catalog) | origin (inferred) | n | d | targetable | nat-miss preserved | used by Lacuna |
|---|---|---|---|---|---|---|
| survey_cps1988 | CPS 1988 (AER/Ecdat) | 28,155 | 3 | 3 | No | **yes (train)** |
| survey_yrbss | YRBSS (openintro) | 11,522 | 5 | 2 | No | yes (train) |
| survey_computers | Ecdat (product) | 6,259 | 7 | 4 | No | yes — **contaminant** |
| survey_psid7682 | PSID 1976–82 (Ecdat) | 4,165 | 6 | 5 | No | yes (train) |
| survey_workinghours | Ecdat | 3,382 | 11 | 5 | No | yes (train) |
| survey_chile | Chile plebiscite (car) | 2,590 | 4 | 3 | No | yes (train) |
| survey_hmda | HMDA (AER) | 2,380 | 6 | 4 | No | yes (train) |
| survey_bfi | Big-Five (psych) | 2,236 | 28 | 1 | No | yes (train) |
| survey_psid1976 | PSID/Mroz (Ecdat) | 753 | 17 | 12 | No | yes (train) |
| survey_cps1985 | CPS 1985 (AER) | 534 | 4 | 4 | No | yes (train) |
| survey_survey | MASS `survey` (teaching) | 170 | 5 | 5 | No | yes — **contaminant** |
| survey_cars93 | MASS `Cars93` (product) | 82 | 18 | 17 | No | yes — **contaminant** |
| **31 non-survey ML benchmarks** | sklearn / UCI | 82–30,000 | 4–82 | — | No | **no** (not survey) |

### 1b. Physically present but NOT in the catalog (role A natural-missingness archive + raw sources)
The default catalog **does not scan** these directories. **All preserve natural missingness.**

| location | dataset | origin | n | d | nat-miss | role | in catalog? |
|---|---|---|---|---|---|---|---|
| `lacuna_survey/evaluation_data/` | nhanes_dpq_phq9_real | NHANES depression PHQ-9 | 500 | 14 | 7.2% | **A (eval anchor)** | no |
| | nhanes_duq_drug_real | NHANES drug use | 500 | 7 | 16.1% | A | no |
| | nhanes_inq_income_real | NHANES income | 500 | 9 | 9.9% | A | no |
| | nhanes_whq_weight_real | NHANES weight | 500 | 10 | 10.0% | A | no |
| | survey_nhanes_demographics_real | NHANES demographics | 500 | 5 | 1.7% | A | no |
| | pisa2018_gbr_rotation_real | PISA 2018 (GBR) | 500 | 10 | 46.9% | A | no |
| | pisa2022_deu_rotation_real | PISA 2022 (DEU) | 500 | 8 | 29.2% | A | no |
| | survey_gssvocab_real | GSS vocabulary | 500 | 4 | 1.1% | A | no |
| | survey_ucla_textbooks_real | UCLA textbooks | 201 | 7 | 43.5% | A | no |
| | survey_{bfi,chile,cars93,survey,yrbss}_real | real-missingness twins of 5 training sets | 93–13,583 | 4–28 | 0.8–6.3% | A | no |
| `/mnt/data/lacuna/nhanes/` | demographics_clean | NHANES (full) | 135,310 | 281 | **79.6%** | A (source) | no |
| | questionnaire_clean | NHANES (full) | 134,515 | 1,445 | **93.7%** | A (source) | no |
| `/mnt/data/lacuna/incoming/` | DEMO_J, DPQ_J, DUQ_J, INQ_J, WHQ_J (`.xpt`) | NHANES 2017–18 raw SAS | — | — | (raw) | A (source) | no |
| | cy07/cy08 `.sas7bdat` | PISA 2018 / 2022 raw SAS (multi-GB) | — | — | (raw) | A (source) | no |
| `/mnt/data/lacuna/rejected/` | ESS11e04_1 | European Social Survey R11 | 50,116 | 691 | 36.2% | **rejected** (documented dead-end) | no |
| `demo/sample_data/` | 6 labeled MCAR/MAR/MNAR demos (chile, pisa, nhanes×4) | demo | 500–2,700 | 4–14 | 1–47% | demo | no |

*(Data dirs are gitignored per `DATA_ACQUISITION.md` storage policy — present on this machine/Forge, not
in git history.)*

---

## 2. What was excluded from the Stage-1 inventory, and why

The Stage-1 inventory listed only `raw/survey_*` because **that is the only directory the catalog scans**
and because **only those are role-B complete-case training bases.** Specifically:

- **NHANES present? YES** — as a **role-A source archive** (`/mnt/data/lacuna/nhanes/` full extracts +
  `incoming/*.xpt` raw modules: demographics, depression PHQ-9, drug use, income, weight) and **5 extracted
  role-A eval anchors** in `evaluation_data/`. Excluded from Stage 1 because (a) not catalog-wired, and (b)
  it is **natural-missingness data → validation/source, not supervised δ-training** (§9.0). No role-B
  complete-case NHANES projection exists yet.
- **BRFSS present? NO** — not anywhere on disk.
- **Other health surveys? YES** — NHANES (above) and `yrbss` (the one health set already in role B).
- **Political / attitude beyond bfi & chile? YES** — **GSS** (`gssvocab_real`), **ESS R11** (present but
  **rejected**: its "rotating modules" are country-specific admin-NaN, not MCAR-by-design — documented in
  `DATA_ACQUISITION.md` §6), and **PISA 2018/2022** (education attitudes). `ANES`/`Pew` are listed as
  *candidates not yet acquired*.

---

## 3. The three-way distinction (the PI's question, answered)

| layer | count | what it is |
|---|---|---|
| **(a) physically present** | **≫ 12** | 12 role-B `survey_*` + 14 role-A eval anchors (NHANES×5, PISA×2, GSS, UCLA, +5 real twins) + NHANES full source (nhanes/ + 5 raw `.xpt`) + PISA raw SAS (2 files) + ESS R11 (rejected) + 6 demo. Spans health, education, attitudes, political, psychology, labor, finance, demographics. |
| **(b) catalog-wired (`cat.list_datasets()`)** | **43** | 12 `survey_*` + 31 generic ML benchmarks. **Only the `raw/` scan** — `create_default_catalog` ignores `nhanes/`, `incoming/`, `evaluation_data/`, `demo/`. |
| **(c) Stage-1 training-eligible (role B)** | **12 → 9 genuine** | the `survey_*` subset, minus the 3 contaminants (cars93, computers, survey). |

**So the Stage-1 inventory is (c): a filtered subset of (b); and (b) is itself a subset of (a) that omits
the entire role-A archive and the NHANES/PISA raw sources.** Critically, the gap between (c) and (a) is
**almost entirely role-A (natural-missingness) data**, which the project's own discipline (§9.0) holds is
**validation/source, not training**. To become role-B training bases, NHANES/PISA/GSS would each need a
**complete-case projection extracted** — which has **not** been done (only the 12 `survey_*` have role-B
projections).

---

## 4. Domain-diversity table

| domain | role-B training-eligible (raw/survey_*) | additional, role-A or source (present, not training) |
|---|---|---|
| **labor/income** | cps1985, cps1988, psid1976, psid7682, workinghours **(5)** | NHANES income |
| **health** | yrbss **(1)** | NHANES depression, drug, weight, demographics, questionnaire (full) |
| **education** | — | PISA 2018, PISA 2022 (extract + raw SAS), UCLA textbooks |
| **social attitudes** | — | GSS vocab; ESS R11 (rejected) |
| **political** | chile **(1)** | chile_real |
| **psychology** | bfi **(1)** | bfi_real |
| **finance** | hmda **(1)** | — |
| **demographics** | — | NHANES demographics |
| **consumer/product** | cars93, computers **(2 — contaminants)** | — |
| **other/teaching** | survey **(1 — contaminant)** | UCLA textbooks |

**Counts:**
- **Total physically present (a):** 12 role-B `survey_*` + ~22 additional survey-relevant (14 eval anchors
  + ~7 raw sources + ESS) + 31 non-survey ML + 6 demo.
- **Total targetable columns (role-B `survey_*`):** **65** (12 datasets); **genuine-9 subset:** ~40.
- **Effective domain count:**
  - **role-B training (genuine-9):** ~**5 domains**, heavily **labor/income-skewed** (5 of 9 datasets) →
    effective ≈ **3–4** independent domains.
  - **all physically-present survey data (a):** ~**8–9 domains** (labor, health, education, attitudes,
    political, psychology, finance, demographics) — but the **non-labor breadth lives almost entirely in
    role-A**, not in the training corpus.

---

## 5. Ground-truth answer to "lack diversity, or incomplete inventory?"

**Both, in different layers — and the distinction matters:**
- **For supervised δ-training (role B):** we **genuinely lack diversity.** Only 12 → 9 genuine complete-
  case bases exist, ~3–4 effective domains, labor-skewed. The Stage-1 view was accurate *for training*.
- **For the physical inventory and for validation (role A):** the inventory was **incomplete.** NHANES
  (health, multiple modules), PISA (education), GSS (attitudes) are **already on disk with natural
  missingness preserved** — a much richer manifold-/OOD-/face-validity set than the 12-view suggested, plus
  raw NHANES/PISA sources that could in principle be projected to role-B without new downloads.

This is a **factual inventory only.** Any decision about projecting role-A sources into role-B training
bases, acquiring new domains, or reconciling the old `lacuna_survey` anchor framework with the current
δ-prior work is **deferred** (no recommendation here, per instruction).
