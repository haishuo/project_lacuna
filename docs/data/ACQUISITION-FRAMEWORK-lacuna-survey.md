# Lacuna-Survey — Data Acquisition, Ingestion, Accounting & Governance Framework

*Design document — **specification & analysis only**. No code, no experiments, no downloads, no
architecture changes, no metadata channel, no Level-1 revisit (binding). Status: **for PI review.**
Prepares the project to incorporate future survey datasets consistently once collected. Governed by
`NORTH-STAR.md` (Scope; §2 identification; §3½ manifold) and `MASTER-lacuna-survey-architecture.md`.
Grounded in the Stage-1 + cross-domain evidence (`feasibility-stage1-crossdomain-findings.md`): the
φ-spine is validated; the bottleneck is corpus diversity + continuous-target coverage; out-of-family
transfer improves with domain diversity (leave-one-domain-out: bfi +0.064, yrbss +0.037 significant;
NHANES +0.045 positive-noisy).*

---

## 1. Acquisition framework — what is an admissible Lacuna-Survey dataset?

**Admissible** iff ALL hold:
1. **Survey-administered** — self- or interviewer-administered questionnaire (North Star Scope). NOT
   instrument/assay/lab data (→ Lacuna-Instrument; assay-LOD is out of scope), NOT purely administrative
   records, NOT derived/aggregate tables.
2. **Respondent-level tabular**, numeric-codable (one row per respondent).
3. **≥1 codebook-curated CONTINUOUS target** (cardinality ≥30, within plausible range, sentinel-clean) —
   because the current idiom vocabulary (top-coding, social-desirability self-censoring) needs continuous
   targets. Ordinal/Likert/binary-only datasets are **not admissible as supervised worlds** under the
   current vocabulary (they may be archived as role-A diversity sources).
4. **≥2 columns** (≥1 continuous target + ≥1 predictor).
5. **Codebook available** — documenting, per variable: type, valid range/set, refuse/DK/NA codes. **No
   codebook ⇒ not ingestible** (the raw-NHANES/ESS auto-curation failure proved heuristics corrupt the
   prior; age 77 vs refuse 77 is unresolvable without a codebook).
6. **Legally usable** (public or licensed for research).

**Inclusion criteria (all required):** survey-administered · ≥1 codebook continuous target · ≥750
projectable complete-case rows (acceptable tier) · codebook present · adds a distinct domain OR world.

**Exclusion criteria (any disqualifies):** instrument/assay/lab data · administrative-only · no continuous
target · no codebook · non-survey contaminant (product/teaching/ML-benchmark) · aggregate/derived only ·
duplicates an existing world with no new variation.

**Distinct DOMAIN vs CORRELATED SLICE.**
- **Domain** = a substantive subject area with distinct variable *semantics* and population (labor/income,
  wealth, health, demographics, education, attitudes, psychology, finance, political…).
- **Correlated slice** = the *same survey program* decomposed by **module**, **country**, or **wave**,
  sharing respondents and/or near-identical instrument. A slice is **not** a new independent world.

**Counting waves / modules / country splits (the discipline that prevents fake diversity):**
- **Modules** (NHANES DPQ/WHQ/INQ): **same respondents** (shared id) ⇒ **ONE block**. May span several
  *domains*, but with a **correlation discount** and **never split across train/test** (leakage).
- **Country splits** (ESS Germany vs France): same questionnaire, different population ⇒ **weakly
  independent**; count as **one world-family with a discount**, not N worlds.
- **Waves** (cross-sectional, e.g. NHANES 2017-18 vs 2015-16; GSS years): different respondents, near-
  identical instrument ⇒ **weakly independent** (the X-manifold repeats); count with a discount. **Panel**
  waves (PSID/NLSY re-interview the SAME people) ⇒ **strongly correlated, one world**.

**Independent world (definition).** A (survey program, population, instrument) combination whose observed-
data **footprint contributes genuinely new variation** to the survey footprint-manifold (§3½ M2).
Operationally, two datasets are the **same world** if they share (a) respondents (id overlap) **or** (b)
the same questionnaire instrument on overlapping/similar populations; **different worlds** if they differ
in program **and** population **and** instrument. Slices get a **fractional** world-weight (a block of *k*
correlated modules ≈ 1 world). **The independent-world count — not the file count — is the diversity
metric** for the cross-domain leave-one-WORLD/BLOCK-out curve.

---

## 2. Corpus accounting framework — the ledger (never file count)

**Standardized inventory = a per-dataset/base "corpus ledger" record:**

| field | meaning |
|---|---|
| `dataset_id`, `program`, `wave`, `country` | provenance keys |
| `domain` | controlled vocabulary (§1) |
| `world_id` / `block_id` | datasets sharing respondents/instrument share this id |
| `correlation_discount` | 1.0 for an independent world; <1 for a slice (e.g. 1/k for k correlated modules) |
| `n_respondents` | role-A row count |
| `n_complete_rows` | role-B projected complete-case rows |
| `d_columns`, `n_continuous_targets` | post-curation column counts |
| `continuous_target_vars` | list (codebook-typed continuous, card≥30, range-ok, spike≤0.002) |
| `idiom_relevance` | which idioms the targets support (top-coding / self-censoring / [future: item-nonresponse]) |
| `natural_missingness_available`, `natural_missingness_rate` | role-A presence |
| `role_A_available` (path), `role_B_feasible` | `preferred`/`acceptable`/`fragile`/`no` |
| `codebook_available` (source/version) | gating requirement |
| `provenance` | source, license, acquisition date, file hash, sentinel map, exclusions, quality checks |

**Aggregate diversity dashboard (the metrics that matter — NOT file count):**
- `n_distinct_domains`; `n_independent_worlds` (Σ correlation-discounted) ;
- `n_continuous_targets` total **and per domain** **and per idiom**;
- total respondents (role-A) and total complete-case rows (role-B);
- **domain-coverage table** (domain × #worlds × #continuous-targets × #rows);
- **idiom-coverage table** (idiom × #targets × #worlds);
- **role-A coverage** (#worlds with natural missingness archived).

**Binding rule:** a dataset's *diversity contribution* = its **marginal** addition to (worlds, domains,
continuous-targets), **discounted for correlation** — never "+1 file." Two NHANES modules add domain
breadth but ~1 world; ten ESS countries add ~1 world-family.

---

## 3. Ingestion & projection standards

**Role-A vs role-B (formal).**
- **Role A** — the raw survey table with **natural missingness preserved** (refuse/DK recoded to NaN per
  codebook, but rows **not** dropped). **Archived; NEVER supervised.** Use: manifold/OOD/detectability
  validation + future generation (§9.0 of the design spec).
- **Role B** — the **complete-case projection** over the **selected** target+predictor columns. **The sole
  supervised δ corpus.** We impose the holes.
- **Ingestion is DUAL-OUTPUT:** every source → (role-A archive, role-B projection). Two artifacts, one
  source. **Never train on natural missingness.**

**Provenance requirements (per artifact):** program/wave/country · acquisition date · license · file hash
· codebook source/version · **per-variable sentinel map applied** · design-columns excluded · target/
predictor columns selected · rows in/out · complete-case rule · usability tier · quality-check results ·
`world_id`/`block_id` · role-A archive path. (Mirrors the NHANES provenance manifest already implemented.)

**Codebook requirements (gating).** A dataset is **not ingestible without a codebook** documenting, per
variable: type (continuous / ordinal / categorical / id / design / weight), valid range/set, refuse/DK/NA
codes. The codebook drives §sentinels and §exclusions. *No codebook ⇒ no ingest.*

**Sentinel-code handling (formal, conservative).**
- **Per-variable, codebook-driven. No global sentinel policy.**
- **Default = NO recoding** for any variable lacking a trusted codebook entry. **Valid values take
  precedence over generic sentinel patterns.**
- Recode a value → NaN **only** if the codebook documents it as refuse/DK/NA for *that* variable **and** it
  is outside the variable's valid range.
- **Age-type variables are the canonical warning:** never recode a code that is a valid observation (age
  77). Default to preserving data.
- **Post-recode gate:** per target, sentinel-spike-frac ≤ **0.002** AND all values within plausible range;
  failing targets are **dropped** (logged), not used.

**Continuous-target eligibility (all required):** codebook-typed **continuous** (not ordinal-coded/
categorical/id/design/weight) · cardinality ≥ **30** (post-recode) · all values within codebook plausible
range · sentinel-spike-frac ≤ 0.002. Ordinal/Likert/binary → **predictors only** (a future ordinal idiom
is out of scope now).

**Design-variable / survey-weight exclusions (by codebook TYPE, not just name):** respondent ids · survey
weights (incl. replicate weights, e.g. `WTMREP*`) · variance/PSU/strata (`SDMV*`, prob) · interview-admin
flags (language/proxy/interpreter) · data-release/cycle markers. **Never targets or predictors.**

**Complete-case projection requirements.** Project over the **selected** target+predictor set **only**
(not all columns — avoids re-introducing sparse/ordinal noise) · keep rows complete over that set ·
usability tiers **preferred ≥1500 / acceptable ≥750 / fragile <750** (fragile ⇒ excluded from training,
logged, an acquisition trigger) · **no standardization at projection** (φ standardizes within observed at
batch time) · **block-aware** (datasets sharing respondents share `block_id`; the cross-domain split keeps
a block together).

**Reusable ingestion checklist (run per dataset):**
1. Confirm survey-administered + admissible (§1).
2. Obtain codebook; map per-variable type + sentinel codes.
3. Recode sentinels → NaN **per codebook** (conservative; default no recode).
4. Exclude id/design/weight columns (by codebook type).
5. Identify continuous targets (continuous-type + card≥30 + range-ok + spike≤0.002).
6. Select target + predictor columns.
7. **Archive role-A** (natural missingness preserved).
8. Complete-case project → **role-B**; record usability tier.
9. Quality checks (spike, range, distribution eyeball); **drop failing targets**.
10. Assign `world_id`/`block_id`, `domain`, `correlation_discount`.
11. Write provenance manifest; **append to the corpus ledger** (§2).
12. If fragile/role-A-only ⇒ **do not use for supervision** (validation/source only).

---

## 4. Acquisition prioritization rubric — expected scientific value (two axes)

Two distinct value axes; report both (the evidence shows they diverge — attitudes gave the biggest
transfer gain yet have the fewest continuous targets):
- **(a) Idiom/continuous-target value** — does it supply continuous targets that match the current idioms
  (top-coding, social-desirability self-censoring)?
- **(b) Domain-diversity value** — does it add an independent domain/world that improves out-of-family
  transfer (the leave-one-domain-out result)?

| category | (a) idiom value | (b) diversity value | why |
|---|---|---|---|
| **Income / earnings** | **HIGH** | high | continuous; **public-use top-coding** + high-earner refusal ⇒ *both* idioms directly; large samples (e.g. CPS/ACS/IPUMS, SIPP) |
| **Wealth / net worth** | **HIGH** | high | continuous; **top-coded + canonical wealth-refusal MNAR** ⇒ top-coding + self-censoring (HRS, SCF, NLSY supplements) |
| **Poverty / income-to-needs** | **HIGH** | med | continuous; **naturally top-coded** (NHANES INDFMPIR caps at 5) ⇒ top-coding |
| **Self-reported physical measurements (weight/height/BMI)** | **HIGH** | med | continuous; **weight underreporting = the social-desirability self-censoring idiom** we most need to test (the flat-likelihood case); BRFSS, Add Health, NHANES WHQ |
| **Expenditures / consumption** | **MED-HIGH** | med | continuous; recall-refusal + top-coding; but complex instruments (CE survey) |
| **Health (SELF-REPORTED continuous only)** | **MED** | med | self-reported continuous health items only — **lab/assay measures are instrument-LOD, out of scope** |
| **Education (years / test scores)** | **MED-LOW** | med | years-of-schooling semi-continuous; scores continuous but missingness often **MCAR by design** (rotated booklets) ⇒ good for the MCAR-vs-not stage, weak for δ |
| **Attitudes / political (GSS, ANES, ESS, WVS)** | **LOW** | **HIGH** | ordinal/Likert-dominated ⇒ **few continuous targets**, and ordinal item-nonresponse is a *different, not-yet-in-vocabulary* idiom — **but** the biggest leave-one-domain-out transfer gain came from psychology/attitudes; **high diversity value, low idiom-target value** |

**Prioritization:** lead with **(a)** — income / wealth / poverty / self-reported physical measurements —
because they supply continuous targets matching the current idioms *and* add domains. Value **(b)** for
attitudes/political as **diversity-only** acquisitions (they help transfer but not idiom coverage). **Rank
by scientific value, not ease of download.** (Example sources above are *illustrative of the category*,
not download instructions — no acquisition is recommended here.)

---

## 5. Evidence framework — when to continue / stop / generate / fund

Tied directly to the cross-domain learning-curve philosophy (the curve is the arbiter, judged at the
dissertation/grant standard — `design-spec §9.A`).

**Continue acquisition** when:
- the **leave-one-domain-out transfer gain** (DIVERSE − NARROW) stays **positive** and becomes
  **significant for the idiom-relevant continuous-target domains** (income/weight/poverty), not just
  bfi/yrbss; **and/or**
- the cross-domain **slope firms up** as continuous-target domains are added (each new domain adds
  non-redundant footprint variation — M2 coverage grows). *The curve is still rising at the edge.*

**Stop acquisition** when:
- the leave-one-domain-out gain **saturates** — adding further continuous-target domains no longer improves
  held-out transfer across several additions — **and** calibration/detectability on held-out domains
  plateaus. *Confirm with continuous-target domains (not attitudes) and enough additions to rule out noise.*

**Introduce synthetic survey generation** when:
- real-data acquisition **saturates** *but* held-out performance is still below useful **and** the
  bottleneck is shown to be **coverage** (footprint-manifold gaps), not identifiability. Then train a
  generator over real survey X to fill gaps (domain randomization). **Guardrail:** generated X expands
  *training* only, **never** validation; adopt only if it improves **held-out REAL** transfer (§9.6).

**Apply for external funding** when:
- the **cheaply-acquirable** corpus shows a **credible positive cross-domain slope but is exhausted** —
  the next continuous-target domains require **funded** effort (restricted-access surveys, e.g. licensed
  panels; large-scale codebook curation labor; NAEP-style restricted-use). The grant case = *φ-spine
  validated + diversity-improves-transfer (out-of-family) + a positive-but-data-limited curve + this
  acquisition rubric with per-domain expected value*.

---

## 6. Dissertation & grant narrative (future chapter foundation)

**The arc, as a single scientific story:**

1. **Architecture audit.** Lacuna began as a 3-class MCAR/MAR/MNAR *mechanism classifier* on a BERT-style
   set-transformer. The inference target shifted (North Star reset) to a **calibrated prior over the
   sensitivity parameter δ** — a *governance* tool, not a mechanism oracle. We asked whether the inherited
   backbone fit the new question (the MLP-vs-CNN concern), made it concrete, and found it **partially**
   valid: right symmetry, wrong axis.

2. **Representation finding.** A Bayes-optimal oracle proved the detectable idiom's signal exists; hand-
   computed observed-marginal **order statistics** carry it; the **encoder did not**. A representation
   probe localized the cause: the BERT backbone's **averaged** representations do not preserve the within-
   column order-statistic signal (probe ~0.52, preservation R²≤0), while a **column-primary distribution**
   representation recovers it (Stage-0: φ 0.735 vs backbone 0.548).

3. **φ-spine replacement.** We replaced the backbone as the δ spine with a **column-primary distribution
   encoder (φ)** — order-statistic pooling over each column's observed values — keeping the validated
   scaffold (generators, oracle, leakage gate, calibration). Validated in-pipeline (≫ backbone).

4. **Cross-domain transfer.** On the small, labor-skewed genuine corpus the learned channel is **data-
   limited**; in-family scaling is weak. But the **tightened, block-aware leave-one-domain-out** shows
   **domain diversity improves out-of-family transfer** (psychology +0.064, health +0.037 significant;
   NHANES weight/poverty +0.045 positive). The value of diversity appears exactly where it should — in
   transfer to *unseen* domains.

5. **Corpus limitation.** Clean **continuous** survey targets are **scarce** (most survey data is ordinal/
   Likert); the genuine corpus is small and labor-skewed; raw-source curation requires codebooks. The
   binding constraint is now **corpus diversity + continuous-target coverage**, not architecture.

6. **Acquisition as the next bottleneck.** The architecture question is settled. The scientific question is
   now: *does a diverse, continuous-target survey corpus close the out-of-family gap and make the δ-prior
   calibrated across domains?* This framework — admissibility, world/domain accounting, codebook-aware
   ingestion, the prioritization rubric, the curve-as-arbiter evidence rules — is the apparatus to answer
   it consistently as datasets arrive. **The grant ask is the funded corpus to run the decisive curve.**

**Deeper contributions to foreground** (governance/epistemology, beyond the engineering): the **comparison-
class** framing (detectability is relative to a *named plausible class*, the manifold — never "air"); the
**two-stage** design (identifiable MCAR-gate → manifold-relative δ-prior → **unknown/abstain**); the honest
**non-identifiability** discipline; and the **"lab-coat fraction"** — quantifying how much of sensitivity-
analysis information is in the *data* vs the *semantics* — a potentially profound claim about the
epistemology of sensitivity analysis, which this corpus program is the prerequisite to measure.

---

*No downloads, no implementation, no runs, no architecture change are proposed by this document. It
defines the consistent apparatus for incorporating future survey datasets, and the evidence rules that
will tell us when to continue, stop, generate, or seek funding.*
