# Level-1 Design Spec — Lacuna-Survey core (prior-aware governance model)

*Design spec only. **No code, no experiments.** Status: **for PI approval.** Defines the architecture we
are willing to build as the new Lacuna-Survey core. Implements Level 1 of `ARCHITECTURE-OBJECT-revised.md`
(object = *observed-data law + named survey-manifold prior → calibrated δ-prior + detectability +
abstention*). Governed by `NORTH-STAR.md` and `CLAUDE.md` (one-job modules, ≤500 LOC, fail-loud,
determinism via injected RNG, tests first-class). **Level-2 conditional reference/deviation is explicitly
excluded** (separate Gate-II decision).*

---

## 1. Model object

### 1.0 Inputs
- `X [n,d]` observed values, `R [n,d]` observed mask (True=observed), and a **supplied target column
  index `t`** (human-parity §4.8: the analyst names the column whose δ she wants).
- Row subsampling to `max_rows` via an **injected `RNGState`** (Rule 6); deterministic thereafter.
- **Column-major** data path (NOT the BERT per-cell row-major tokenization): per column `j`, carry the
  observed-value set `{x_ij : R_ij=1}` + the mask column `R_·j` + a column id; the target column `t` is
  flagged. New module `column_batching.py`.

### 1.1 Per-column φ distribution encoder (load-bearing; Stage-0 proven)
- **One job:** map a column's observed values (a variable-length, **row-permutation-invariant** set) to a
  fixed-width **distribution embedding** that *preserves order statistics*.
- **Realization (Stage-0 design):** standardize values **within observed** (scale-invariant) → per-value
  MLP `h: R→R^m` → **masked-quantile pooling** (`Q` quantiles + max; reuse the tested
  `distributional_stream.masked_quantile_pool` on **raw** values) → small MLP `ρ` → `e_t ∈ R^{E_col}`.
- **Scope in Level 1:** applied to the **target column** (`e_t`). (φ is column-agnostic and weight-shared,
  so running it on predictors is trivial later — but Level 1 does **not** use predictor value
  distributions; that is the Level-2 conditional path.)
- Fail-loud on degenerate columns (constant / `< MIN_OBS` observed → documented safe embedding, mirroring
  `consequence_features`).

### 1.2 Mask-topology stream
- **One job:** represent the missingness **pattern structure** (the MCAR/MAR-identifiable anchor, §2 solid
  ground) — per-column miss rates, **cross-column co-missingness** (correlation of missingness
  indicators), and missingness-vs-observed-value association (point-biserial) for the target.
- **Realization:** a deterministic mask-topology feature block (revives the *concept* of v1.0
  `MissingnessFeatureExtractor`, as a current `mask_topology.py`) → LayerNorm → small MLP → `m_global`.
- **Anti-leakage:** the overall rate is matched (uninformative); the topology stream must key on
  **structure**, not rate — enforced by the blocking leakage gate (`leakage.py`).

### 1.3 Fusion
- Level-1 fusion = **`[e_t ; m_global]`** → fusion MLP → shared representation `z`.
- **The Level-1 line is explicit:** cross-column **mask** structure enters (via `m_global`); cross-column
  **value/reference** structure does **not** (that is Level 2). No predictor value distributions, no
  conditional reference, no deviation.

### 1.4 Heads (three, from `z`)
- **δ-prior head:** `z → MLP → δ-bin logits` (reuse the `DeltaBinHead` idiom) → temperature → calibrated
  ordered δ-prior. Ordering imposed by the RPS loss.
- **Detectability head:** `z → scalar ∈ [0,1]` — the (calibrated) **informativeness** of the footprint
  about δ (§3). Oracle-calibrated target where the oracle exists (§4).
- **OOD / coverage / abstention head:** a **separate** novelty score = distance of `z` (or the raw
  footprint) from the **training footprint manifold** of `P_prior` (§1.5) → abstain when above a
  held-out-calibrated threshold. **Distinct from detectability** (the flat-vs-OOD distinction, §3).

### 1.5 Explicit exclusions
- **No Level-2 conditional reference/deviation module.** No predictor-conditional reference, no
  observed-vs-reference deviation, no cross-column value modeling.
- **No metadata channel.**
- **The BERT `LacunaEncoder` is NOT the δ spine** (Gate I; §6). No row-wise cross-column attention spine.

---

## 2. The named prior (`P_prior`)

### 2.1 Representation
`P_prior` is **not** a stored density and **not** an opinion. It is the explicit semi-synthetic **training
measure**, realized operationally as *(data-generation config) + (trained weights)*:

> `P_prior = {dataset catalog} × {idiom vocabulary V} × {δ-grid + sampling weights} × {matched-rate
> regime} × {φ / architecture inductive bias}`.

The model learns `P(δ | footprint)` by training on draws from `P_prior`; on a new column it returns the
**posterior under `P_prior`**. Auditable because every factor is enumerable and changeable.

### 2.2 Recording (manifest)
Extend `run_manifest` with a **`named_prior` block**:
```
named_prior:
  dataset_catalog: [names + provenance/content hashes]
  idiom_vocabulary: [{family, params}]            # e.g. own_value_self_censoring; lod_top_coding {tau}
  delta_grid: [...]; delta_grid_weights: [...]     # the δ sampling distribution = the prior marginal
  rate_regime: {target_rate, matched: true}
  phi_config: {m, quantile_levels, E_col, ...}
  prior_marginal: P(δ-bin)                          # explicit, from delta_grid_weights
```
This makes the prior reproducible and inspectable, and is required for any Level-1 run to be interpretable.

### 2.3 Prior marginal in the output when the footprint is flat
The **prior marginal** `P(δ)` is the δ-grid sampling distribution (recorded in 2.2). A calibrated model on
a flat footprint predicts the base rate = this marginal. We make it explicit and auditable: the output
includes **`info_gain = KL(posterior δ-prior ‖ prior_marginal)`**. `info_gain ≈ 0` ⇒ prior-dominated
(footprint uninformative); large ⇒ data-updated. **Detectability (§3) is this information gain**,
calibrated against the oracle.

---

## 3. Output semantics

Per target column the model emits: **(a)** the calibrated **δ-prior** (distribution over δ-bins); **(b)**
**detectability**; **(c)** **OOD/abstain**. (Estimand × δ-prior → tipping-point is a *separate downstream
reporting layer*, §3¾, not in the network.)

- **δ-prior** — the posterior over the sensitivity parameter δ under `P_prior` given the footprint. **Not**
  a mechanism class label; never a claim of identifying δ from data alone.
- **Detectability** — the calibrated **information gain** of the footprint about δ
  (`KL(posterior ‖ prior_marginal)`), validated against the oracle Bayes-error where available. High ⇒ the
  footprint pins δ; ~0 ⇒ footprint uninformative (posterior ≈ prior).
- **Abstention** — the footprint is **outside `P_prior`'s training manifold** ⇒ the model is out of its
  competence ⇒ output the uninformative prior + **"escalate to manual review."** Distinct from low
  detectability.

**The three situations (must be kept distinct):**

| situation | δ-prior | detectability | OOD / abstain |
|---|---|---|---|
| **strong footprint, in-manifold** (LOD, sharp truncation) | **sharpened** toward true δ | **high** | no |
| **flat-likelihood, in-manifold** (own-value, matched rate) | **≈ prior marginal** (high entropy) | **low** | **no** — *confidently uncertain* (we *know* it is flat) |
| **off-manifold / novel idiom** | uninformative prior | low | **yes** — *we don't know* |

The two-headed output (detectability ⟂ OOD) is what disambiguates **known-flat** from **unknown** — the
operational core of §3 ("honestly uncertain where Molenberghs bites is *working*") and §4.3/§6.

---

## 4. Training objective

- **Main: δ-prior RPS** over ordered δ-bins (reuse `loss.rps_loss`), with **post-hoc temperature**
  calibration on val (`fit_temperature`). No binary/3-class mechanism objective as the main target.
- **Auxiliary: detectability**, **oracle-calibrated where the oracle exists.** For each training example
  with known `(idiom, dataset, δ, rate)` we compute the oracle informativeness
  (`lod_oracle` / profiled-MAR-null → Bayes-error → information measure) and regress the detectability head
  to it. Where no oracle exists, fall back to the empirical `info_gain` target. *Caveat recorded:* the
  oracle uses a fitted (Gaussian) X-model — necessary-not-sufficient — so detectability is calibrated to a
  shape-prior-dependent target; this is the best available ground truth and is reported as such.
- **Auxiliary: OOD/coverage.** Trained as a **one-class / density model on the TRAINING footprint
  distribution only** (no access to held-out families — that would leak). Threshold set on a held-out
  in-distribution split; **validated** for separation on held-out idiom families (§5). Candidate density
  models (Stage-3 choice): Mahalanobis / kNN in embedding space, or a small normalizing flow.
- **Optional auxiliary: MCAR-departure detection** — identifiable (§2 solid ground), honest to report; an
  auxiliary, never the main target.

Loss = `RPS + λ_det·detectability_reg (+ λ_ood handled separately as a density fit) (+ λ_mcar·aux)`.
λ's are config; the δ-prior RPS is primary.

---

## 5. Evaluation

All on the **held-out semi-synthetic survey ladder** (§8.1): train on some survey datasets, validate/test
on **held-out** datasets, semi-synth holes, known δ. Reuse the leave-datasets-out discipline,
`leakage` gate, `metrics`, `run_manifest`.

1. **Calibration / coverage** of the δ-prior on held-out (ECE, coverage tables) — the headline governance
   number.
2. **Out-of-family / leave-datasets-out** — calibration on unseen datasets and (the §5 headline) unseen
   idiom **forms**.
3. **Detectability-vs-oracle** — agreement (correlation/calibration) of predicted detectability with the
   oracle Bayes-error per cell.
4. **Abstention on held-out idioms** — train on family A, test OOD/abstain on held-out family B.
5. **Own-value flatness behavior** — on own-value matched-rate: δ-prior ≈ prior marginal, low
   detectability, **not** abstain (confidently uncertain).
6. **LOD/top-coding sharpness behavior** — δ-prior sharpens toward true δ, high detectability.
7. **Leakage gate** — unchanged blocking gate (`leakage.py`); any run failing it is uninterpretable.

**Pre-registered framing (non-goal, §8):** Level 1 is **not** evaluated by, nor expected to beat,
raw-ECDF on δ **discrimination**. It is evaluated on **governance** metrics: calibration,
detectability-oracle-agreement, abstention correctness, and the honest-uncertainty behaviors (5)/(6).

---

## 6. Salvage map

| component | verdict | note |
|---|---|---|
| Generators (`delta_generator`, `lod_generator`, answer sheets, matched-rate solve, `subsample_raw`) | **Keep** | supervision ground truth; architecture-agnostic |
| δ-bins / coarse-bins, **RPS loss**, temperature, **metrics** | **Keep** | output-layer machinery |
| **Leakage gate** (`leakage.py`) | **Keep** | blocking discipline |
| **Oracle** (`lod_oracle`, profiled MAR null) | **Keep + extend** | now also the **detectability training target** source |
| **Manifest** (`run_manifest`) | **Keep + extend** | add the `named_prior` block (§2.2) |
| Example-source interface / batching | **Keep concept, adapt** | new **column-major** `column_batching.py` |
| `consequence_features` (fixed 17-d value-ECDF) | **Keep** | baseline / φ sanity / deterministic fallback |
| `distributional_stream.masked_quantile_pool` | **Keep (reuse in φ)** | the order-statistic pooling primitive |
| `LacunaEncoder` (BERT backbone) | **Remove as the δ spine** | Gate I. May survive elsewhere; **not** in the L1 δ model |
| `DeltaPriorModel`, `TargetConditionedDeltaModel`, `rep_ecdf_pooling` (encoder-coupled) | **Remove from δ path** | superseded by the φ spine (pooling primitive reused) |
| `transfer_features` | **Remove** | failed the no-training gate (0.523) |
| MoE / reconstruction / 3-class / decision (`assembly.py`) | **Stay out of δ path** | already bypassed |
| **New:** `column_phi.py`, `mask_topology.py`, fusion+heads, `detectability.py`, `coverage_ood.py` | **Rewrite (new)** | the Level-1 spine + heads |

**Explicit: the BERT backbone does not return as the δ spine.** A "rewrite" here = a **representation
rewrite on the retained scaffold**, not a from-zero project.

---

## 7. Staged build plan (each stage gated; tests per Rule 7; ≤500 LOC/module)

- **Stage 1 — φ spine + δ-prior head + column-major data path.** Milestone: held-out δ-prior calibrated,
  OOF discrimination ≈ Stage-0/raw-ECDF on LOD, own-value ≈ prior marginal, leakage-gated, manifest with
  `named_prior`. (Confirms the spine reproduces Stage 0 inside the full pipeline.)
- **Stage 2 — mask-topology stream + fusion + detectability head (oracle-calibrated).** Milestone:
  detectability tracks the oracle; optional MCAR-departure auxiliary reported.
- **Stage 3 — OOD/coverage/abstention head.** Milestone: abstains on held-out idiom families; stays
  *confidently uncertain* (no abstain) on in-manifold flat (own-value).
- **Stage 4 — calibration/reporting integration.** Temperature, coverage tables, `named_prior` +
  detectability + abstention surfaced in the manifest/report; interface to the **separate** downstream
  estimand × δ-prior → tipping-point reporting layer (§3¾).
- **Level 2 conditional reference/deviation — explicitly deferred** to a separate Gate-II decision.

---

## 8. Non-goals (binding)

- **No metadata channel yet** (§8.3 deferred, gated).
- **No conditional reference/deviation yet** (Level 2; Gate-II).
- **No attempt to identify δ from real natural missingness** — semi-synthetic supervision only; real data
  = face-validity/plausibility only (§4.4).
- **No claim that Level 1 beats raw-ECDF on discrimination.** Level 1's value is **governance**:
  calibration, detectability, abstention, and honest uncertainty. φ ≈ raw-ECDF on the footprint is expected
  and acceptable (Stage 0).
- **No 3-class mechanism-classification objective** as the main target.
- **No claim to beat non-identifiability** (§2 theorem).

---

## 9. Data sufficiency / prior richness

**Why this is first-class, not a footnote.** `P_prior` *includes* `{real survey datasets}`, so the
**catalog is part of the scientific prior**, not merely training data. The posterior Lacuna reports is
only as meaningful as the prior is rich. The governing question is therefore not "can the model train?"
but **"is the empirical prior rich enough to justify the posterior Lacuna reports?"** This section
specifies the analysis (read-only inventory + a learning-curve diagnostic) that must run **alongside
Stage 1** and that **gates acquisition-vs-architecture resource allocation** before Stages 2–4.

### 9.A Three evidence standards (which question are we answering?)
Data sufficiency is meaningless without a **standard**. Three, in increasing demand:

| standard | goal | sufficient evidence |
|---|---|---|
| **1. Proof-of-concept / dissertation** | establish the program is **viable** | signal exists for some idioms; the right representation recovers more than the old backbone *(already shown — §8 / Stage 0)*; Level-1 reproduces the column-primary signal **in-pipeline**; calibration / detectability / abstention behave correctly; **performance improves with catalog diversity**; the learning curve suggests more data would plausibly help. **Needs enough diverse datasets to show a credible positive trend — not hundreds.** |
| **2. Grant-proposal** | justify funding for larger acquisition / generation | **positive learning curve** as real catalog grows; evidence performance is **data-limited, not conceptually flat**; an estimate of how much more is needed; named sources; a plan to **preserve natural missingness (role A) while supervising only on semi-synthetic (role B)**. **A credible "there is a there there" slope — not solving Lacuna.** |
| **3. Deployment / commercial** | robust on arbitrary survey data | dozens–hundreds of distinct surveys; natural-missingness ingestion; broad idiom vocabulary; cross-domain calibration; OOD/abstention validation; possibly synthetic-real generation; external expert face-validity. **Beyond the dissertation MVP.** |

**Binding reframe.** The Level-1 data-sufficiency program asks the **dissertation/grant** question —
*"do we have enough to demonstrate a credible scaling trend and justify the next acquisition phase?"* —
**not** the deployment question *"do we already have enough for deployment?"* This makes the data question
a **scientific result (the scaling curve)**, not a blocker.

- **Dissertation-level success:** Level-1 works on the current (9 genuine) catalog well enough to
  **reproduce the Stage-0 column-primary signal in-pipeline**, **and** performance **improves** as the
  catalog becomes more diverse.
- **Grant-level success:** the **slope of the learning curve remains positive at the edge of the current
  catalog (N=9)**, implying broader survey-data acquisition is a scientifically justified next investment.

### 9.0 Three data roles (binding) — natural missingness is NOT supervised δ-data
Lacuna's scientific defensibility rests on **every supervised example having a known answer sheet**
(known dataset, imposed mechanism, imposed δ, true label). Natural missingness has **none** of these —
unknown mechanism, unknown δ, no counterfactual — it **is** the non-identifiability problem Lacuna exists
to govern. Therefore natural missingness must **never** enter the supervised δ-training/eval stream. Three
distinct data roles, kept separate:

| role | what it is | δ ground truth? | use |
|---|---|---|---|
| **A. Natural survey data** | real missingness, raw | **No** (unknowable) | **manifold (M2) validation, OOD/coverage validation, detectability/footprint-plausibility validation, future survey-world generation** — face-validity only (§4.4); **never supervised δ-data** |
| **B. Complete projection** | natural missingness removed/resolved; the base for semi-synthetic hole-punching | **Yes** (after holes imposed) | **the SOLE supervised δ-training/eval corpus** (the validation ladder) |
| **C. Synthetic survey worlds** | generated from real survey structure; imposed mechanism + δ | **Yes** (imposed) | **large-scale training amplifier** (domain randomization); validated only on held-out real A/B; **never a validation substitute** |

**Binding rules:**
- **Ingestion is dual-output:** it **preserves** natural missingness into the **A-archive** *and*
  separately emits the **B complete-projection** for supervision. Two representations, one source.
- **The supervised stream = B (+ C once validated). Never A.**
- **A is used only** for manifold/OOD/detectability validation and generation — never as labeled δ.

*This corrects an earlier conflation in this section: "preserve natural missingness" means **archive it
for validation (role A)**, not "train on it." Preserve — yes; supervise — no.*

### 9.1 Current catalog inventory (measured, read-only — `cat.list_datasets()` filtered to `survey_*`)
**12 survey datasets, 114 columns, 65 targetable (cardinality ≥ 10), total n ≈ 62,228** (min 82, median
≈ 2,485, max 28,155).

| dataset | n | d | targetable | card range | domain (inferred) | genuine survey? |
|---|---|---|---|---|---|---|
| survey_cps1988 | 28,155 | 3 | 3 | 19..5970 | labor/income (CPS) | yes |
| survey_yrbss | 11,522 | 5 | 2 | 7..238 | health-risk behavior | yes |
| survey_computers | 6,259 | 7 | 4 | 3..808 | product pricing | **no (product data)** |
| survey_psid7682 | 4,165 | 6 | 5 | 7..1017 | labor/income (PSID) | yes |
| survey_workinghours | 3,382 | 11 | 5 | 2..1102 | labor/income | yes |
| survey_chile | 2,590 | 4 | 3 | 7..2012 | political attitudes | yes |
| survey_hmda | 2,380 | 6 | 4 | 4..1537 | mortgage/finance | yes |
| survey_bfi | 2,236 | 28 | 1 | 2..59 | psychology (Big-Five Likert) | yes |
| survey_psid1976 | 753 | 17 | 12 | 4..697 | labor/income (PSID) | yes |
| survey_cps1985 | 534 | 4 | 4 | 17..238 | labor/income (CPS) | yes |
| survey_survey | 170 | 5 | 5 | 41..73 | student measurements (MASS) | **no (teaching set)** |
| survey_cars93 | 82 | 18 | 17 | 3..74 | automotive specs | **no (product data)** |

- **Natural missingness = 0.000 across ALL 12.** The ingestion path drops NaNs and yields complete
  matrices. **Consequence (a recorded limitation):** natural survey missingness — the real item-
  nonresponse patterns that are themselves part of the survey manifold — is **not preserved**. The only
  missingness studied is our matched-rate semi-synthetic holes. The §3½ **M2 manifold check** (do real
  masks lie in the span of our generated footprints?) **cannot currently be run** because real masks were
  discarded at load. *In §9.0 terms: the current ingestion emits only representation **B** and **discards
  representation A**. The fix is to **also archive A** (for manifold/OOD/detectability validation) — **not**
  to route natural missingness into supervision.*
- **Column types** are overwhelmingly **right-skewed positive continuous economic variables** (wage,
  income, price, hours) + ages / counts / education-years; sparse on categorical/ordinal **attitude
  (Likert)** items (only `bfi`, low-cardinality → just 1 targetable), and **no** skip-logic-bearing or
  social-desirability-sensitive items.
- **Sample sizes** are highly uneven; **4 of 12 have < 800 rows** (cars93 82, survey 170, cps1985 534,
  psid1976 753).

### 9.2 Effective diversity (the real concern)
- **These are not 12 independent survey worlds.** Five are labor/income economics (cps1985, cps1988,
  psid1976, psid7682, workinghours) sharing near-identical column vocabularies (wage/education/experience/
  age/hours) → high redundancy. **Effective distinct "worlds" ≈ 6–7**, and three of the twelve
  (cars93, computers, survey) are **not genuine surveys**.
- **Domain gaps (major survey families entirely absent):** demographic census (ACS/IPUMS), general social
  attitudes (GSS/ANES/ESS/WVS), comprehensive health (NHANES/BRFSS), education surveys.
- **Idiom-vocabulary gaps:** the §3½ survey-idiom vocabulary the prior is meant to span — skip logic,
  social-desirability censoring, LOD on lab values, income top-coding, DK/refusal coding — is barely
  represented (we *impose* LOD/own-value synthetically, but the X-tables that would make those footprints
  realistic, e.g. income top-coding in CPS/ACS, lab LOD in NHANES, are absent).
- **Verdict:** the catalog is **small, redundant, domain-skewed (labor-econ), natural-missingness-
  stripped, and contaminated with non-survey tables.** For a *named prior claiming to represent the
  survey-missingness manifold*, this is **thin**.

### 9.3 Learning-curve / scaling plan — *the deliverable scientific result (§9.A)*
- **Plan:** train/eval Level-0/Level-1 as a function of **#base training datasets at N ∈ {4, 8, 9}**
  (9 = the genuine maximum after dropping the 3 contaminants), with **many seeds** (leave-datasets-out is
  high-variance at small catalog), on a fixed held-out test set. The curve **continues to 12, 16, 32 … as
  acquired domains are added** — reaching N>9 *requires* acquisition, which is exactly what the curve is
  meant to justify (grant standard).
- **Report (per N, with seed/split variance):**
  - **δ-prior calibration / coverage** (ECE, coverage tables);
  - **LOD/top-coding sharpness** (binary δ0-vs-δ2.5 OOF AUC; δ-prior concentration);
  - **own-value prior-dominated behavior** (δ-prior ≈ `prior_marginal`, low info-gain);
  - **detectability-vs-oracle agreement** *(matures at Stage 2)*;
  - **OOD / abstention** on held-out idiom families *(matures at Stage 3)*;
  - **variance across seeds/splits** (essential at small N).
  *(At Stage 1 the available subset is calibration + LOD AUC + own-value flatness + variance; the
  detectability and OOD curves come online as Stages 2–3 land.)*
- **Interpretation (the §9.7 decision rule):** slope clearly **positive at N=9** → **data-limited →
  acquire** (grant-level success). Apparent **saturation** → architecture/prior-formulation *candidate* —
  **but binding caveat:** saturation on a *redundant* 9 is **not** evidence that *diverse* data would not
  help; **must not be read as an architecture verdict** without ≥1 genuinely new **domain** added.

### 9.4 Dataset-acquisition threshold (explicit estimate)
- **A handful more is not enough.** To make leave-datasets-out meaningful (hold out whole datasets while
  training on a *diverse* remainder) **and** to span the idiom/domain vocabulary, we need **dozens now**
  — target **≥ 30–50 genuinely distinct survey datasets** across health / demographics / income / labor /
  education / social-attitudes / risk-behavior / political — scaling toward **hundreds** (survey *waves*:
  NHANES cycles, BRFSS years, GSS waves, IPUMS extracts) for a defensible manifold prior.
- **Stated plainly: dozens near-term, hundreds for a mature prior.** This is a precondition for the
  posterior to be scientifically meaningful — not gold-plating.

### 9.5 Candidate data sources (all public; legally downloadable)
| source | domain it adds | idiom realism it adds |
|---|---|---|
| **NHANES** waves | health / nutrition / lab | **LOD on lab values** (real), item nonresponse |
| **BRFSS** years | health-risk behavior (large) | categorical/ordinal nonresponse |
| **GSS** waves | social attitudes | **skip logic**, social-desirability, DK/refusal |
| **CPS / ACS / IPUMS** extracts | demographics / labor / income (huge) | **income top-coding** (real) |
| **Add Health** | adolescent health/behavior | sensitive-item censoring |
| **ANES** | political attitudes | income/vote nonresponse |
| **ESS / WVS** | cross-national social values | Likert, DK/refusal coding |

*Formatting requirement:* the pipeline must **preserve natural missingness** (unlike the current
ingestion, §9.1) — this simultaneously fixes the M2 gap and supplies realistic top-coding/LOD X-tables.

### 9.6 Synthetic-real dataset generation (later data-expansion arc) — role C (§9.0)
- Train a generative model over real survey columns/tables → sample realistic synthetic survey **X** →
  impose semi-synthetic missingness → train Lacuna on generated worlds → **validate only on held-out
  REAL datasets.**
- **Binding guardrail:** generated X is **domain randomization** (expands the training factor of
  `P_prior`), **not proof.** It can **never** replace held-out real validation (§4.4, §8.1). The held-out
  **real** ladder remains the sole arbiter.

### 9.7 Decision rule (explicit)
1. Level-1 performance **improves predictably with more REAL datasets** → **prioritize acquisition.**
2. It **saturates quickly on the current catalog** → architecture/prior-formulation is a candidate
   bottleneck — **but** confirm with ≥1 genuinely new **domain** before concluding (the §9.3 redundancy
   caveat); saturation on a redundant 12 is not an architecture verdict.
3. Generated survey-X improves **generated-X** but **not held-out REAL-X** → **reject as generator
   overfitting** (domain randomization that did not transfer).
4. Generated survey-X improves **held-out REAL-X** → the generator is a **validated training amplifier**;
   adopt it as a `P_prior` training factor (still **not** a validation substitute).

### 9.8 Verdict — answered at the right standard (§9.A)
**Do not ask the deployment question of a dissertation MVP.** At the **deployment** standard the current
catalog is **not** rich enough (dozens → hundreds, natural-missingness-preserving, broad idioms — §9.4) —
but that is **not the question Level 1 answers**. At the **dissertation / grant** standard the operative
bar is the **scaling trend**, and the current 9-genuine catalog is **sufficient to attempt it**:
- **Dissertation:** reproduce the Stage-0 signal in-pipeline **and** show performance improves with
  diversity (the 4→8→9 curve).
- **Grant:** the curve's **slope is still positive at N=9**, justifying acquisition as the next
  investment.

The learning curve (§9.3) is therefore **the deliverable scientific result**, not a gate the project must
pass before proceeding. The build (Stages 1–4) and the data program (9.3 → 9.4 → 9.5) run **in parallel**;
the curve's slope — not a deployment-readiness check — allocates effort between *acquire* and
*architecture* (§9.7).

---

## Decisions (resolved — PI, 2026-06-05)
1. **φ capacity** — **APPROVED** Stage-0 default: `m≈16`, 12 quantiles + max, moderate `E_col`. *Stage-1
   goal is to **reproduce** the known column-primary signal in-pipeline, not a capacity sweep.*
2. **Detectability target** — **APPROVED** oracle information-gain where available (necessary-not-
   sufficient caveat recorded); simpler fallback elsewhere, **labeled as weaker**. Detectability =
   "informativeness of the footprint," never a mechanism classifier.
3. **OOD density model** — **APPROVED** Mahalanobis/kNN first (inspectable); richer density models only if
   it fails to separate held-out families. No normalizing flow to start.
4. **MCAR-departure auxiliary** — **APPROVED** for Stage 2, auxiliary only.
5. **Data-sufficiency program (§9)** — **APPROVED** all three: (a) run the learning-curve diagnostic
   (multi-seed) on the current catalog; (b) acquisition program approved in principle (dozens → hundreds);
   (c) **drop the 3 non-survey contaminants** (cars93, computers, survey) from the survey catalog. Plus the
   **§9.0 correction**: dual-output ingestion **archives natural missingness (role A)** and emits the
   **complete projection (role B)**; **natural missingness is validation/generation data only, never
   supervised δ-data.** *Note: dropping the 3 contaminants leaves **9 genuine surveys**, so the
   learning-curve runs **4→8→9** (we cannot reach 12 with genuine surveys) — which **sharpens** the
   acquisition case (§9.4).*

**Status: Level-1 design approved. Next deliverable = the Stage-1 implementation spec
(`PROPOSAL-Stage1-implementation-spec.md`). No code until that spec is approved.**
