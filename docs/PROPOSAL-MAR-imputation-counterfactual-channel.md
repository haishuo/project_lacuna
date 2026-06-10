# MAR-Imputation Counterfactual Channel — Specification-Only Proposal

*Proposal document — **no implementation, no runs, no architecture change** (binding; PI 2026-06-08).
A possible future Lacuna signal, spec'd before any decision to build. Governed by `NORTH-STAR.md`
(never train on natural missingness; §2 identification; comparison-class discipline) and
`MASTER-lacuna-survey-architecture.md` (§5½ detectability split; §7 data roles). Engages three prior
results that bound this idea: the demoted Level-2 reference/deviation
(`INFERENCE-OBJECT-CRITIQUE.md`/`ARCHITECTURE-OBJECT-revised.md`), the failed predictor-referencing
gate (`transfer_features`, OOF AUC 0.523), and the oracle-vs-channel divergence
(`CONSOLIDATION-MEMO-detectability-target.md`).*

---

## 0. The idea, scoped honestly

In the **semi-synthetic role-B setting only** — complete table, holes punched by a known mechanism at
known δ (the answer sheet) — fit a **MAR-style imputer on observed data only**, impute the punched-out
cells, and compare imputations to the **known true hidden values**. The summarized error/bias is a
diagnostic channel. The scientific question:

> **Does the failure mode of a MAR imputer carry information about the missingness mechanism or δ?**

This is a **counterfactual MAR baseline** ("what would a MAR-faithful analyst's imputations have gotten
wrong here?"), **not** a test that the data are MAR, and **never** a use of natural missingness as labels.
The truth-based quantity exists **only where the answer sheet exists**; at deployment, true imputation
error is unavailable **by construction** — so this cannot naively become a runtime feature (§3–§4).

**Where it sits conceptually.** It inserts a new rung in the three-levels hierarchy
(oracle / feature / learned): the oracle knows the **true mechanism family**; this channel knows the
**true hidden values** at eval but only a generic MAR predictor; the learned channel knows neither. The
channel therefore measures *how much of the oracle-visible signal is recoverable without mechanism
knowledge but with truth access* — a semi-parametric "middle level" that is scientifically useful
**even if it never becomes a runtime feature** (it sharpens the lab-coat-fraction decomposition).

## 1. Exact quantities computed under semi-synthetic truth (Q1)

Per semi-synthetic example (dataset, target column *t*, mechanism, δ, matched rate), with punched index
set *M* (|M| ≈ rate·n), truth `x_j`, imputation `x̂_j`, and (where the imputer supplies one) a predictive
distribution `F̂_j`; all values standardized within the observed column (φ convention):

**Point-error family**
- **Signed mean bias** `B = mean_{j∈M}(x̂_j − x_j)` — the primary quantity; under self-censoring the truth
  sits above the MAR-conditional prediction, so `B < 0` with |B| increasing in δ.
- `MAE`, `RMSE` (magnitude; needed for the §6 unpredictability control, not themselves MNAR evidence).
- **Localized bias**: `B` restricted to truth-deciles (top-coding concentrates bias above τ; own-value
  spreads it) — the *localization profile* is a candidate mechanism-family signature.

**Distributional family**
- **W₁(imputed ‖ truth)** over *M* (how wrong the imputed *distribution* is, beyond its mean).
- **PIT shift**: `u_j = F̂_j(x_j)` — under a well-specified MAR imputer the PIT values are ≈ Uniform(0,1);
  under self-censoring the truth concentrates in the upper predictive tail (mass of `u_j` near 1); under
  top-coding, a near-degenerate spike. The **PIT histogram distance from uniform** is arguably the
  cleanest single mechanism signature (calibration-form, imputer-scale-free).
- **Interval coverage** of `F̂_j` central intervals at punched cells.

**The paired MCAR control (mandatory, the §6 confound-killer)**
For the *same example*, additionally punch a **matched-rate MCAR mask** on the same column (disjoint
cells where feasible), run the *same* imputer, compute the *same* statistics → `B_mcar`, `PIT_mcar`, …
The channel's signals are the **paired differences** (`ΔB = B_mech − B_mcar`, etc.). This transplants the
profiled-null/matched-rate discipline to imputation: intrinsic unpredictability, imputer weakness, and
rate effects cancel in the pair; what survives is the **selection-induced** component.

**Imputer class (named, simple, auditable — no GAN first):**
1. linear conditional-mean (+ Gaussian residual for `F̂`),
2. MICE-style chained equations,
3. random forest (quantile forest for `F̂`),
4. gradient boosting,
5. a *simple* neural imputer (MLP, fixed HP) — included only to test imputer-class sensitivity.

All fit on **observed rows only** of the example (which are themselves selection-tilted under MNAR — that
is part of what the counterfactual measures, not a bug). One fixed configuration each; no tuning.

## 2. What counts as evidence against MAR (Q2)

Under semi-synthetic truth, for a named imputer class **I**:

> The example's missingness is **inconsistent with MAR-as-operationalized-by-I** iff the paired
> differences (`ΔB`, ΔPIT-shift) are significantly nonzero — i.e., the imputer's failure at the punched
> cells **exceeds its own matched-rate MCAR failure on the same column**, in a direction/profile
> consistent with a selection effect.

Binding comparison-class discipline (MASTER §3): the claim is never "not MAR" simpliciter — it is
**"inconsistent with the best imputer in class I trained on observed data."** A weak class I makes the
claim cheap; the spec therefore reports results **per imputer** and treats only signals stable across the
class as mechanism evidence. (This makes the imputer class the operational *plausible MAR class* — the
M3 open question in yet another guise, and it must be stated in every result.)

Note what is *not* evidence: large `RMSE` alone (unpredictable column), bias present equally in the MCAR
control (imputer misspecification), or PIT non-uniformity present in the control (bad `F̂` calibration).

## 3. Available only under training/evaluation truth (Q3)

Everything in §1 that touches `x_j` (truth at punched cells): signed bias, localized bias, W₁, PIT shift,
interval coverage at punched cells, and all paired differences. These exist **only** where we punched the
holes ourselves — role-B semi-synthetic cells. They may be used as: eval-time instrumentation,
study targets, and validation anchors for proxies. They may **never** appear in any runtime path, and
natural missingness never supplies a substitute truth (Role-A discipline unchanged).

## 4. Runtime-available proxies — and their hard ceiling (Q4)

At deployment the input is a real table with natural missingness; computable without truth:

1. **Imputer predictive uncertainty** at the actually-missing cells (spread of `F̂_j`).
2. **Observed-holdout reconstruction residual**: punch **our own MCAR holes into *observed* cells**
   (allowed — we impose them), impute, score against the known held-out values → the column's intrinsic
   predictability and the imputer's calibration *on this table*. This is the runtime analogue of the §1
   MCAR control and the strongest proxy idea.
3. **Distribution shift** between imputations at real-missing positions and at our MCAR-holdout positions
   (or observed values), conditional on predictors.
4. **Posterior shift**: change in the δ-prior/model outputs computed before vs after imputation.

**The hard ceiling (the Level-2 lesson, restated):** every proxy above is a function of observed data
only. Under MAR, missing cells legitimately differ in predictor profile, and the imputer conditions on
predictors — so proxies 1–3 can detect *risk* (unpredictability, calibration problems, profile shift) but
**cannot identify the MNAR deviation itself**: the truth at missing positions is exactly what observed
data cannot supply. A runtime proxy can therefore become a *mechanism/δ* signal **only as a learned,
P_prior-calibrated channel**: train/validate the mapping proxy→(truth-based ΔB / δ) on semi-synthetic
cells, held-out leave-domain-out — inheriting **all** the transfer caveats this project has documented,
and the E-JUSTIFY/E-FALSIFY lesson in particular: **proxy-vs-truth rank agreement must be tested OOF
before any runtime claim; assume divergence until shown otherwise** (the analogous assumption just failed
for oracle-vs-I_gain at Spearman ≈ 0).

## 5. Does it add information beyond φ-regime geometry and the δ-prior? (Q5)

The channel's only novel content is **conditional** (target-given-predictors) structure — φ reads the
observed *marginal*. Conditional/predictor-referencing is precisely where the project has a documented
failure (`transfer_features` OOF 0.523 — the no-training LR gate). So the value-add test is pre-specified
and cheap-first:

- **A. Truth-level increment (eval question):** on semi-synthetic cells, leave-domain-out: does
  {ΔB, ΔPIT, localization profile} predict (mechanism family, δ-bin) **beyond** the φ/raw-ECDF feature
  baseline — measured as LR-level AUC/RPS increment, same gate discipline as `transfer_features`
  (pre-registered threshold; failure ⇒ the channel duplicates the marginal footprint).
- **B. Proxy-level increment (runtime question, only if A passes):** same test with **runtime-computable
  proxies only** (§4.1–4.4) replacing truth-based quantities.
- **C. Orthogonality check:** correlation of the channel's signals with the D2 regime coordinates and
  with φ-features — if the channel is a re-parameterization of column shape, A may pass while C shows
  redundancy.

Honest prior from existing evidence: at the *truth* level the channel almost certainly carries δ
information for detectable idioms (the signed bias is nearly the definition of the selection shift) — A's
real content is the **own-value** column and the **increment over φ**, not top-coding's headline. At the
*proxy* level, the prior is pessimistic (transfer_features; E-study) and the burden of proof is on B.

## 6. Failure modes (Q6)

1. **Intrinsic unpredictability confound** (the PI's example): large error because R²(t|predictors) is
   low, not because MNAR. *Control:* the paired matched-rate MCAR punch (§1) — all headline quantities
   are paired differences; raw `RMSE` is never evidence.
2. **Imputer misspecification masquerading as MNAR:** a bad imputer is biased everywhere. *Control:* the
   MCAR pair (cancels), plus the multi-imputer stability requirement (§2).
3. **Proxy absorption:** high-R² predictors let the MAR imputer partially mimic the selection (the P2.2
   proxy-sweep concern) → ΔB shrinks with R². *Handling:* report per R²-stratum (the existing
   stratification machinery); a channel that only works at low R² where everything is noisy is not useful.
4. **Selection-tilted training set:** the imputer itself is fit on MNAR-selected observed rows; its bias
   conflates "selection in training" with "selection at prediction." Not a bug (the counterfactual is
   *defined* against the observed-data analyst) but a reason the channel measures the **composite**
   MAR-analyst failure, not a clean δ functional.
5. **Spectrum redundancy:** the channel may simply reproduce the known idiom spectrum (top-coding huge,
   own-value small) — informative-looking but adding nothing beyond φ (§5.C).
6. **Small-|M| noise:** at n=384, rate 0.3 ⇒ ~115 punched cells; paired differences at small δ may be
   underpowered — power must be reported, not assumed.
7. **The runtime mirage (the cardinal failure):** quietly promoting a truth-validated quantity to a
   runtime feature without the §4 OOF proxy gate — exactly the category of error the detectability head
   review just caught. Structurally forbidden by this spec.
8. **Multiple-imputation subtlety:** SCF-style multiply-imputed sources must never be used as "truth"
   (their values are themselves imputations); truth = actually-observed role-B values only.

## 7. Relation to the rejected Level-2 reference/deviation (Q7)

They are the **same object at different epistemic stations**. Level-2's "observed − MAR-reference
deviation" was demoted because, on one real dataset, the MAR reference is **not identified** — a
reference fit on observed data yields deviation ≡ 0 without a prior (`INFERENCE-OBJECT-CRITIQUE.md`).
This channel is **Level-2 made honest**:

- At **eval**, the answer sheet supplies the identification the data cannot: the "deviation" is computed
  against *truth*, not against an unidentifiable reference. The deviation object is therefore legitimate
  — but only inside `P_prior`.
- At **runtime**, the original Level-2 objection returns in full force (§4's hard ceiling): no truth, no
  identified deviation; only class-relative, P_prior-calibrated proxies.
- The earlier *empirical* failure of predictor-referencing (`transfer_features` 0.523 OOF) is the direct
  ancestor of §5's gate: this proposal does not assume the conditional channel transfers — it
  pre-registers the same test that killed its ancestor.

So: this is not a revival of Level-2 as an architecture module; it is a **measurement program** that
could, at most, *earn* a future Level-2-like module through gates its ancestor failed.

## 8. What would justify building it (Q8)

Staged, each gate pre-registered before its runs (no gate is run by this document):

- **G1 — Truth-level information (eval-only justification):** on semi-synthetic role-B cells,
  leave-domain-out, the paired-difference channel (ΔB/ΔPIT/localization) separates {MCAR-control,
  own-value, top-coding} × δ-bins at the LR level **and** adds a pre-registered increment over the
  φ/raw-ECDF baseline (§5.A), stable across ≥2 imputer classes. **Passing G1 alone justifies building the
  channel as eval-time instrumentation** — a mechanism-knowledge-free middle level between oracle and
  learned channel, enriching the lab-coat-fraction decomposition. *(Cheap: no new architecture; imputers +
  statistics on the existing corpus/generators.)*
- **G2 — Proxy transfer (runtime relevance):** ≥1 runtime-computable proxy (§4) tracks the truth-based
  quantity held-out (pre-registered rank-agreement threshold, OOF, both idioms, coverage-gated). Failing
  G2 caps the channel at eval-only — a recordable, useful result, not a defeat.
- **G3 — Runtime admission (separate review):** only after G1+G2, a runtime feature/channel proposal goes
  through the same kind of pre-registered architectural review the detectability head just failed —
  targeted at **deployed-channel reliability** quantities (§5½ MASTER), never at truth quantities the
  runtime cannot see, and never the gate role.

**Not** justification: top-coding-only success (already detectable by φ); truth-level success without the
φ-increment (redundant); any result on natural missingness (no labels exist there, by design).

---

*No code, runs, or architecture changes are proposed or scheduled. This document defines the channel, its
truth-only quantities, its paired-MCAR control, its runtime ceiling, the value-add tests it must pass
(G1–G3), its failure modes, and its precise relationship to the demoted Level-2 deviation. Whether to run
G1 is a PI decision.*
