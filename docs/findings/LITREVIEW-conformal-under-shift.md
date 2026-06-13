# Literature review: conformal prediction under distribution shift — and whether the C2ST realism gap can certify coverage transport

**Date:** 2026-06-13
**Branch:** `p2/delta-prior-rearchitecture`
**Status:** READ-ONLY literature task. Deepens `docs/LITERATURE-REVIEW.md` §3 (previously one source: Gibbs 2023).
**Search tool:** the user's authenticated Consensus connector (full result sets, Semantic Scholar / PubMed / Scopus / arXiv). NOT the capped bio-research plugin. See [[consensus-search-routing]].

---

## 0. Why this is the load-bearing question

Lacuna's deployable claim (`PROPOSAL-recoverability-reformulation.md` §4) is a per-column ×
per-functional **three-factor bound** — `gap ≈ rate × residual-spread × δ` — *certified to hold at
e.g. 95% coverage*. The certification is performed on **plasmode** (real X, injected known
missingness mechanism) and then applied to **real** columns. Conformal prediction delivers
finite-sample coverage **under exchangeability**. Plasmode→real is a **distribution shift** that
breaks exchangeability — and, worse, it is a shift in the *mask-generating mechanism* (the δ /
value-dependence), not merely in X. So the entire soundness of "certified coverage" rests on
**conformal-prediction-under-shift**, and specifically on whether the coverage gap can be tied to a
**measurable** quantity — the generator-realism gap, which the charter (§6) proposes to measure via
a **Classifier Two-Sample Test (C2ST)**.

The current `LITERATURE-REVIEW.md` §3 asserts this link closes the argument *rigorously*: "the
coverage guarantee transports iff the plasmode→real shift is small, which is exactly the measurable
layers 1–3 of generator realism." **This document tests that assertion against the literature. The
short answer: the link is real in *direction* but the assertion as written is too strong — a C2ST
certifies in the *falsification* direction, not the *certification* direction.** Details below.

---

## 1. Weighted conformal prediction under covariate shift

The foundational result is **Tibshirani, Foygel Barber, Candès & Ramdas (2019)**,
[Conformal Prediction Under Covariate Shift](https://consensus.app/papers/details/b265ce36a632506e9976a8d30fde306a/?utm_source=claude_desktop)
(662 cites). When the test covariate distribution differs from calibration but **the conditional
`P(Y|X)` is unchanged** ("covariate shift"), exact coverage is restored by re-weighting each
calibration point by the **likelihood ratio** `w(x) = dP_test/dP_train`, normalized over the
calibration+test pool. This is the "weighted exchangeability" generalization. Two assumptions are
load-bearing: (i) pure covariate shift (no conditional/concept change), and (ii) `w(x)` known or
**estimable from unlabeled test covariates**.

Successors relax/operationalize (ii), because estimating `w` is where the guarantee leaks:

- **Yang, Kuchibhotla & Tchetgen Tchetgen (2022, JRSS-B)**,
  [Doubly robust calibration of prediction sets under covariate shift](https://consensus.app/papers/details/4d58157064385987908afad1be4554a4/?utm_source=claude_desktop)
  (60 cites) — semiparametric-efficient calibration; coverage protected if *either* the shift model
  *or* an outcome model is correct. Explicitly frames covariate shift as the MAR analogue (directly
  relevant: Lacuna *is* a missing-data problem).
- **Jin, Qin et al. (2024, JASA)**,
  [Distribution-Free Prediction Intervals Under Covariate Shift](https://consensus.app/papers/details/ede4c918f39a531f9d70e4d3403afcd5/?utm_source=claude_desktop)
  — resampling/pivotal approach that avoids data-splitting; double robustness.
- **Bhattacharyya & Barber (2024, EJS)**,
  [Group-weighted conformal prediction](https://consensus.app/papers/details/5faec6a73c9b5ef6b4f12f2b209dfc9b/?utm_source=claude_desktop)
  — when the shift is determined by a *finite set of groups* (e.g. stratified sampling), the
  likelihood-ratio-estimation-error penalty can be drastically reduced. **Relevant to Lacuna:** if
  the plasmode→real shift can be discretized by reference-class / instrument, group-weighting tightens
  the guarantee.
- **Wang & Barber (2026)**,
  [Weight Clipping for Robust Conformal Inference under Unbounded Covariate Shifts](https://consensus.app/papers/details/7c174ae171f25f45811e0c13339a76b5/?utm_source=claude_desktop)
  — when `w` is unbounded or learned, WCP undercovers; clipped density-ratio fitting gives *bounded
  expected undercoverage*, **correctable by inflating the target level by a data-estimable amount.**
  This "estimate the slack, inflate the level" pattern recurs and is the practical template Lacuna
  should adopt.
- **Joshi et al. (2025)**,
  [Conformal Inference under High-Dimensional Covariate Shifts via Likelihood-Ratio Regularization](https://consensus.app/papers/details/fec448c710335a4e8e20b426f980f7ab/?utm_source=claude_desktop)
  — avoids explicit `w` estimation in high dimensions (relevant given wide survey X).
- **Fannjiang et al. (2022, PNAS)**,
  [Conformal prediction under feedback covariate shift for biomolecular design](https://consensus.app/papers/details/0a9a8070f1615309833d4327c8c84ce8/?utm_source=claude_desktop)
  (79 cites) — handles shift where test inputs *depend on* training data; finite-sample valid. A
  template for "design/optimize then certify" loops.

**Takeaway for Q1.** Mature, exact machinery exists for *covariate* shift. The catch for Lacuna is
that plasmode→real is **not** pure covariate shift (Q2), so weighted CP is a component, not the whole
answer — and the residual leak is always the quality of the likelihood-ratio estimate.

---

## 2. Beyond covariate shift: label/concept/joint shift, and impossibility

Lacuna's shift lives in the **conditional** law (the real mask mechanism / δ differs from the
injected one), which is exactly the case weighted CP does *not* cover. Two literatures bear on this.

**2a. Non-exchangeable conformal + the coverage-gap-by-divergence bound.**
**Barber, Candès, Ramdas & Tibshirani (2022, Ann. Statist.)**,
[Conformal prediction beyond exchangeability](https://consensus.app/papers/details/e208c1bb1a585b38b8dd983e5ce40dcd/?utm_source=claude_desktop)
(429 cites) is the key theory: with fixed weights, the coverage gap from non-exchangeability is
**bounded by a (weighted) sum of total-variation distances** between the data sequence and its
permutations. This is the first rigorous "coverage gap ≤ divergence" statement — it is what makes
the *idea* of tying coverage to a measured shift magnitude legitimate. See also
**Oliveira et al. (2022, JMLR)**,
[Split Conformal Prediction and Non-Exchangeable Data](https://consensus.app/papers/details/740e5e838988582c8ee6f3ec42d54665/?utm_source=claude_desktop)
— split CP stays valid up to a small, characterizable coverage penalty under weak dependence.

**2b. Impossibility / non-identifiability under label & joint shift.**
Without assumptions, you **cannot** identify a joint/label shift from unlabeled target data — so you
cannot transport coverage exactly:

- **Podkopaev & Ramdas (2021)**,
  [Distribution-free uncertainty quantification for classification under label shift](https://consensus.app/papers/details/fadf6a05c47f5592926e23a6c97c58cd/?utm_source=claude_desktop)
  (115 cites) — label shift *also* degrades coverage/calibration; fixable **only** by reweighting
  using some labeled/anchor information from the target.
- **Tasche (2022/2023)**,
  [Factorizable Joint Shift](https://consensus.app/papers/details/347b22cf5c3859e8a8817299e5fbd393/?utm_source=claude_desktop)
  and [Invariance assumptions for class distribution estimation](https://consensus.app/papers/details/677c8f8145fb5fb18ee1a82013ee813a/?utm_source=claude_desktop)
  — general dataset shift is **not fully identifiable** without invariance assumptions; spells out
  exactly which assumptions buy identifiability.
- **Chen et al. (2022)**,
  [Estimating Model Performance When Both Covariates and Labels Shift (Sparse Joint Shift)](https://consensus.app/papers/details/ec0b9efa1e975738b159271eb696afa9/?utm_source=claude_desktop)
  — joint shift becomes tractable **only** under sparsity (few features + label shift).
- **Wu, Winston, Kaushik & Lipton (2019)**,
  [Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment](https://consensus.app/papers/details/506aac26aec45403a2dd8cc7adc918cb/?utm_source=claude_desktop)
  — "absent assumptions, domain adaptation is impossible." The canonical statement of the wall.

**Takeaway for Q2 (a genuine negative result the project needs).** **Exact** distribution-free
coverage transport under the kind of conditional/mechanism shift Lacuna faces is **provably
impossible without extra assumptions.** This is not a gap in current methods; it is a theorem. Any
"certified coverage" claim must therefore name its assumption. The defensible path is not to claim
exact transport but to claim **robust coverage over an explicit ball of shifts** (Q3/Q4) — and to
make the *radius* of that ball the thing you argue for (from anchors), partly measure (lower-bound
via C2ST), and report.

---

## 3. Coverage degradation bounded by a divergence — and the C2ST link (the crux)

This is the heart of the task: is there theory bounding the coverage gap by a distance between
calibration and deployment distributions, and can that distance be tied to the **C2ST-measured**
realism gap?

**3a. Coverage gap ≤ divergence — yes, in several metrics.**

- **Total variation:** Barber et al. 2022 (§2a) — coverage gap ≤ weighted TV.
- **f-divergence ball (the DRO route):** **Cauchois, Gupta, Ali & Duchi (2020, JASA)**,
  [Robust Validation: Confident Predictions Even When Distributions Shift](https://consensus.app/papers/details/af8b46c254645a5aa1eaf3ce22bd9172/?utm_source=claude_desktop)
  (116 cites) — produces sets with valid coverage for **any** test distribution in an
  **f-divergence ball of radius ρ** around the calibration population, by inflating the conformal
  level as a function of ρ; and develops **estimators of the expected future shift ρ** with
  consistency proofs. This is the single most important paper for Lacuna's framing: it operationalizes
  "measure the shift, inflate the level."
- **Fine-grained (covariate part exact + conditional part robust):** **Ai & Ren (2024)**,
  [Not all distributional shifts are equal: Fine-grained robust conformal inference](https://consensus.app/papers/details/9b3294f6f578546993835237bf377973/?utm_source=claude_desktop)
  — reweight to correct the *identifiable covariate* shift, while protecting against worst-case
  *conditional* shift bounded in an f-divergence ball. **This is almost exactly Lacuna's situation**
  (correctable covariate structure + an unidentifiable conditional/δ residual), and its use case is
  *sensitivity analysis of treatment effects under hidden confounding* — structurally identical to
  Lacuna's δ sensitivity.
- **Wasserstein (sharper, separates covariate vs concept):** **Xu et al. (2025)**,
  [Wasserstein-regularized Conformal Prediction under General Distribution Shift](https://consensus.app/papers/details/d8bbb7b714885037a44efe8318824844/?utm_source=claude_desktop)
  — explicitly notes prior **TV bounds "cannot identify the gap changes under distribution shift at a
  given α"**, derives a **Wasserstein upper bound** on the coverage gap, and *separates the covariate
  and concept-shift contributions* to the gap. Directly addresses the joint-shift case.
- **Lévy–Prokhorov (local + global perturbations):** **Aolaritei et al. (2025)**,
  [Conformal Prediction under Lévy-Prokhorov Distribution Shifts](https://consensus.app/papers/details/d7bf7d55924c5b60a6a00584e34855bf/?utm_source=claude_desktop)
  — propagates the ambiguity set *through the score function*, reducing a high-dim shift to a 1-D shift
  on scores and giving exact worst-case quantiles. A clean way to make the bound computable.
- **PAC / training-conditional version:** **Pournaderi & Xie (2024)**,
  [Training-Conditional Coverage Bounds under Covariate Shift](https://consensus.app/papers/details/01aa2ab216565e86b4460545409fdcf7/?utm_source=claude_desktop)
  — bounds depend on **"the severity of distributional changes"** and calibration size; gives the
  per-realization (not just on-average) guarantee a deployed tool actually needs.

So: **coverage gap is provably upper-bounded by a divergence (TV / f-div / Wasserstein / LP)** between
calibration and deployment. The "tie coverage to a measured quantity" ambition is sound *in principle*.

**3b. Does the C2ST measure that divergence? Partly — and the direction is the problem.**

The conceptual chain the charter wants is:
`near-chance C2ST ⟹ small divergence ⟹ small coverage gap.`

The pieces exist:
- A domain classifier *is* a density-ratio / divergence estimator. **Ben-David et al. (2010)**,
  [A theory of learning from different domains](https://consensus.app/papers/details/50c3320315ae55d98a6895d4edaa5690/?utm_source=claude_desktop)
  (4052 cites) — the **H-divergence** between domains is exactly a classifier-distinguishability
  measure, estimable from finite unlabeled samples. For the Bayes-optimal classifier, accuracy =
  `(1 + TV)/2`, so C2ST accuracy and TV are two views of the same quantity.
- **Lopez-Paz & Oquab (2016)**,
  [Revisiting Classifier Two-Sample Tests](https://consensus.app/papers/details/7ddfcc50ab0f51a38245814e2cf151ed/?utm_source=claude_desktop)
  (496 cites) — C2ST in interpretable units, and it **localizes where P and Q differ** (actionable for
  fixing generators).
- **Ramdas et al. (2016)**,
  [Classification Accuracy as a Proxy for Two-Sample Testing](https://consensus.app/papers/details/2a7400ce94ad5bf0b81dfa9a7b3a6670/?utm_source=claude_desktop)
  (87 cites) — consistency/rates of the accuracy-as-test approach.

**But here is the fatal subtlety, and it is a real finding.** A two-sample test (C2ST included)
controls the **Type-I error for `H0: P = Q`**. *Failing to reject does not certify closeness* — it can
equally mean low power (weak classifier, finite sample, or a representation **blind** to the actual
shift direction). Concretely:

- **Michel et al. (2020)**,
  [High Probability Lower Bounds for the Total Variation Distance](https://consensus.app/papers/details/06dd83b402ae5583815ae95dcf9729f1/?utm_source=claude_desktop)
  — what a classifier-based two-sample procedure delivers is a **high-probability *lower* bound on TV**
  (the minimal fraction of samples pointing to a difference). Certification needs an ***upper*** bound
  on the divergence. **A lower bound on the divergence and an upper bound on the coverage gap do not
  compose into a coverage guarantee.** This is the crux mismatch.
- **Zhu et al. (2019, Bernoulli)**,
  [Interpoint distance based two-sample tests in high dimension](https://consensus.app/papers/details/450a3cd703815e8bb323f97a071f88d4/?utm_source=claude_desktop)
  — common two-sample tests (energy distance, MMD with Gaussian/Laplacian kernels) are **inconsistent**
  for differences beyond marginal mean/variance; they can be blind to exactly the higher-order
  structure a mask mechanism lives in.
- **Szabadváry (2026)**,
  [Conformal Blindness: A Note on A-Cryptic change-points](https://consensus.app/papers/details/191e8c59fb3b5d47a0002b816e613063/?utm_source=claude_desktop)
  — even with the *oracle* conformity score, a **massive** distribution shift can produce **perfectly
  uniform p-values** (zero detected signal) if the shift is orthogonal to the score. A test's silence
  is not a certificate.

**The conformal C2ST fixes validity, not the direction problem.**
**Bansal et al. (2025)**,
[The surprising strength of weak classifiers for validating neural posterior estimates](https://consensus.app/papers/details/562eb154bed55727a8e1976c4eaee93d/?utm_source=claude_desktop)
(building on **Hu & Lei (2020, JASA)**,
[A Two-Sample Conditional Distribution Test Using Conformal Prediction](https://consensus.app/papers/details/ceef31f6bccf5b99ae77a6f096c84517/?utm_source=claude_desktop))
converts *any* classifier's scores — even weak/overfit — into **exact finite-sample p-values** with
(i) Type-I control and (ii) power that **degrades gently** with classifier quality. This is genuinely
valuable: it makes the realism gate statistically honest without a strong discriminator. **But it is
still a test.** A valid, well-powered p-value that fails to reject raises confidence that the shift is
small; it does **not** output a certified upper bound on the divergence (let alone on the coverage
gap). "Power degrades gently" mitigates, but does not eliminate, the lower-bound-vs-upper-bound
asymmetry.

**Verdict on the holy-grail link.** The C2ST→coverage-transport link is **real in direction and is the
right instrument**, but the strong form in the current §3 ("transports *iff* the C2ST gap is small;
this closes the argument rigorously") is **not** supported. What is supported:

> A C2ST/conformal-C2ST realism gate is a **necessary screen and a falsifier**: if it rejects (or its
> lower bound on the shift is large), certification is **void**. Passing it is **necessary but not
> sufficient** for a small coverage gap, because (a) it lower-bounds rather than upper-bounds the
> divergence, and (b) it can be blind to the specific (δ / conditional) direction in which plasmode
> and real differ. Sufficiency requires the *additional* assumption that the true deployment lies
> inside the swept/assumed shift ball — which is argued from anchors and domain-randomization (L4),
> **not** established by the C2ST.

---

## 4. Robust / DRO conformal and conformal **risk** control (the functional target)

Lacuna certifies a **functional** (mean / quantile / regression coefficient gap), not set membership.
The relevant generalization is conformal **risk** control.

- **Angelopoulos, Bates, Fisch, Lei & Schuster (2022)**,
  [Conformal Risk Control](https://consensus.app/papers/details/9805998eccd354e8801cf47963f61c07/?utm_source=claude_desktop)
  (253 cites) — controls **the expected value of any monotone loss**, generalizing coverage; tight to
  `O(1/n)`. Crucially, it **already includes extensions to distribution shift, quantile risk control,
  and U-statistics** — i.e. the exact objects (quantile gaps, coefficient gaps) Lacuna needs.
- **Zecchin & Simeone (2025, ISIT)**,
  [Generalization and Informativeness of Weighted Conformal Risk Control Under Covariate Shift](https://consensus.app/papers/details/271006daa48b50b8b255c597ef87e759/?utm_source=claude_desktop)
  — weighted CRC transports risk control under covariate shift and relates *efficiency* (bound width)
  to the base predictor's generalization and the shift extent.
- **Farinhas et al. (2023)**,
  [Non-Exchangeable Conformal Risk Control](https://consensus.app/papers/details/796ed82de22957669d8cc2691bf98b29/?utm_source=claude_desktop)
  — CRC for non-exchangeable data via relevance weights; tighter bounds under drift.
- **Yeh et al. (2025)**,
  [Conformal Risk Training](https://consensus.app/papers/details/7938ae3952b85945a36177e737152729/?utm_source=claude_desktop)
  and **Angelopoulos (2026)**,
  [Conformal Risk Control for Non-Monotonic Losses](https://consensus.app/papers/details/74a9c554271252568923b5ea1bd970bc/?utm_source=claude_desktop)
  — extend to OCE/CVaR tail risks and to non-monotone, multi-dimensional losses (validity then depends
  on algorithm **stability**) — relevant because a functional-gap loss over a δ-grid need not be
  monotone.
- **Hong (2025, Eur. Actuarial J.)**,
  [Conformal prediction of future insurance claims](https://consensus.app/papers/details/d7dbc142dfe352ccb408d3222a6e8e06/?utm_source=claude_desktop)
  — a worked precedent of conformal bounds used to meet a **regulatory** capital requirement
  (Solvency II): direct evidence that conformal certification is accepted in regulated decision-making,
  reinforcing the §1 "this is a demanded method" defensibility argument from the main review.

**Takeaway for Q4.** The functional target is **not** an obstacle — CRC is the right generalization and
its shift-aware variants exist. Lacuna should state its guarantee as **risk control on the functional
residual**, and inherit the covariate-vs-concept caveat from §2–3 unchanged.

---

## 5. Sim-to-real / synthetic-to-real coverage (plasmode as a conformal problem)

Work that treats "calibrate on synthetic, deploy on real" as a conformal problem — the closest direct
analogue to Lacuna's plasmode setup:

- **Bashari et al. (2025)**,
  [Synthetic-Powered Predictive Inference](https://consensus.app/papers/details/4c303d5b8a465801a0a059f0a1fb79b2/?utm_source=claude_desktop)
  — the most on-point. A **score transporter** (empirical quantile mapping) aligns nonconformity scores
  from synthetic data to trusted real data, achieving **finite-sample coverage with *no* assumptions on
  the real/synthetic distributions**. The price: it needs a **trusted real calibration anchor** to map
  onto — synthetic data improves *efficiency*, it does not by itself manufacture validity. This is the
  honest shape of "synthetic helps": it sharpens, real anchors validate. **Direct design lesson for
  Lacuna:** hold out a small set of *real* observed-cell residuals as the transport anchor wherever one
  exists (e.g. NHANES self-report-vs-measured).
- **Cabezas et al. (2025)**,
  [CP4SBI: Local Conformal Calibration of Credible Sets in Simulation-Based Inference](https://consensus.app/papers/details/313175354529592b9b25c572949a5373/?utm_source=claude_desktop)
  — conformal recalibration of simulator-trained posteriors to restore local coverage; the SBI
  community's standard answer to "simulator-trained, possibly miscalibrated."
- **Gopakumar et al. (2024)**,
  [Uncertainty quantification of surrogate models using conformal prediction](https://consensus.app/papers/details/4185108361255c5fbd3974907a153fa2/?utm_source=claude_desktop)
  — empirically, conformal coverage **held even on out-of-distribution physics regimes** — but the
  authors are explicit that this rests on marginal coverage + an exchangeability assumption they probe
  by sensitivity analysis. A model for *how to report* sim-trained coverage honestly.
- **Robotics sim-to-real precedent:** **Zhao et al. (2023, ICCPS)**,
  [Robust Conformal Prediction for STL Runtime Verification under Distribution Shift](https://consensus.app/papers/details/c32f301d01435f12a0e71bd3e4417702/?utm_source=claude_desktop)
  — CPS designed in a simulator, deployed in the real world; **assumes a known upper bound on the
  f-divergence** between design-time and deployment distributions and uses robust CP. This is the
  cleanest statement of the operating assumption Lacuna must adopt: *you must posit (and defend) a
  bound on the sim→real divergence; you do not get it for free.*

**Takeaway for Q5.** Synthetic-to-real conformal is an active, validated paradigm — but **every**
rigorous instance either (a) anchors on some trusted real calibration data, or (b) assumes a known
divergence-ball radius. None manufactures validity from synthetic data alone. That is the boundary
condition Lacuna inherits.

---

## 6. What this means for Lacuna

**Is "certified coverage" defensible?** **Yes — but only as *robust / assumption-named* coverage, not
as exact transported coverage.** The literature is unambiguous on three points:

1. **Exact distribution-free transport under mechanism (conditional/δ) shift is impossible without
   assumptions** (§2 — Tasche; Wu/Lipton; the label-shift non-identifiability results). This is a
   theorem, not a missing method. So the claim "we certify 95% coverage on real columns" must become
   **"we certify 95% coverage for every deployment within an explicit shift ball around the plasmode
   family."**
2. **Coverage gap *is* upper-boundable by a divergence** (§3a — Barber 2022 TV; Cauchois 2020 f-div;
   Xu 2025 Wasserstein; Aolaritei 2025 LP). The mechanism to "inflate the level by a function of the
   shift radius ρ" is standard and accepted (Cauchois 2020; Wang & Barber 2026). So the bound *can* be
   tied to a shift magnitude.
3. **The C2ST tells you the shift magnitude only in the falsification direction** (§3b — Michel 2020
   lower bound; Szabadváry 2026 blindness). It is a *necessary screen and a lower-bound monitor*, not
   a sufficient certificate.

**Required assumptions, stated plainly (this is the dissertation's honesty ledger):**
- **(A1) Bounded-residual / swept-ball assumption.** The true real deployment lies within the
  f-divergence (or Wasserstein) ball spanned by the plasmode family — most bindingly, the real δ /
  value-dependence lies within the **domain-randomized δ-grid (L4)**. This is the assumption that
  *replaces* the impossible exact-transport claim. It is argued from **anchors** (Jabkowski ESS income;
  NHANES self-report-vs-measured weight), not from the C2ST.
- **(A2) Covariate part is correctable.** The identifiable covariate component of the shift is handled
  by weighting (Tibshirani 2019; Ai 2024 for the fine-grained split), with a defensible likelihood-
  ratio estimate (clip per Wang & Barber 2026; group-weight per Bhattacharyya 2024).
- **(A3) A trusted real calibration anchor exists** for at least the residual-spread factor (Bashari
  2025; the residual term is already "measured on real observed cells, fully supervised" in §4 of the
  reformulation — this is the part of Lacuna that *does* satisfy exchangeability and needs no shift
  assumption).

**Can the coverage gap be bounded by the C2ST-measured realism gap (the holy-grail link)?**
**Not as a one-way certificate — and the current §3 wording overclaims.** Precisely:
- ✅ **As a falsifier / necessary gate:** if the (conformal) C2ST rejects, or its lower bound on the
  plasmode→real divergence is large, the certification is **void**. This is rigorous and is the correct,
  defensible use (it is also a circularity guard on the generators — charter §6).
- ✅ **As a localizer:** C2ST says *which* mask features are unrealistic, directing generator fixes
  (Lopez-Paz 2016).
- ⚠️ **As a sufficient certificate of small coverage gap:** **no.** A passing C2ST lower-bounds, not
  upper-bounds, the divergence, and can be blind to the δ-direction that matters most. Sufficiency rides
  on (A1), which the C2ST cannot establish.

**The corrected one-sentence soundness story (proposed replacement for the §3 closer):**

> *Lacuna does not transport coverage exactly — that is impossible under mechanism shift. It certifies
> **robust** coverage of the functional bound over an explicit δ/shift ball (Cauchois-style level
> inflation; CRC for the functional target), corrects the identifiable covariate component by weighting,
> anchors the residual factor on real observed cells, and uses the conformal-C2ST realism gate as a
> **necessary falsifier** that voids certification when plasmode and real are distinguishable. The
> guarantee's remaining load-bearing assumption is that the real deployment lies inside the swept ball —
> argued from nonresponse anchors, monitored (one-sidedly) by the C2ST, and reported, never assumed
> silently.*

This is **weaker but true**, and it is still a strong, publishable, regulator-legible claim (the
insurance/Solvency-II precedent, Hong 2025, shows conformal certification is accepted in exactly this
register). It converts "trust our coverage" into "here is the one assumption, here is how we bound it,
here is the monitor that catches its failure."

---

## 7. Carried risks / open problems

1. **The upper-bound gap is the open research problem.** Nothing in the literature gives a
   *distribution-free, finite-sample **upper** bound* on the plasmode→real divergence from samples alone
   (the C2ST gives a lower bound). Lacuna must either (a) **assume** the ball radius and defend it from
   anchors (A1), or (b) pursue confidence-bounded divergence *estimation*, which is hard in high
   dimensions and not solved here. Be explicit that (a) is the chosen path.
2. **C2ST blindness to the δ-direction.** Because the plasmode and real masks can be engineered to share
   marginal/footprint statistics while differing in value-dependence (precisely Lacuna's L4), a footprint
   C2ST may pass while the *coverage-relevant* shift is large (Szabadváry 2026; Zhu 2019). Mitigation:
   make the C2ST's representation include **consequence/order-statistic features** (the same features the
   network needed in P2.2c), and run a **conditional** two-sample test (Hu & Lei 2020), not just a
   marginal one.
3. **Likelihood-ratio estimation error (A2).** Undercoverage from a learned `w` is real (Wang & Barber
   2026); needs clipping + a data-estimated level inflation. Budget for this in the certification.
4. **Continuous monitoring is available and should be adopted.** Conformal test martingales detect when
   real deployment drifts out of the certified ball post-hoc: **Vovk (2021)**,
   [Retrain or not retrain: Conformal test martingales for change-point detection](https://consensus.app/papers/details/58969f8391e05738849cf9b4bb4cb49e/?utm_source=claude_desktop);
   **Prinster et al. (2025)**,
   [WATCH: Adaptive Monitoring for AI Deployments via Weighted-Conformal Martingales](https://consensus.app/papers/details/729175bfe9ef5ce898b0848b29c5807c/?utm_source=claude_desktop)
   (diagnoses covariate vs concept shift online). But note **conformal blindness** (Szabadváry 2026)
   applies to martingales too — monitoring is not a substitute for the up-front assumption.
5. **Training-conditional vs marginal coverage.** A deployed per-column bound needs the
   *training-conditional* (per-calibration-draw) guarantee, which is strictly harder than marginal and
   can be impossible for some methods (**Bian & Barber 2022**,
   [Training-conditional coverage for distribution-free predictive inference](https://consensus.app/papers/details/9f4cc3d093a055e389affc2eb3bcb2cc/?utm_source=claude_desktop);
   Pournaderi & Xie 2024). Use split CP (which *does* have it) and the **Small-Sample Beta Correction**
   (**Zwart 2025**,
   [Probabilistic Conformal Coverage Guarantees in Small-Data Settings](https://consensus.app/papers/details/a5792e7d105b57589f379ee3ee765a28/?utm_source=claude_desktop))
   to convert "coverage in expectation" into "coverage with probability 1−δ over the calibration draw."
6. **Per-column calibration sample size.** Tighter divergence-ball ⇒ wider inflation ⇒ more calibration
   data per reference class. The rate/efficiency tradeoff (Zecchin 2025) should be quantified before
   claiming a usable bound width.

---

## 8. What to fold back into `docs/LITERATURE-REVIEW.md` §3

§3 currently cites only Lei 2016, Gibbs 2023, Hulsman 2022, and asserts the C2ST link "closes the
argument rigorously." Recommended edits:

1. **Soften the closing claim.** Replace "This closes the argument rigorously: the coverage guarantee
   transports iff the plasmode→real shift is small …" with the §6 corrected one-sentence story
   (robust/assumption-named coverage; C2ST as necessary falsifier, not sufficient certificate). This is
   the single most important correction — the current wording is the project's biggest soundness
   overclaim.
2. **Add the coverage-gap-by-divergence backbone:** Barber et al. 2022 (non-exchangeable CP, TV bound);
   Tibshirani et al. 2019 (weighted CP / covariate shift); Cauchois et al. 2020 (f-divergence robust
   validation — the level-inflation template); Ai & Ren 2024 (fine-grained: covariate-correct +
   conditional-robust, the closest structural match); Xu et al. 2025 (Wasserstein, separates covariate
   vs concept).
3. **Add the impossibility result** (Tasche; Wu/Lipton 2019; Podkopaev & Ramdas 2021) as the named
   reason exact transport is off the table — this is a *strength*, not a weakness, of the framing.
4. **Add conformal risk control** (Angelopoulos et al. 2022; Zecchin 2025; Farinhas 2023) as the correct
   machinery for the *functional* target, and **Hong 2025** (Solvency-II precedent) for regulatory
   defensibility.
5. **Add the sim-to-real anchor papers** (Bashari 2025 synthetic-powered PI; Zhao 2023 robust-CP sim2real
   assuming a known divergence ball) under a new "plasmode→real as a conformal problem" sub-bullet.
6. **Sharpen the C2ST entry** (currently in §4) with the direction caveat: Michel 2020 (classifier ⇒
   *lower* bound on TV); Szabadváry 2026 (conformal blindness); Hu & Lei 2020 (use the *conditional*
   two-sample test, not just marginal). Note that the conformal C2ST (Bansal 2025) fixes test *validity*
   but not the lower-vs-upper-bound asymmetry.

Net effect: §3 goes from one paper + an overclaim to a defensible, theorem-aware account in which the
certified-coverage claim is **true as stated once the assumption is named** — which is exactly the
standard a dissertation chapter needs to survive its defense.

---

### Headline
- **Is certified coverage defensible?** Yes — as **robust coverage over an explicitly assumed shift
  ball**, with conformal risk control for the functional target. Not as exact transport (that is provably
  impossible under mechanism shift without assumptions).
- **Is the C2ST→coverage link real or wishful?** **Real in direction, but the strong "C2ST gap =
  coverage gap, argument closed" form is wishful.** The C2ST is a rigorous **necessary falsifier and
  localizer** (and a *lower-bound* monitor of the shift); it is **not** a sufficient certificate of a
  small coverage gap, because it lower-bounds rather than upper-bounds the divergence and can be blind to
  the δ-direction. The guarantee ultimately rides on the domain-randomized δ-grid being wide enough —
  argued from anchors, not proven by the classifier.

*Search note: performed with the user's authenticated Consensus connector (full result sets across
Semantic Scholar/PubMed/Scopus/arXiv); no result cap or signup gating encountered. ~100 papers
screened across 9 queries; ~45 cited.*
