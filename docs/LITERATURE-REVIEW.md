# Literature Review — Lacuna (Recoverability Reformulation)

*Conducted 2026-06-13, before the generator body of work, to position every limb of the
reformulated design (`docs/proposals/PROPOSAL-recoverability-reformulation.md`) against the state of the
art. Searches via Consensus (peer-reviewed corpus) + web. Organized by design limb: for each,
what the field does, where Lacuna sits, and the gap we fill. Citations are traceable links;
DOIs to be finalized for the dissertation.*

---

## 0. One-paragraph positioning

Every component of Lacuna has mature prior art — and in every case the field stops one step short
of what the reformulation requires. Recoverability is *defined* (Mohan-Pearl) but only decidable
given a missingness graph nobody has. Sensitivity analysis via δ-shift is *regulatory-standard*
(tipping-point) but manual and per-analysis. Coverage guarantees are *rigorous* (conformal) but
assume exchangeability that plasmode→real breaks. Synthetic missingness is *standardized*
(ampute/plasmode) but built to rank methods, not to certify transportable bounds. Item nonresponse
is *well-studied* (and our exact ESS income case is published) but never turned into a calibrated,
scalable per-column prior. Lacuna's contribution is the *join*: empirical recoverability bounds,
per-column and per-functional, certified on plasmode and anchored by the nonresponse literature —
automating tipping-point analysis at scale with a coverage statement.

---

## 1. Recoverability theory (the object)

- **Mohan, Pearl & Tian (2013)**, [Graphical Models for Inference with Missing Data](https://consensus.app/papers/details/cca133c3b5105709b773a820cb62c538/) — introduces *m-graphs* and the formal question: does a consistent estimator of a query Q exist when data are MNAR? Recoverability is **per-query**, not per-dataset.
- **Mohan & Pearl (2018, JASA)**, [Graphical Models for Processing Missing Data](https://consensus.app/papers/details/d99ea926adb556a782b6b2ef0058a0e6/) (226 cites) — conditions guaranteeing consistent estimation even under MNAR; the transparency/estimability/testability lens.
- **Holovchak et al. (2024, Biostatistics)**, [Recoverability of causal effects under missing data](https://consensus.app/papers/details/2c0783e3fcb85807947a429286a8b16c/) — applies m-DAGs to real longitudinal data. **Critical findings for us:** "no general algorithms are available to decide recoverability — case-by-case"; recoverability is "sensitive to even the smallest changes in graph structure"; and under MNAR, complete-case can *beat* multiple imputation.

**Where Lacuna sits / the gap.** Mohan-Pearl give recoverability as a *graph-theoretic* property — decidable only if you know the missingness DAG, which in practice nobody does, and the answer is fragile to graph mis-specification (Holovchak). **Lacuna is the empirical counterpart: we never assume the graph; we MEASURE recoverability on plasmode (where we control truth) and BOUND the unidentified residual.** This is the project's central theoretical positioning: graph-based recoverability is the theory, plasmode-calibrated recoverability is the practice. Holovchak's "imputation isn't always the answer" also vindicates our "bound, don't blindly impute" stance.

## 2. Sensitivity analysis / the δ output (R2)

- **Gorst-Rasmussen et al. (2022, J Biopharm Stat)**, [Fast tipping point sensitivity analyses](https://consensus.app/papers/details/f6bec635e78e516cbc0b528a98dad7f8/) — tipping-point analysis is "routinely requested by regulatory agencies": add a successively more extreme shift δ to missing predictions; if the δ needed to overturn the conclusion is implausibly large, the result is robust.
- **Liu et al. (2017)**, [Control-Based Imputation and Delta-Adjustment Stress Test](https://consensus.app/papers/details/f8a44567d9595b5bac3dfc893256ee42/) — formal link between control-based imputation and δ-adjustment tipping points.
- **Fataliieva et al. (2022)**, [Pattern Mixture Models and Tipping Point Analysis](https://consensus.app/papers/details/71175452abc0511e8c1ad123e9d4a8bf/) — pattern-mixture + tipping point in applied social research.

**Where Lacuna sits / the gap.** Our R2 output — "here is the δ-range; if your conclusion flips inside it, it's not robust" — **IS tipping-point analysis**, the regulatory-standard MNAR sensitivity method. This is enormous for defensibility: we are not inventing an output, we are *automating and anchoring* an established one. The field's tipping point is manual (analyst picks shift parameters), per-analysis, one variable at a time. Lacuna supplies the δ-range *from data + the nonresponse literature* (§6), runs it *per-column at scale*, and reports it in *functional units*. Contribution = automated, anchored, scalable tipping-point analysis.

## 3. Coverage machinery (certification)

- **Lei et al. (2016, JASA)**, [Distribution-Free Predictive Inference for Regression](https://consensus.app/papers/details/18e5e78f0d1c53598b687cec06a6c299/) (1179 cites) — conformal prediction: finite-sample marginal coverage with ANY estimator, no distributional assumptions.
- **Gibbs et al. (2023, JRSS-B)**, [Conformal prediction with conditional guarantees](https://consensus.app/papers/details/fd1b4c4dcc1b5d7f869b216154f1fb36/) — exact *conditional* coverage is impossible in finite samples; reformulates conditional coverage as **coverage over a class of covariate shifts**.
- **Hulsman (2022)**, [Distribution-Free Finite-Sample Guarantees and Split Conformal Prediction](https://consensus.app/papers/details/6c01103534d45860a0291824ce7553ae/) — tolerance regions; coverage of order-statistic sets follows a Beta distribution.

**Where Lacuna sits / the gap.** "Certified coverage of the bound" should be built on conformal/tolerance-region machinery rather than invented — the functional gap is a residual we can conformalize on the plasmode calibration set. **But the honest subtlety, which Gibbs makes precise:** conformal coverage assumes exchangeability; plasmode→real is a *covariate/distribution shift* that breaks it. So (a) the relevant tool is conformal-under-shift, and (b) the shift magnitude IS the generator-realism gap (§5). This closes the argument rigorously: the coverage guarantee transports iff the plasmode→real shift is small, which is exactly the measurable layers 1–3 of generator realism. Coverage validity is thus *tied to a quantity we can measure*, not asserted.

## 4. Synthetic missingness: the generators (see also the charter)

- **Schouten, Lugtig & Vink (2018, J Stat Comput Simul)**, [multivariate amputation (`ampute`/`pyampute`)](https://www.tandfonline.com/doi/full/10.1080/00949655.2018.1491577) — the field-standard tool. Weighted-sum-score: MAR = weights on observed vars, MNAR = nonzero weight on the amputed var itself. Deliberately simple; makes **no realism claim**.
- **Weberpals et al. (2024, Clin Epidemiol)**, [plasmode framework + missingness diagnostics](https://consensus.app/papers/details/39e1c13d5f3850e5b9ccb56cad441e82/) — "plasmode" = the established epidemiology name for our paradigm (real X + injected known mechanism). Tests whether diagnostics can distinguish MCAR/MAR/MNAR.
- **Radosavljević et al. (2024, BMC Med Res Methodol)**, [A generative model for evaluating missing data methods](https://consensus.app/papers/details/9c9937dbfe2d592e8f9f98268dde0899/) — **closest prior art to realistic generators**: learns block/co-missingness structure from real data (UK Biobank) via hierarchical clustering of missingness patterns + inter-variable correlation + informative missingness.
- **Mangussi et al. (2025, Neurocomputing)**, [`mdatagen`](https://consensus.app/papers/details/c571ab0f76eb52c499585fd894408773/) — 20 standardized amputation scenarios for reproducibility.

*How to validate generator realism (the L1 gate):*
- **Lopez-Paz & Oquab (2016)**, [Revisiting Classifier Two-Sample Tests (C2ST)](https://consensus.app/papers/details/7ddfcc50ab0f51a38245814e2cf151ed/) (496 cites) — train a classifier real-vs-generated; near-chance accuracy ⇒ same distribution; interpretable units; *localizes where P and Q differ*. The canonical realism gate.
- **Bansal et al. (2025)**, [The surprising strength of weak classifiers for validating neural posterior estimates](https://consensus.app/papers/details/562eb154bed55727a8e1976c4eaee93d/) — **conformal C2ST** (building on Hu & Lei): any classifier's scores → exact finite-sample p-values, Type-I control, power degrading gently with classifier quality. Ties the realism gate to the conformal coverage machinery (§3) and removes the need for a strong discriminator.
- **Dankar & Ibrahim (2022, IEEE Access)**, [A Multi-Dimensional Evaluation of Synthetic Data Generators](https://consensus.app/papers/details/541925ff6e27543fa15874bbd338492e/) (85 cites) — fidelity taxonomy: attribute / bivariate / population / application. The footprint-statistic panel.
- **Alaa et al. (2021)**, [How Faithful is your Synthetic Data? (α-Precision, β-Recall, Authenticity)](https://consensus.app/papers/details/7b9880e7990158b0ba8f6d814fe79f21/) (295 cites) — adds an **authenticity** axis (guards against memorizing real samples) — the anti-mask-memorization check.

**Where Lacuna sits / the gap.** Adopt "plasmode" framing; subsume `pyampute` as a baseline vocabulary; extend Radosavljević's method to learn layers 1–3 from our corpora; validate realism with conformal C2ST + the fidelity taxonomy. **The divergence:** all of these build generators to *rank imputation methods* — a weak realism bar robust to mis-specification. Lacuna needs generators realistic enough to *certify a transportable quantitative bound* — a strictly stronger bar (§5, charter). No one in this literature certifies coverage of a deployable bound.

## 5. The empirical contest: imputation benchmarks

- **Jäger et al. (2021, Front Big Data)**, [A Benchmark for Data Imputation Methods](https://consensus.app/papers/details/31b49f7f9bbe5dc5937c8dd4a7ebdcc1/) (223 cites) — classical vs deep imputation under realistic conditions.
- **Toye et al. (2025, PMLR)**, [Benchmarking with Real-World Test Cases](https://consensus.app/papers/details/d9947d2b9e2150a5bebf039d87b87a71/) — "current evaluation practices do not provide an accurate picture of real-world performance"; linear interpolation often wins.
- **Poette et al. (2026, Sci Rep)**, [Benchmarking under real-world-inspired ICU scenarios](https://consensus.app/papers/details/67af7729b3d459d0ad9c35e127880f28/) — **MCAR compresses method differences; structured gaps reveal separation; linear interpolation a strong baseline — BUT Transformers/GANs win on structured gaps in rich data.**

**Why this matters for us.** (a) "Scenario choice determines the conclusion" is the empirical proof that generator quality is load-bearing. (b) Simple methods being brutally competitive vindicates our frozen-null discipline. (c) **Deep methods separating specifically on structured missingness in rich data is the first evidence-backed reason to expect the imputer-network slot may win where the v1.0 detector lost** — and it points at *where* (structured, realistic, high-dimensional missingness — exactly what good generators produce).

## 6. Item nonresponse — the semantic channel's home

- **Jabkowski et al. (2023, Field Methods)**, [Not Random and Not Ignorable: income nonresponse in the ESS 2008–2018](https://consensus.app/papers/details/885d29a159385b0993c400315e21710d/) — **our exact dataset and variable.** Two mechanisms: task complexity + question sensitivity. **Refusal is HIGHEST among LOWER-income respondents**; nonresponse propensity correlates with income and biases downstream estimates. A published, conditional δ-anchor for ESS income — and a warning that the δ *direction* is population-specific (not the naive "rich refuse").
- **Mignogna et al. (2023, Nature Human Behaviour)**, [Item nonresponse is systematic and associated with genetic loci](https://consensus.app/papers/details/9d1070f88c395fbe8b5b8c294034297d/) — UK Biobank, 109 items: "Prefer not to answer" and "I don't know" factors predict nonresponse in *follow-up* surveys; genetically correlated with education/health/income; PNA and IDK genetically *distinct*. External validation of (i) serial-disposition nonresponse (our 0.90 covariate-predictability), (ii) refusal≠DK as real distinct types (our Arm-3 dissociation), (iii) covariate (MAR) dependence.
- **Silber et al. (2021, JRSS-A)**, [Question, respondent and interviewer effects on two types of item nonresponse](https://consensus.app/papers/details/03cf9172b9a35471bc286d92f0bd14fd/) — DK and REF are "substantially different... distinguishable disruptions of the cognitive response process"; question + respondent characteristics drive both, interviewer characteristics drive only DK. The cognitive-model / satisficing theory backing for "question characteristics predict nonresponse."

**Where Lacuna sits / the gap.** The semantic channel's premise — question text predicts nonresponse behavior, and refusal vs DK are distinct — is *established survey methodology*, now independently confirmed genetically. Our Arm-3 result (frozen suffices for DK, training helps refusal) is consistent with Silber's "different cognitive disruptions." **Gap:** this literature studies nonresponse to *explain* it per-study; nobody builds a *transferable, calibrated, per-column text→behavior model* used as an automated sensitivity prior. The δ-anchor ledger (R2) draws directly on Jabkowski-type studies.

## 7. Functional-preserving / distribution-matching imputation

- **Muzellec et al. (2020)**, [Missing Data Imputation using Optimal Transport](https://consensus.app/papers/details/d907fdaf9cc756cb9915fe65b305e631/) (180 cites) — "two batches from the same dataset share a distribution" → OT loss → impute; matches/beats SOTA under MCAR/MAR/MNAR. **The nearest neighbor to our functional-loss objective.**
- **Feydy et al. (2018)**, [Interpolating between OT and MMD via Sinkhorn Divergences](https://consensus.app/papers/details/7b2c74abf5cf5edba975253ab618eff6/) (674 cites) — the MMD↔OT bridge; GPU-scalable distribution losses. The technical tool for the functional battery.

**Where Lacuna sits / the gap.** OT-imputation already trains imputers to match *distributions* rather than cells — strong validation of "preserve functionals, not blanks." Our refinement (§7.1 of the reformulation): target the *analyst's specific estimand* (μ, σ, β), use draws, and handle δ explicitly via the ignorable-only training split. Muzellec is the baseline/ancestor; the per-functional-battery + recoverability-bound use is new.

## 8. Amortization / PFN — the amortized-estimator framing

- **Vetter et al. (2025)**, [Effortless Bayesian Inference using Tabular Foundation Models (NPE-PFN)](https://consensus.app/papers/details/7baa6472058f5c46bd40ed08424d785d/) — TabPFN as an amortized conditional density estimator for simulation-based inference; "amortize Bayesian inference under a broad synthetic prior" — **exactly our "train on plasmode = amortize under P_prior" framing.** Robust to misspecification, orders-of-magnitude fewer simulations.
- **Mourão et al. (2026)**, [PFNs for Causal Inference](https://consensus.app/papers/details/adae1cb8aeb5578798f89f086c962a1a/) — **CausalPFN "exhibited poor coverage of its 95% credible interval due to estimation bias and inadequate uncertainty quantification."**

**Where Lacuna sits / the gap.** PFNs are the architectural lineage for any amortized slot (train a network on synthetic draws from a prior; one forward pass at deployment). **The documented PFN coverage failure (Mourão) is precisely the weakness our certified-coverage design targets** — we position as "amortized inference *with* a transportable coverage guarantee," against PFN methods that amortize but miscalibrate.

## 9. Synthesis — the novelty map (what is ours)

| limb | mature prior art | Lacuna's one step further |
|---|---|---|
| recoverability | Mohan-Pearl m-graphs (needs graph) | empirical, plasmode-calibrated, no graph |
| δ-output | tipping-point / δ-adjustment (manual) | automated, anchored, per-column, in functional units |
| coverage | conformal (assumes exchangeability) | conformal-under-shift; validity tied to measured generator realism |
| generators | ampute / plasmode / Radosavljević (rank methods) | certification-grade realism (stronger bar) |
| imputer | OT/distribution-matching (cell/dist) | per-functional battery + draws + bound-tightening |
| semantic prior | item-nonresponse studies (explain per-study) | transferable calibrated text→behavior model as a prior |
| amortization | PFN/TabPFN (miscalibrated coverage) | amortized inference WITH certified coverage |

**The thesis in one line, literature-grounded:** *Recoverability theory says what is estimable given a graph you don't have; regulatory practice shifts a δ you pick by hand; Lacuna measures recoverability where it can, anchors δ from the nonresponse literature, and certifies the resulting per-column functional bounds with conformal coverage whose validity is tied to generator realism we validate empirically.*

## 10. Open risks the literature flags (carry into the build)

1. Recoverability is fragile to mechanism mis-specification (Holovchak) ⇒ generators and the δ-grid must be broad/honest, not point estimates.
2. Conformal coverage breaks under distribution shift (Gibbs) ⇒ plasmode→real transport is the binding assumption; measure the shift (realism gates), don't assume it.
3. Simple imputers are hard to beat (Toye, Jäger) ⇒ frozen nulls are MICE-with-draws, not strawmen; the imputer-network claim must clear them.
4. Amortized/PFN methods miscalibrate (Mourão) ⇒ never trust a learned interval without coverage certification.
5. δ direction is population-specific (Jabkowski: low-income refuse more in ESS) ⇒ anchors are per-reference-class and per-population; no universal "sensitive ⇒ skews high."
