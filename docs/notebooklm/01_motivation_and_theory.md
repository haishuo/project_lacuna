# Project Lacuna — Document 1 of 3: Motivation, Problem Framing, and Theoretical Position

This is the first of three companion documents collating Project Lacuna for ingestion into NotebookLM. Document 1 explains *why* Lacuna exists, what problem it addresses, and the theoretical position it takes. Document 2 explains the architecture in full. Document 3 covers the experimental history, ablations, and empirical results.

---

## 1. The Missing Data Problem

Missing data is ubiquitous in scientific research, medical records, survey datasets, and observational studies. Sensors fail. Patients drop out. Respondents refuse questions. By the time a dataset reaches an analyst, some non-trivial fraction of cells are typically NaN.

The standard statistical machinery for handling missing data — multiple imputation, complete-case analysis, sensitivity analysis, selection models, pattern-mixture models — is only valid when applied to data whose generating *mechanism* matches the method's assumptions. Using the wrong method on data with the wrong mechanism produces systematically biased estimates whose direction is determined by the unobservable values themselves. This is not a small error: it is the kind of error that invalidates the conclusions of an entire study without the analyst ever realising it.

The field, following Rubin (1976), classifies missingness mechanisms into three canonical regimes.

### MCAR — Missing Completely At Random

The probability of a value being missing is entirely independent of both observed and unobserved data. Formally: `P(R | X_obs, X_mis) = P(R)`. Examples include a lab technician dropping samples at random, or a sensor failing for reasons unrelated to what it was measuring. Under MCAR, complete-case analysis (simply dropping rows with missing values) is valid and unbiased, though inefficient. Mean imputation is also acceptable.

### MAR — Missing At Random

The probability of missingness depends on the *observed* variables but not on the missing values themselves. Formally: `P(R | X_obs, X_mis) = P(R | X_obs)`. The canonical example: younger patients are less likely to report income — age is observed, and missingness depends on age, not on the unreported income value. Under MAR, complete-case analysis is biased. Multiple imputation (MICE, Amelia), likelihood-based methods (EM, FIML), or similar conditional methods are required for unbiased inference.

### MNAR — Missing Not At Random

The probability of missingness depends on the *unobserved* values themselves. Formally: `P(R | X_obs, X_mis) ≠ P(R | X_obs)`. Example: high earners are less likely to report their income, so missingness in the income column depends on the missing income value. MNAR is the most dangerous regime because even multiple imputation produces biased results. Sensitivity analyses, selection models (Heckman correction), and pattern-mixture models are required.

### Why Mechanism Matters Practically

The choice of mechanism is *load-bearing* for downstream analysis. Misclassifying MNAR as MAR yields biased estimates whose direction is determined by the unobservable values. The mechanism therefore shapes both the validity of inference and the ethical defensibility of substantive conclusions in fields ranging from medicine to public policy.

In current practice, the choice is rarely made empirically. Standard procedure assumes MAR, fits under that assumption, and (sometimes) runs sensitivity analyses with hypothesised MNAR shifts. The MAR assumption is treated as a modelling choice grounded in domain knowledge rather than something to be tested.

Lacuna addresses the diagnostic problem: *given a dataset with missingness, which mechanism generated it?*

---

## 2. The Theoretical Obstacle: Non-Identifiability

A central caveat structures the entire project. **The mechanism is not identifiable from the data alone.** Molenberghs, Beunckens, Sotto and Kenward (2008) prove that for any observed dataset there exist both MAR and MNAR data-generating processes producing identical likelihoods for the observed data. From the data alone, the analyst cannot distinguish them.

This is not a defect of any particular algorithm. It is a mathematical fact about what observed data can in principle convey about unobserved data. Every classifier that operates on observed cells alone — including Lacuna — runs into this floor.

The pragmatic implication: a system that returns a categorical mechanism label ("this is MAR") is making a claim it cannot in general support. Either the claim is wrong some non-trivial fraction of the time, or the system is silently relying on auxiliary assumptions (sample-size, domain priors, regularity conditions) that the user is not told about.

Lacuna takes the view that the only honest response to non-identifiability is to *report a calibrated posterior* over mechanisms rather than to commit to a single label. Decisions made under that posterior — for example, how much weight to put on a sensitivity analysis, or whether to escalate to a more conservative estimator — remain the analyst's responsibility. The model's job is to provide a defensible *belief distribution* that combines (a) per-mechanism likelihood evidence from the data and (b) a domain-appropriate prior, into a single probability vector the analyst can act on.

---

## 3. Lacuna as a Calibrated Posterior Estimator

The architectural and evaluative consequences of the calibrated-posterior framing are significant.

### Output Type

Rather than a categorical classifier, Lacuna is framed as

> P(mechanism | data, domain prior, learned likelihood)

The output is a probability vector `[P(MCAR), P(MAR), P(MNAR)]` together with a downstream "decision" (Green/Yellow/Red) computed from a Bayes-optimal rule under an asymmetric loss matrix. The probabilistic output is the primary product; the categorical decision is a convenience for users who prefer one.

This framing is *analogous in spirit* to confidence-scoring approaches such as AlphaFold's pLDDT (Jumper et al. 2021), with one important disanalogy: AlphaFold's pLDDT is calibrated against a task with verifiable ground truth (resolved protein structures), whereas Lacuna's mechanism labels come from literature-consensus readings on partially unidentifiable processes. The calibration target is therefore weaker than AlphaFold's, and the project does not claim otherwise.

### Validation Metric

A model that assigns `P(MNAR) = 0.35` to a contested case (where the missing-data literature is itself split between MNAR and MAR-conditional readings) is performing correctly. The right metric is whether `P(MNAR)` *systematically elevates* on real MNAR-consensus data relative to MAR-consensus data — not whether `argmax(P)` matches a textbook label that itself rests on contested assumptions. Argmax accuracy on synthetic data is reported, but it is not the headline number; calibrated posterior elevation on real anchors is.

### Specialisation by Collection Process

The plausibility of MNAR is determined by the data-generation process. Surveys can have refusal-driven MNAR. Sensors can have detection-limit MNAR. Clinical longitudinal data can have informative dropout. There is no domain-invariant fingerprint to learn — the *kinds* of MNAR that arise differ qualitatively between collection processes. A generic missingness classifier was tried first (the `lacuna_tabular_110` build, 110 generators across all data types). It produced 87% synthetic accuracy but failed on real-world data, with 0/3 textbook-MAR-consensus datasets correctly classified. Six rounds of generator-coverage tweaks, class re-weighting, and loss adjustments did not move real-world performance. The diagnostic was that *specialisation by collection process is structural*, not an optional feature.

The current build, **Lacuna-Survey**, is calibrated specifically for self- and interviewer-administered survey questionnaires. Its MNAR head is trained on mechanisms specifically plausible for surveys: self-censoring, threshold-based truncation, social-desirability under- and over-reporting, quantile-based censoring, gaming and volunteer effects, and module-level refusal driven by latent values. The "MNAR" label as Lacuna-Survey uses it is therefore *survey-plausible MNAR*, not the full mathematical class of mechanisms satisfying Rubin's MNAR definition. Other MNAR processes — detection-limit truncation in sensor data, informative dropout in clinical longitudinal panels — are explicitly out of scope and would require a separate specialised variant (Lacuna-Sensor, Lacuna-Clinical, etc.).

---

## 4. Why Anyone Should Care

Three audiences have reason to care about Project Lacuna.

### Applied Analysts

For a researcher with a real dataset and a missing-data problem, the workflow today is: assume MAR, run multiple imputation, hope. Lacuna offers a second opinion grounded in the empirical fingerprint of the missingness. It does not — and cannot, given Molenberghs — replace domain expertise. But it produces a calibrated probability that the data are something other than MAR, and that probability is informative for choosing whether to invest in a sensitivity analysis or selection model. On real survey anchors with literature-consensus MNAR readings (income refusal, depression-screener module refusal, self-reported weight, drug-use battery), Lacuna-Survey systematically elevates `P(MNAR)` by 3–5× relative to clean MAR baselines, giving the analyst a quantitative trigger for "this case warrants closer scrutiny."

### Methodologists

For researchers in missing-data methodology, Lacuna is a concrete instantiation of the calibrated-posterior approach to a problem that has historically been treated either as untestable (the assume-MAR tradition) or as testable in restricted ways (Little's MCAR test, which can rule out MCAR but cannot distinguish MAR from MNAR). Lacuna's evaluation harness — including the per-mechanism reconstruction-error decomposition and the explicit confrontation with Molenberghs's identifiability bound on the NHANES anchors — provides a worked example of what honest reporting looks like when the problem itself is partially unidentifiable. The project's commitment to documenting negative results (the asymmetric expert pool that failed, the ESS dead end, the OOD detector that was removed rather than retained as half-broken) is part of that exercise.

### Machine-Learning Practitioners

For ML practitioners, Lacuna is a non-obvious application of transformer architectures. The natural temptation is to flatten the dataset into a sequence and let attention figure it out, but flattening loses the row identity that MAR detection critically depends on. Lacuna's row-wise attention scheme is one principled response: each row is treated as an independent sequence over its columns, and cross-row aggregation happens only via two stages of attention pooling. The Mixture of Experts head, the auxiliary reconstruction heads (one per mechanism), and the post-hoc temperature calibration are individually familiar, but the combination — an MoE whose gate sees both learned representations and explicit handcrafted features and per-mechanism reconstruction errors — is specific to this problem and the result of the experimental arc described in Document 3.

---

## 5. The Stakes of Getting It Wrong

The downstream cost of misclassifying mechanism is encoded explicitly in Lacuna's decision rule, which uses an asymmetric loss matrix. The rationale is not incidental — it directly motivates the project.

|  | True MCAR | True MAR | True MNAR |
|--|-----------|----------|-----------|
| **Green (assume MCAR)** | 0.0 | 0.3 | 1.0 |
| **Yellow (assume MAR)** | 0.2 | 0.0 | 0.2 |
| **Red (assume MNAR)** | 1.0 | 0.3 | 0.0 |

Treating MNAR data as MCAR (Green when truly Red) carries the maximal cost: the analyst runs complete-case analysis on data whose missingness is informative about the missing values, producing biased point estimates and invalid confidence intervals. This is the canonical failure mode the field has been warning about since Rubin (1976).

Treating MCAR data as MAR (Yellow when truly Green) carries a small cost. Multiple imputation on truly MCAR data is wasted effort but produces unbiased estimates with slightly inflated variance. This asymmetry — that confusing MNAR with MCAR is catastrophic, but confusing MCAR with MAR is merely wasteful — is the practical reason mechanism diagnosis matters at all. It also motivates why Lacuna's decision rule tilts toward Yellow under uncertainty: the cost of being too conservative (Yellow on MCAR) is a fifth the cost of being too aggressive (Green on MNAR).

---

## 6. What This Document Does Not Cover

For the architecture in detail — tokenization, encoder design, MoE construction, reconstruction heads, generator system, training pipeline, post-hoc calibration — see Document 2.

For the experimental history — the ten core experiments leading to the published 82.6% accuracy on synthetic data, the Lacuna-Survey v1–v11 iteration arc, the NHANES anchor validation, the negative results and dead ends — see Document 3.
