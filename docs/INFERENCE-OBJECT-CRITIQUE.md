# Critique — is "observed-vs-reference deviation" the correctly-specified inference object?

*Scientific critique, not implementation. **No code.** Status: **for PI review.** Evaluates the proposed
4-layer decomposition (column distribution encoder → cross-column reference → deviation → prior/
calibration) for *coherence of the object*, per the PI's five questions. Tone: adversarial — the job is
to find where the specification leaks, before more thought is invested. Grounded in the identification
theory (`NORTH-STAR.md` §2/§3½), the oracle's known caveat, and the accumulated evidence (Stage 0; the
`transfer_features` failure; proxy absorption).*

## 0. Verdict up front

- **As an engineering factorization:** coherent and useful, and it has a mature statistical home
  (pattern-mixture / selection models with a sensitivity parameter).
- **As a specification of the inference *object*:** **leaky in one decisive way** — it presents Layers
  1–3 as "the estimable consequence" and Layer 4 as "the identification," but **identification re-enters
  at Layer 2.** "Expected distribution under MAR" is **not an identified quantity estimable from one
  dataset**; any non-trivial reference is a *prior*, and the deviation is therefore *observed-minus-
  prior*, not a pure observable. The split is real and useful, but it is **prior/observable**, not
  **identified/non-identified**.
- **Strategic consequence:** the decomposition concentrates **all** the load-bearing novelty in its
  **least-proven, most-fragile** layer (the conditional reference), while the part that is already proven
  (marginal φ) needs none of it.

## The core objection (read this first; everything else follows)

Let `Y` = target, `X` = observed predictors, `R` = response indicator for `Y`. The only thing identified
from one dataset is the **observed-data law**: `P(X)`, `P(R|X)`, and `P(Y | X, R=1)` (the distribution of
`Y` among *respondents*).

MAR says `P(R | Y, X) = P(R | X)`, which implies `P(Y|X,R=1) = P(Y|X,R=0) = P(Y|X)`. But `P(Y|X,R=0)` is
**never observed** (those `Y` are missing). So:

> **MAR imposes no testable restriction on the observed-data law of a single dataset.** Every observed
> `P(Y|X,R=1)` is consistent with *some* MAR mechanism (Molenberghs §2). Therefore "observed minus
> expected-under-MAR," computed within one dataset against an identified MAR reference, is **identically
> ambiguous** — the reference can always be chosen to make the deviation zero.

A **non-zero** deviation can only arise if the reference encodes information *not in the observed-data
law* — i.e. a **prior about the un-censored distribution**:

- If the reference is `P(Y|X,R=1)` re-estimated from the observed (respondent) data → it equals the
  observed → **deviation ≡ 0** (structurally; the reference fits what it is compared to).
- If the reference is a **cross-dataset / shape prior** ("survey columns of this type have a Gaussian-ish
  upper tail") → deviation ≠ 0, and the signal is exactly the *prior's prediction error*. **This is what
  the LOD oracle already does**: it assumes a fitted-Gaussian X-model and measures truncation against it
  — and its recorded caveat ("Gaussian-oracle pass is necessary-not-sufficient") **is precisely this
  objection.** The deviation module is a *learned generalization of the oracle's profiled-MAR-null test*,
  and it **inherits the oracle's exact limitation**: its reach equals how well the assumed/learned
  un-censored shape transfers.

So the deviation is real, but it is **deviation-from-a-prior**, and the prior is doing the work. That is
not disqualifying — the North Star says the non-arbitrariness "lives in the prior." But it means the
decomposition must **name the reference as a prior**, not call it "the MAR expectation," or it hides its
own load-bearing assumption.

## Q1. Does the decomposition separate the consequence layer from the identification layer?

**Partially, and not where it claims.** The clean identified/non-identified boundary is **not** between
Layer 3 (deviation) and Layer 4 (deviation→δ). It is between the **observed-data law** (identified:
per-column distributions, the joint with `X`, the mask) and **everything built on a reference**
(prior-dependent). Identification leakage enters at **Layer 2**, because the reference is a prior. A
faithful statement:

> Identified: the observed-data law. Prior-dependent: the reference, the deviation, and δ. The deviation
> does not isolate "the consequence" from "the identification" — it *fuses the observed law with a
> reference prior*, and the residual non-identification (consequence→δ, many-to-one) sits in Layer 4 *on
> top of* the prior already injected at Layer 2.

So the decomposition has **two** prior-injection points (Layer 2 reference, Layer 4 manifold), not one,
and they are not independent. Presenting Layer 3 as "the estimable object" overstates what is identified.

## Q2. Is "observed-vs-reference deviation" the right object, or still smuggling assumptions?

**It smuggles the reference model as an assumption** — specifically a model of the *un-censored*
distribution. This is legitimate (priors are the door, §2) **but must be explicit**. The phrase
"expected under MAR" is the smuggle: it implies an identified counterfactual, when the operative object
is "expected under our **survey-manifold shape prior**." Rename it and the object becomes honest.

A cleaner specification of the object, with no smuggling:

> **The identified object is the observed-data law** (per-column observed distributions + cross-column
> joint + mask topology). **The product is a calibrated, abstaining map from that law to a δ-prior under
> an explicit, named manifold/reference prior.** "Deviation" is **one parameterization** of that map —
> useful iff the reference prior is valid and transferable — **not a more fundamental object** and not an
> axiom of the specification.

Under this framing the deviation is an *optional internal structure to be justified empirically*, not a
definition of what Lacuna estimates.

## Q3. Does Stage 0 (φ ≈ raw-ECDF) imply the load-bearing novelty must be in the reference/deviation layer?

**Yes — and that is the decomposition's central risk, not its vindication.** Stage 0 reached the
baseline with φ on the **raw marginal and *no explicit reference*.** φ's "reference" is **implicit**: the
supervised cross-dataset training bakes "what an un-truncated tail looks like" into its weights. So:

- The **proven** signal (marginal shape) needs **none** of Layers 2–3.
- The decomposition's **only** value-add over φ is the **predictor-conditional** reference — which is the
  exact thing that **failed before** (`transfer_features` OOF AUC 0.523 < 0.65, *worse* than marginal),
  and the exact thing most exposed to **proxy absorption** (P2.2b): a predictor-conditional reference fit
  on observed data absorbs *both* the MAR-explainable part *and* part of the δ-censoring (because the
  observed data *is* the censored data), so the deviation **under-detects** precisely where the proxy is
  strong. To avoid this, the reference must know the *un-censored* `P(Y|X)` — i.e. a stronger prior or
  auxiliary data — a **new dependency** the decomposition does not yet name.

So Stage 0 implies: the decomposition puts all its novelty in its **least-proven, theoretically-most-
exposed** layer. That is an argument for *testing Layer 2/3 in isolation before building*, not for
adopting the decomposition as the object.

## Q4. Is there an existing family that represents this object, or is it new territory?

**The object has a mature statistical home; the neural realization is a genuine hybrid.**

- **Statistical home: pattern-mixture and selection models with a sensitivity parameter** (Little; Rubin;
  Molenberghs & Kenward; Daniels & Hogan). There, δ is *defined* as the part the observed data cannot
  identify, and the analyst supplies a reference (the "identified" pattern) and a sensitivity parameter
  (the departure). **The decomposition is a neural re-derivation of exactly this framework** — which is
  reassuring (we are not inventing a new estimand) but carries a **hard warning from that theory** (Q5.3).
- **Neural realization: no off-the-shelf family.** It composes: a **distribution encoder** for Layer 1
  (DeepSets-with-quantile-pooling / Neural Statistician / ECDF-net), a **conditional reference** for Layer
  2 (conditional density estimator / normalizing flow / set-transformer over columns), and a **conditional
  two-sample / discrepancy head** for Layer 3 (deep MMD, conditional-discrepancy). The deviation head is
  literally a **conditional two-sample test**; that subfield exists, but gluing it to a learned reference
  and a calibrated δ-prior is a bespoke hybrid. So: **mature estimand, novel architecture.**

## Q5. The strongest arguments AGAINST the decomposition (most important)

Ordered by force.

1. **Non-identification of the reference (the core objection).** On one dataset, the MAR reference is not
   identified; the deviation is observed-minus-*prior*. The decomposition's clean consequence/
   identification story is therefore wrong — identification enters at Layer 2. (If the reference is fit on
   observed data, deviation ≡ 0; if it is a shape prior, it inherits the oracle's necessary-not-sufficient
   caveat.)

2. **Risk concentration in the unproven layer.** The proven signal (marginal φ) needs no reference; the
   only value-add (conditional reference) is the historically-fragile, proxy-exposed part. The
   decomposition maximizes reliance on the one component with a prior *negative* result.

3. **The sensitivity parameter is, by definition, what the data cannot see (classical-theory warning).**
   In pattern-mixture/selection theory δ is the *non-identified* departure. A deviation the data *can*
   compute is, by construction, the **identifiable shadow** of δ (e.g. the LOD truncation edge), **not δ**.
   So the deviation module — however good — recovers only the partially-identified footprint: it works for
   LOD and is *zero* for truly-non-identified own-value-at-matched-rate. **That is exactly our empirical
   pattern**, which means the deviation is doing the *identifiable* work that φ already does, and the real
   δ-inference is carried by the **manifold prior at Layer 4**, not by the deviation. Elaborate deviation
   machinery may only re-derive the identifiable part.

4. **Reference-fitting circularity / propagated error.** Estimating the reference `P(Y|X)` well enough to
   subtract is often **harder than the δ-classification itself**; the reference's errors enter the
   deviation as **false signal**, and a reference fit on censored data absorbs the very effect we want to
   isolate.

5. **Deviation is a lossy projection.** Forcing the signal through a subtract-the-reference bottleneck can
   **discard** information a direct supervised map would keep — absolute tail shape, cross-column
   co-missingness topology, the joint pattern. The "right object" may be the **full observed-data law +
   mask**, of which "deviation" is one lossy summary.

6. **Layer 2 is under-specified as stated.** "Estimate what the target would look like under a MAR-style
   reference process" collapses to either (i) the observed conditional (deviation ≡ 0) or (ii) a prior
   reconstruction (smuggles the manifold). The PI must choose (ii) **explicitly** and own the prior.

**Arguments in favor (for balance):**
- It correctly localizes *where the prior must enter* and is honest that consequence→δ is many-to-one.
- It is the natural neural form of a **mature, defensible estimand** (sensitivity analysis), so we inherit
  its theory and its guarantees.
- It **predicts the detectability spectrum** (non-zero for partially-identified LOD, zero for non-
  identified own-value) — a coherence point.
- **Governance value:** an *explicit* deviation + detectability state is far more **auditable** than a
  black-box φ — and Lacuna is a governance tool, so interpretability may justify the decomposition even at
  equal accuracy.

## What would make the decomposition sound (conditions, not a build)

1. **Rename Layer 2** "expected under the survey-manifold shape prior," not "under MAR." Own the prior.
2. **Demote the deviation** from "the object" to "an optional, auditable internal parameterization,"
   justified *empirically* against the marginal-φ baseline, not by axiom.
3. **Specify the object as:** observed-data law → (named manifold prior) → calibrated δ-prior +
   detectability + abstention. Identified part = the law; everything else is prior-dependent and labeled.
4. **Pre-register the only question that earns the decomposition** (Gate II of the architecture
   investigation): *does an explicit conditional reference/deviation add OOF signal over the marginal φ,
   while own-value stays lower — and without a proxy-absorption collapse?* Given Q5.2–Q5.4, the prior is
   that this is **hard**; treat a positive as the thing to be shown, not assumed.

## Bottom line for the PI

The decomposition is **scientifically coherent as a neural sensitivity-analysis model**, and its
governance/auditability story is genuinely attractive. But **as a specification of the inference object it
is mis-stated in one decisive way**: the identified object is the **observed-data law**, not the
deviation; the reference is a **prior**, not a MAR expectation; and identification therefore enters at
Layer 2, not only Layer 4. Corrected (rename the reference, demote the deviation to optional structure),
the object is sound — but its load-bearing novelty (the conditional reference) is exactly the unproven,
theoretically-exposed piece, and classical sensitivity theory warns that a computable deviation recovers
only the *identifiable shadow* of δ. **Specify the object as "observed-data law → named prior → calibrated
δ-prior"; keep the deviation as an auditable, empirically-gated option — not as the definition of what
Lacuna estimates.**
