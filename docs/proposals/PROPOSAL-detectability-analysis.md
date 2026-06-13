# Detectability in Lacuna — Specification-Only Analysis

*Analysis document — **no implementation, no training, no architecture change** (binding; PI 2026-06-07).
Decides what detectability IS before deciding what builds it. Governed by `NORTH-STAR.md` (§2 per-mechanism
detectability; §6 coverage; "abstain, do not confabulate") and `MASTER-lacuna-survey-architecture.md` (§1
object, §3 comparison class, §5 two-headed output). Engages the design of record
(`PROPOSAL-Level1-design-spec.md` §3: detectability = oracle-calibrated info-gain head, Stage 2) against
the D2/D3 evidence (`REGIME-MAP-findings.md`, `D3-regime-transfer-findings.md`). Output: a decision rule
for whether detectability should be a **learned component**, a **derived metric**, or a
**reporting/governance quantity** — with the experiments that would settle it.*

---

## 0. The three D3 facts this document must explain

1. **Coverage-distance predicts transfer** (Pearson −0.91 across 8 pre-registered (D, pool) pairs).
2. **Coverage-distance predicts calibration failure** (ECE rises monotonically with coverage-distance:
   0.16 → 0.26 → 0.34).
3. **The model becomes confidently wrong as coverage-distance increases** (entropy *falls* 0.83 → 0.63 →
   0.51 while accuracy falls below chance) — the opposite of graceful degradation.

Fact 3 is the constraint that reshapes the design question: **any detectability signal that is itself a
model output inherits the failure mode it is supposed to detect.**

## 1. What is detectability in Lacuna?

Per the North Star and the revised object (`ARCHITECTURE-OBJECT-revised.md`), Lacuna's output is
**calibrated δ-prior + detectability + abstention**. Detectability answers, for a flagged column:

> **"How much does this column's observed-data footprint actually inform δ — relative to a NAMED plausible
> class — as opposed to the δ-prior simply returning the manifold prior?"**

Binding properties already settled:
- **Per-mechanism, never universal** (NORTH-STAR §2): top-coding is blatantly detectable; smooth own-value
  self-censoring at matched rate is near-indistinguishable *from the best plausible MAR-on-predictors*.
- **Comparison-class-relative** (MASTER §3): every (non-)detectability claim names its class (profiled
  MAR-on-predictors / `P_prior` support). Unqualified "non-identifiable" is forbidden.
- **Distinct from OOD/abstention** (Level-1 spec §3): *in-manifold-flat* ("we know this footprint; it is
  uninformative; return the prior, **confidently uncertain**, do NOT abstain") is a different state from
  *off-manifold* ("a mechanism/geometry we haven't learned; **UNKNOWN**; abstain"). The two-headed output
  exists to disambiguate exactly these.
- It is a **governance quantity**: its consumer is the survey analyst deciding whether a sensitivity
  analysis is data-driven or "domain knowledge in a lab coat" (the lab-coat fraction). It must therefore be
  **honest under failure** — a detectability signal that is overconfident off-manifold is worse than none.

## 2. Three candidate definitions — and why they are not rivals

| candidate | the question it answers | computed from | available at deployment? |
|---|---|---|---|
| **(A) Oracle distinguishability** | *Is there signal in principle?* — Bayes-error of MNAR-vs-best-plausible-MAR (profiled-MAR-null) for a known (idiom, dataset, δ, rate) cell | the generating cell + fitted X-model | **No** (needs known truth) — training/eval only |
| **(B) Coverage support** | *Is this footprint inside the region where the learned posterior has support?* — distance from the training pool in footprint-regime space (D2 metric) | the column's observed values + the training corpus; **model-independent** | **Yes** (pure data statistic) |
| **(C) Calibration support** | *Can the posterior's numbers be trusted here?* — empirical ECE/entropy behavior of the model's output | the model's outputs vs held-out truth | held-out eval yes; runtime only via proxies (the posterior's own entropy/info-gain) |

**The key structural observation: these are not competing definitions of one thing — they are three layers
of one chain:**

> **(A) defines** detectability (the ground truth: signal exists relative to the named class) ·
> **(B) is the precondition** for trusting any *estimate* of (A) at deployment (no footprint support ⇒ no
> trustworthy estimate of anything) · **(C) is the observable symptom** when (B) is violated (D3 fact 2/3).

Treating (B) or (C) as *the definition* would be a category error — coverage says nothing about whether a
*covered* footprint is informative (own-value is covered AND flat), and calibration is a property of the
estimator, not of the data. Treating (A) alone as the operational quantity fails at deployment, where the
oracle is uncomputable. **Detectability-as-deployed must be: an estimate of (A), gated by (B), validated
via (C).**

## 3. Which definition is most North-Star-consistent?

**(A) oracle distinguishability is the definition** — it is exactly the North Star's per-mechanism,
comparison-class-relative notion (§2), and the Level-1 spec already designates the oracle as the
detectability *target* source. But the North Star **also** demands §6 coverage honesty ("Lacuna can only
recognize what spans its training; abstain, do not confabulate") — and D3 has now shown empirically that an
estimate of (A) produced by the model **cannot be trusted without (B)**: the model's confidence *rises* as
support vanishes. So the North-Star-consistent answer is the **layered** one:

> **Detectability = estimated distinguishability-from-the-plausible-class (A), reported ONLY within
> footprint support (B), validated against the oracle and held-out calibration (C).**

This is a sharpening, not a revision, of the design of record — the Level-1 spec's "detectability ⟂ OOD"
two-head structure anticipated it. What D3 changes is the **epistemic ordering**: the OOD/support gate is
not an auxiliary safety feature beside detectability; it is the **precondition** for detectability meaning
anything. And D3 supplies a candidate gate that is **model-independent and already validated as a
predictor** (coverage-distance), where the design of record had left OOD as another learned/fitted
component (Mahalanobis/kNN on learned footprints).

## 4. Is a learned detectability head necessary?

Decompose what a runtime detectability output needs, against what exists **without any new learned
parameters**:

| required function | derived candidate (no new learning) | known gap |
|---|---|---|
| **Off-manifold gate** ("is the estimate meaningful at all?") | **coverage-distance** (D2 metric vs the training corpus) | hand-designed 4-D space; **pool-level** evidence only (D3 §4); per-column, blind to mask topology |
| **In-manifold informativeness** ("how much does this footprint move the posterior?") | **posterior info-gain** `KL(posterior ‖ prior_marginal)` — computable from the *existing* δ-prior head's output, no extra parameters | it is a model output ⇒ trustworthy **only inside the gate**; its agreement with the oracle in-manifold is **untested** |
| **Validation target** | the **oracle** (profiled-MAR-null Bayes-error) per training/eval cell | necessary-not-sufficient caveat (fitted Gaussian X-model); unavailable at deployment (fine — it is the *validator*, not the runtime signal) |

**The self-reference argument (the load-bearing new point).** The design-of-record head is a regression
from the model's internal representation `z` to oracle informativeness. But D3 fact 3 shows the
representation pipeline degrades *confidently* off-manifold — the same `z` that makes the δ-prior
confidently wrong would feed the detectability head. A learned head can only learn "I am off-manifold" from
**off-manifold supervision**, which for genuinely novel regimes does not exist *by definition* (we can
train it on held-out-domain examples, but D3 showed the failure appears precisely on regimes *unlike all
training domains* — wealth — and a head trained on moderate-regime holdouts has no more support at the
wealth boundary than the δ-prior does). **Coverage-distance does not have this problem**: it is computed
from the column's data and the corpus, never passes through the model, and was validated in D3 as
predicting exactly the failures the head would need to catch.

**Against pure-derived (the honest other side):** coverage-distance is idiom-blind — at *identical*
coverage, top-coding is detectable and own-value is flat, and only something that reads the footprint
(oracle at eval; posterior info-gain at runtime) can tell them apart. Posterior info-gain in turn has
**zero** validated evidence yet: D3 measured entropy/ECE of the *binary δ-head*, not info-gain-vs-oracle
agreement. It is possible that even in-manifold, the posterior's info-gain disagrees with the oracle (e.g.
overstates informativeness for own-value) — in which case a *small calibrated mapping* (≈ a learned head,
though perhaps only a 1-D monotone recalibration, not a representation-reading head) becomes necessary.

**Position after D2/D3 (the recommendation):**

> **Detectability should be, until proven insufficient: a DERIVED metric — coverage-distance gate +
> posterior info-gain — validated against the oracle, surfaced as a GOVERNANCE quantity in the
> manifest/report. A learned head is NOT yet justified. The burden of proof is on the head.**

This is the minimal-structure answer (no new parameters, no new failure modes, every component already
exists or is already validated), and it converts the design-of-record's Stage-2 head from "planned
component" to "conditional component, pending E-JUSTIFY (§7)."

## 5. What the existing evidence supports (position by position)

| position | supporting evidence | weakness of the evidence |
|---|---|---|
| **Coverage-distance as the gate** | D2: postdicts the SCF transfer split (wealth isolation 10×); D3: **pre-registered** coverage predicts transfer (r = −0.91) AND calibration failure (ECE monotone) across 8 pairs — model-independent, deployable | pool-level only (size-control did not isolate single-dataset regime effects); 4-D hand metric; n = 3 held-out domains; thresholding (gate vs continuous) unstudied |
| **Learned head (design of record)** | the oracle target exists and is the right validator; in-manifold idiom separation needs *something* that reads the footprint | **no direct evidence yet** that a learned mapping beats the free posterior info-gain; the self-reference problem (§4) is evidence *against* its off-manifold half; the D3 entropy behavior shows the representation it would read is misleading exactly where it matters |
| **Posterior info-gain (derived, in-manifold)** | free (no parameters); structurally the right quantity (the spec itself defines detectability as this KL); within-coverage calibration is moderate (matched-arm ECE 0.16) | **untested against the oracle** — the single biggest evidence gap; D3 fact 3 caps it: meaningless outside the gate |
| **Reporting/governance quantity** | whatever the estimator, the consumer is governance (lab-coat fraction; manifest `info_gain` block already specified); the North-Star deliverable is column *triage*, not a new score to optimize | reporting alone does not answer *which number* to report — it is the wrapper, not the resolution |

**Note the asymmetry:** the derived-gate position rests on *pre-registered, replicated* evidence (D2→D3);
the learned-head position rests on *design intent* from before D3 existed. D3 was not designed to test the
head — but both of its calibration facts landed on the side of "model outputs degrade where the head would
be needed most."

## 6. The three-state output this implies (unchanged from the design of record, re-grounded)

| state | trigger | δ-prior output | detectability output | abstain? |
|---|---|---|---|---|
| **covered + informative** | low coverage-distance, high (validated) info-gain | data-updated posterior | high | no |
| **covered + flat** (own-value) | low coverage-distance, info-gain ≈ 0 | ≈ `prior_marginal` | **low — "confidently uncertain"** | **no** (this is an *answer*: the data genuinely cannot distinguish; sensitivity analysis is semantics-driven here) |
| **uncovered** (wealth-like) | high coverage-distance | **withheld** | **undefined — not "low"** | **yes — UNKNOWN** |

D3's contribution is the empirical demonstration that without the third row's gate, the model silently
produces the first row's confidence in the third row's situation.

## 7. E-JUSTIFY — the experiment that would justify a learned head

*Eval-only; no architecture change needed to RUN it (the candidate head for comparison can be a post-hoc
probe, not a wired module). Pre-registered before running.*

**Setup.** On the existing corpus + oracle machinery: for a grid of (idiom ∈ {top_coding, own_value},
dataset, δ, rate) cells, compute per cell — (i) **oracle informativeness** (profiled-MAR-null Bayes-error →
info measure; the target); (ii) **posterior info-gain** from the existing trained δ-prior (derived
candidate); (iii) a **probe head** (small regressor from the model's pooled representation to the oracle
target, trained leave-domain-out — the learned candidate, evaluated as a probe so nothing ships).

**Decision rule (pre-register thresholds before running):**
- The head is **justified** iff, *within coverage* (gate fixed to coverage-distance for both candidates),
  the probe's leave-domain-out agreement with the oracle (rank corr + calibration) **exceeds** the derived
  info-gain's agreement by a pre-registered margin, **and** the probe correctly separates top_coding (high)
  from own_value (low) at matched coverage where info-gain fails to.
- If the probe wins only **in-distribution** but not leave-domain-out, the head is **not** justified (it
  would inherit the transfer gap, the P2.2c lesson).

**Out of scope for the head either way:** the off-manifold gate. Per §4, no learned component is eligible
for the gate role; E-JUSTIFY only adjudicates the *in-manifold informativeness estimator*.

## 8. E-FALSIFY — the experiment that would falsify the need for one

*Also eval-only. The mirror image, runnable as the same study.*

The need for a learned head is **falsified** iff the fully derived stack already does the job:
1. **In-manifold validity:** posterior info-gain, within the coverage gate, agrees with the oracle
   (pre-registered rank-corr/calibration threshold) and separates top_coding from own_value at matched
   coverage — i.e. reproduces the known idiom spectrum without any new parameters.
2. **Gate validity:** coverage-distance, thresholded by a held-out-calibrated cut, flags the D3 failure
   cases (wealth-as-target arms, the below-chance + low-entropy regime) at a pre-registered hit rate while
   passing the covered arms — i.e. the three-state table (§6) is achievable with the derived gate alone.
3. **Honesty composition:** on held-out domains, the *gated* info-gain's calibration failure cases are
   concentrated in the gated-out region (the D3 ECE pattern disappears once gated).

If 1–3 hold, detectability is **a derived metric + governance reporting**, the Stage-2 head is struck from
the plan, and the saved complexity is the result. If 1 fails specifically (info-gain disagrees with oracle
in-manifold), E-JUSTIFY's probe comparison decides whether learning closes that gap. If 2 fails
(coverage-distance does not gate well at column level), that is a finding **against the D2 metric's
column-level sufficiency** — neither candidate survives unchanged, and the result redirects to improving
the support metric (e.g. adding mask-topology coordinates), not to a head.

**Note:** E-JUSTIFY and E-FALSIFY are one study with one pre-registration — two named outcomes of the same
measurement (oracle vs derived vs probe, gated, leave-domain-out). Estimated scope: oracle sweeps + eval
forward passes on existing checkpoints/corpus + one small probe; no architecture change, no new data.

## 9. Recommendation (for PI decision — nothing started)

1. **Adopt the layered definition** (§3): detectability = estimated distinguishability from the named
   plausible class, **gated by footprint support, validated by the oracle**. Fold into MASTER §5 wording.
2. **Reclassify the Stage-2 detectability head as CONDITIONAL** — pending the E-JUSTIFY/E-FALSIFY study.
   Default-off. The off-manifold gate role is **permanently closed to learned components** (the
   self-reference argument, D3 fact 3).
3. **Treat coverage-distance as the candidate gate** — a *derived* metric with pre-registered D3 validation
   — while recording its open limits (pool-level evidence, 4-D hand metric, no mask-topology axis).
4. **The deliverable surface is governance**: whatever estimator wins, detectability ships as manifest/
   report fields ({δ-prior, detectability, coverage-state, UNKNOWN}) feeding column triage and the
   lab-coat-fraction measurement — never as a training objective to maximize.
5. **Next concrete step, if approved:** pre-register and run the single E-JUSTIFY/E-FALSIFY study (§7–§8).
   It is eval-only, uses existing machinery (oracle, trained models, corpus, D2 metric), and its two named
   outcomes directly select between "derived metric" and "learned component."

---

*No implementation, training, or architecture change is performed or scheduled by this document. It fixes
the definition of detectability, assigns the burden of proof to the learned head, and specifies the single
pre-registered study that would settle the head's necessity. Execution awaits PI decision.*
