# Survey-Realism of Idioms, the Core Question, and the "Lab Coat" Hypothesis

*Conceptual registration, not implementation. **No code.** Status: **for PI review.** Records three
binding framings raised 2026-06-06: (1) idiom survey-realism audit; (2) the canonical problem statement;
(3) the data-vs-semantics ("domain knowledge in a lab coat") hypothesis. Governed by `NORTH-STAR.md`
(Scope — Lacuna-Survey; §2 identification; §5 oracle). Engages each critically, not as validation.*

## 1. Idiom survey-realism audit — "LOD" is the wrong name; the mechanism is top-coding

**The challenge is correct and we have been loose.** "LOD" (limit of detection) is an **instrument/assay**
concept — a measurement device cannot read below a threshold (NHANES *laboratory* biomarkers, environmental
assays). Per the North Star *Scope*, that is **Lacuna-Instrument**, **not** Lacuna-Survey. We should not
test assay-LOD here.

**But the mechanism we actually implemented is survey-realistic under a different name.** Our `lod_generator`
is a **sharp value-localized threshold censoring**: `P(miss) = σ(β₀ + β₁z_p + δ·1[z_t > τ])`. The
survey-realistic instance of that exact mechanism is **top-coding** — standard in public-use survey/census
microdata (CPS, ACS, SIPP income/wealth top-coded for confidentiality), and **unfolding-bracket / "prefer
not to say above $X"** designs. Identical observed footprint (truncated upper tail); **survey-legitimate
generative story.** So:

- **Rename the idiom `lod_top_coding` → `top_coding` (survey).** Drop "LOD" from the Lacuna-Survey idiom
  vocabulary; record assay-LOD as **explicitly out of scope** (a Lacuna-Instrument idiom).
- **NHANES projection restriction (binding):** when projecting NHANES to role-B (cross-domain plan), use
  **questionnaire + demographic** items only (income, depression, drug-use questions, demographics) — the
  *survey* component. **Exclude the laboratory/examination assay items**, whose missingness is instrument-LOD,
  not survey missingness. Otherwise we smuggle an instrument idiom into the survey instrument.

**Mapping our two tested idioms to the survey vocabulary (§3½ M1):**

| our generator | survey idiom it represents | survey-realistic? | detectability (measured) |
|---|---|---|---|
| own-value smooth self-censoring | **social-desirability / sensitive-item self-censoring** (income, drug, sexual behavior, voting underreporting) | **yes** | **flat** at matched rate (oracle + model) |
| step/threshold (`lod_top_coding`) | **top-coding / bracketing** (confidentiality threshold on income/wealth) | **yes (as top-coding, not LOD)** | **detectable** (Stage 0: 0.735) |

**The survey idiom vocabulary** that future work may draw from (and only this): item nonresponse
(don't-know/refuse), **skip logic / branching** (structural), **top-coding / bracketing**, **social-
desirability self-censoring**, attrition / unit nonresponse, don't-know-vs-refuse coding. **Instrument
concepts (assay LOD, calibration drift, sensor saturation) are out of scope.** The named-prior's
`idiom_vocabulary` (Level-1 spec §2.2) must enumerate only survey-realistic idioms.

## 2. The canonical problem statement (register this as THE question)

> **If we strip the column names (semantics) from every column of a survey table — and may pool many such
> tables — can the observed data *alone* tell us which columns deserve a sensitivity analysis, and how
> strong a δ-prior they warrant?**

This is **exactly** what Lacuna-Survey, as currently scoped, tests — because Level-1 has **no metadata
channel**: φ reads raw *values* (within-observed-standardized), never column names. The problem statement
is therefore the precise, honest operationalization of the no-metadata design. **It is registered as the
canonical question of record.**

Two refinements it sharpens:
- **The primary deliverable is column triage, not δ-magnitude.** "Which columns deserve a sensitivity
  analysis?" is a **screening/flagging** task — more robust and more useful than estimating δ magnitude.
  This maps onto the **detectability head** (data-informative → flag + δ-prior) and the **abstention head**
  (off-manifold → escalate). The δ-magnitude prior is secondary to the triage.
- **Honest caveat — "data alone" still leaks coarse semantic *type*.** Distributional *shape* implies a
  variable's kind (right-skewed-positive ≈ income/wealth; bounded 0–10 ≈ a Likert attitude). So semantics
  are not fully removed even without names. Matched-rate + within-observed standardization (what φ does)
  strips much of it, but the residual *type* signal means the experiment is "near-semantics-free," not
  "semantics-free." This should be stated whenever we report the data-alone result.

## 3. The "domain knowledge in a lab coat" hypothesis — profound, but idiom-dependent and oracle-gated

**The hypothesis** (PI): if the semantics-stripped data is definitively uninformative, then a large fraction
of sensitivity-analysis practice is driven not by the observed data but by **substantive knowledge carried
in the variable semantics** — current "sensitivity analysis" is "domain knowledge in a lab coat." **I agree
this would be a genuinely important finding**, and the project is uniquely positioned to test it. But three
qualifications make the claim defensible rather than overstated:

1. **It is idiom-DEPENDENT, not uniform — and that is the precise, stronger version of the claim.** Our
   evidence already shows a **spectrum**: top-coding is data-detectable (Stage 0: 0.735); smooth self-
   censoring at matched rate is **flat** (oracle + model). So the honest statement is **not** "the data is
   never informative." It is:
   > **For the flat-likelihood survey idioms — smooth social-desirability self-censoring at matched rate,
   > which is exactly the regime of the *sensitive items that dominate real-world MNAR concern* (income,
   > drug use, voting) — the observed data alone is uninformative, so sensitivity analyses on those items
   > are necessarily driven by semantics / domain knowledge, not data. For sharp idioms (top-coding) the
   > data IS informative.**
   That is more precise *and* more striking: the "lab coat" critique bites hardest exactly where applied
   sensitivity analysis is most often invoked.

2. **A "no" is a scientific result only when it is oracle-gated (§4.9, §5).** "The data can't tell us" from
   a weak model is confounded with "the model was too weak." The claim must rest on the **Bayes-optimal
   oracle** (information-theoretic ceiling on the observed-data law) or the real from-scratch model — not a
   proxy. We *have* that machinery (`lod_oracle`, profiled-MAR-null), which is what licenses calling the
   flat result genuine non-identifiability rather than a null. **The "lab coat" finding is only as strong
   as its oracle backing.**

3. **The most powerful form is a MEASUREMENT, not a binary.** Rather than "data informative: yes/no,"
   Lacuna can **quantify the decomposition**: how much calibrated δ-prior is recoverable from **data alone**
   (Level-1, no metadata) vs how much **only the metadata/semantics channel** can supply (the deferred
   §8.3 channel, itself held to semi-synthetic calibration). That literally measures the **"lab-coat
   fraction"** — the share of sensitivity-analysis information that is domain knowledge rather than data —
   per idiom and per column type. The project's Level-1-vs-metadata structure is *already* the apparatus
   for this measurement. **This is arguably the project's most important potential contribution**, and it
   reframes a possible negative (Level-1 flat) as a *positive* scientific measurement rather than a failure.

**Net:** point 3 is endorsed, sharpened to the spectrum form, gated on the oracle, and elevated from a
binary to a measurement. If Lacuna-Survey shows that data-alone recovers δ for top-coding but not for
sensitive-item self-censoring, and quantifies how much the semantics channel must carry, that is a
publishable claim about the **epistemology of sensitivity analysis**, independent of whether a deployable
tool results.

## 4. What this changes (registered; actioned later, no code now)

- **Idiom vocabulary:** rename `lod_top_coding → top_coding`; assay-LOD out of scope; enumerate only
  survey-realistic idioms in the named-prior. *(Spec/code change deferred.)*
- **NHANES projection:** questionnaire + demographic items only; **exclude lab/examination assays** (cross-
  domain plan §2). *(Deferred.)*
- **Deliverable framing:** primary output is **column triage / sensitivity-analysis flagging** (detectability
  + abstention), with the δ-magnitude prior secondary; report the **near-semantics-free** caveat.
- **Headline scientific aim (added):** **measure the data-vs-semantics ("lab-coat") decomposition** per
  idiom — Level-1 (data-alone) recovery vs the metadata channel — oracle-gated. The own-value flat result
  is reframed as a *measurement* of that decomposition, not a failure.

No code; these update the framing of the existing specs and will be folded in on approval.
