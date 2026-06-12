# PROPOSAL — Lacuna Reformulated: Recoverability, Functional Bounds, and Certified Coverage

*Capstone of the 2026-06-12 session. Synthesizes: the T-review verdict (`docs/T-review-findings.md`),
the real-missingness showdown (`docs/REAL-MISSINGNESS-stage1-findings.md`), the semantic-channel
results (Arm-3 first-cut, commit dcdf90b), and the PI's operational reframing developed in
conversation. Status: PROPOSAL for PI/advisor decision. No code, no runs. If adopted, each
experimental component gets its own pre-registration in the 7714f0c discipline.*

---

## 1. The thesis statement

**Mechanism classification (MCAR/MAR/MNAR) was always a proxy for the question analysts actually
ask: "how wrong can my estimate be, and can anything observable shrink that?" We answer that
question directly: measure recoverability where it is measurable, bound it where it is not, and
certify the bounds on semi-synthetic ground truth.**

Supporting evidence, all from locked or committed runs:
- The matrix channel cannot detect the dangerous mechanisms (own-value: flat to features,
  networks, and on heavy tails even the truth-oracle) and does not need a network for the
  detectable ones (T1/T2/T3 fail/fail/fail; real-label showdown GBM 0.901 > network 0.878).
- The information driving the expert "income ⇒ MNAR" judgment lives in item SEMANTICS, and a
  behavior-trained encoder transfers refusal prediction to unseen instruments better than a
  frozen one (Arm-3 first-cut: pooled Spearman +0.121 vs +0.066, all 3 legs) while frozen
  suffices for don't-know — the topic-vs-sensitivity dissociation.
- A coded refusal is an EVENT, not a mechanism: refusal-vs-DK is 0.90-predictable from observed
  covariates ⇒ much of real refusal has a large MAR component; only the residual dependence on
  the hidden value (after conditioning on everything observed) is MNAR-relevant.

## 2. The object: recoverability (per column × per functional)

Replace the output taxonomy. Rubin's classes describe mechanisms; analysts need the
recoverability of THEIR estimate. Recoverability factors on two axes:

- **Identification** (theory): is the needed quantity a function of the observed-data law at
  all? (the only axis MCAR/MAR/MNAR ever measured; MNAR caps it by theorem)
- **Estimability** (practice): with this n, these predictors, this overlap, how well can it
  actually be computed? (MAR can be practically irrecoverable: weak proxies, support holes)

Lineage and gap: Mohan & Pearl's m-graph "recoverability" (JASA 2021) classifies queries as
recoverable GIVEN a known missingness graph — which no one ever has. Lacuna is the empirical
counterpart: measure the recoverable fraction directly; bound the unidentified residual with an
anchored prior; never assume the graph.

**Output classes (what the user receives; mechanism labels never surface):**

| class | meaning | action |
|---|---|---|
| **R0 recovered** | functional gap certifiably ≈ 0 (ignorable + well-estimated) | use the column |
| **R1 recoverable, measured loss** | tight bound; residual exposure quantified in analyst units | use with stated uncertainty |
| **R2 structurally exposed** | bound dominated by the δ-term | sensitivity analysis mandatory; anchored δ-range supplied |
| **R3 unknown** | off-manifold / no anchor / no coverage guarantee | abstain; escalate to a human |

MCAR/MAR/MNAR is DEMOTED to the generator layer: still required to punch calibrated holes and to
sweep coverage, never reported to the user.

## 3. The core instrument: the functional-gap experiment (semi-synthetic, oracle-split)

**Setup.** Punch holes with known mechanism into known values (semi-synthetic — the ONLY
substrate where ground truth exists; real missingness is unverifiable by nature). Train the best
imputer available on the observed view. Impute. Compare functionals (mean, quantiles, regression
coefficients) of the imputed column against truth. Per column.

**Case logic (corrected, central):** with a competent DISTRIBUTION-AWARE imputer (draws, not
point predictions — conditional-mean imputation preserves the mean while destroying
variance/quantiles):
- MCAR ⇒ gap ≈ 0 (observed distribution already unbiased — even for unpredictable columns).
- MAR ⇒ gap ≈ 0 (tilt is correctable from predictors).
- MNAR ⇒ gap > 0, and **the gap = the bias the analyst would have suffered, in their units.**

A large functional gap therefore isolates NON-IGNORABILITY specifically (not "MCAR or MNAR").

**Built-in negative controls:** MCAR and MAR holes punched into the same columns must show
gap ≈ 0 before any MNAR gap is interpretable. An imputer that fails the ignorable nulls has no
standing to cry MNAR.

**The oracle split (why semi-synthetic is the right substrate, not a compromise).** Because we
know the true generative mechanism, we can compute the ORACLE imputer (best recovery possible in
principle). This separates the three causes of a failed imputation, which on real data are
permanently confounded:
- **oracle gap** = information genuinely absent (the result),
- **network gap − oracle gap** = engineering shortfall (imputer/architecture/generators — the
  to-do list, with a progress meter).

## 4. The deployable output: the three-factor bound + coverage certification

On a REAL column no gap is computable (no truth). Decompose the gap:

> realized gap ≈ (missing rate) × (residual spread after best imputation) × (value-dependence δ)

- **rate** — observed.
- **residual spread** — MEASURED on real data: cross-fitted imputation on the column's observed
  cells (fully supervised; no labels we authored). This term also absorbs the estimability axis
  (weak proxies / support holes ⇒ large residual ⇒ wide bound).
- **δ** — never learnable (theorem); supplied as an ANCHORED RANGE by the semantic channel (§5).

**Certification:** across thousands of semi-synthetic cells spanning {datasets × mechanism
vocabulary × δ-grid × rates}, verify COVERAGE — the bound must bracket the realized gap at the
stated level (e.g., 95%). Only a coverage-certified bound may be printed on a real column.
Generator realism (§6) is what makes the certification believable.

**Income walkthrough (the running example):** 8% missing; proxies explain much of income,
residual spread $25k; semantic channel: income-class items sit ~95th percentile of measured
refusal propensity, anchored δ-range says refusers skew high by up to ~0.5 residual-SD ⇒ "your
mean may be understated by up to ≈ $1,000 (R2): if that changes your conclusion, run the
sensitivity analysis; here are the anchors." A column can be flagrantly MNAR with a TINY bound
(low rate / strong proxies / small residual) ⇒ honest output: "technically non-ignorable,
operationally ignorable" — the direct answer to "does the mechanism matter?": exactly in
proportion to B, and B is computable.

## 5. The semantic channel (where δ comes from) — three links, three epistemic statuses

1. **Text → nonresponse behavior (measured, learnable).** The trained encoder is an empirical
   regression of MEASURED refusal/DK behavior on question semantics over a cited corpus (ESS +
   NHANES + GSS; 6,700+ items; `docs/DATA-CITATIONS.md`). It replaces "intuition/experience"
   with an auditable artifact: corpus enumerable, mapping a checkpoint, calibration published,
   nearest measured neighbors printable. Arm-3 first-cut: training beats frozen on refusal
   transfer (all 3 LOIO legs), frozen suffices for DK — sensitivity cuts across topic; topic is
   already in commodity embeddings. Needs its own prereg (locked margins, more instruments,
   split resampling) to be dissertation-grade.
2. **Refusal → value-dependence (not learnable; anchorable).** A cited ledger per reference
   class: bracket follow-ups (SCF/CPS), record linkage, panel recovery, internal anchors
   (NHANES self-report vs measured weight — on disk). QUALITY BAR (binding): an anchor counts
   only if it shows value-dependence CONDITIONAL on observables (unconditional "refusers are
   richer" may be pure MAR). The dangerous classes are a small recurring family (~a dozen:
   money, health behaviors, illegal acts, stigmatized attitudes) — link 1 generates the
   anchoring worklist; the long tail needs no anchor (low refusal ⇒ small B regardless).
3. **The residual (never claimed).** No anchor ⇒ R3/unanchored: full sensitivity grid or
   escalate. Replaces the expert's confabulated "80%?" with an honest "we don't know."

## 6. Generators: pseudo-semi-synthetic (learned, conditional, realism-gated)

Upgrade the hand-coded mechanism vocabulary with a LEARNED conditional missingness model
(values bootstrapped/copula — the mask model is the contribution):
- Train G(mask | X, documented label) on real triples (the typed GSS .r/.d codes, ESS sentinel
  families, NHANES codes). Conditioning on documented labels is what keeps generated data
  LABELED (unconditional mask-matching is unlabelable — Molenberghs).
- The generator learns only the OBSERVABLE half (covariate-dependence of refusal — measured at
  0.90 predictable); the value-dependence is IMPOSED via the explicit δ-grid. Pseudo-semi-
  synthetic = real values × learned observable structure × imposed unidentifiable part.
- **Realism gates (necessary, not sufficient — stated honestly):** a discriminator near chance
  vs real masks; footprint statistics (refusal-by-covariate curves, spikes, gate AUCs) match
  held-out real distributions. Realism is verifiable in every OBSERVABLE respect; the
  unobservable respect is exactly the δ-sweep.
- Generated data NEVER enters validation/test (fake instruments are the slice-inflation trap);
  it amplifies training and densifies the coverage sweep only.

## 7. Where trained networks are load-bearing (each slot falsifiable, each with a frozen null)

| slot | job | null to beat | status |
|---|---|---|---|
| **semantic encoder** | text → behavior propensity, transfer to unseen instruments | frozen embeddings (+ keyword null) | first positive signal (Arm-3); needs prereg |
| **imputer** | shrink residual spread ⇒ TIGHTEN the certified bound | GBM/ridge imputers (vicious on tabular) | untested; fully supervised on real observed cells |
| **conditional generator** | realistic labeled training missingness | hand-coded mechanism vocabulary, judged by real-held-out detection + realism gates | spec'd (Part B); ~35% prior |

The matrix-channel DETECTOR slot is retired (settled negative, synthetic + real). Do NOT rebuild
"predict the gap from footprints" — that learned channel floored repeatedly (D2/D3); the bound
MEASURES two factors instead of predicting three.

## 8. Why this survives every objection raised against every prior version

- *"Statistics learn your generators"* — the deployed claims are measured (rate, residual) or
  cited (δ); generators only certify coverage, and are themselves realism-gated against real
  missingness.
- *"The data says nothing"* — correct, and priced: that is the δ-term, explicit and swept.
- *"A lookup table could read the label"* — the encoder attaches calibrated numbers, transfers
  across wordings/instruments (measured), and catches folk-prior failures (NHANES's top refusal
  item is HOUSEHOLD SIZE, not income).
- *"Refusal isn't MNAR"* — event ≠ mechanism; the MAR component is eaten by the imputer; δ
  prices only the conditional residual; anchors must be conditional.
- *"Based on what?"* — every number traces to a measurement or a citation; the residue is R3,
  said out loud.
- *"Why a network at all?"* — three slots, each facing a frozen shallow null, each killable by
  its own pre-registered test. The dissertation claim rests on whichever slots win their
  showdowns; the architecture does not require all three.

## 9. What is kept from existing Lacuna

The semi-synthetic machinery, generator vocabulary, oracle ladder, leakage/split discipline,
calibration code, and eval spine — all become load-bearing (the certification layer). v1.0's
reconstruction instinct becomes the imputer slot, first-class. The matrix-channel verdicts
become the chapter explaining why mechanism detection was the wrong target. Nothing from the
negative results is wasted; they are the argument for this design.

## 10. Sequencing (proposed; all gated on PI/advisor)

1. **Advisor conversation** — this document + the findings package; confirm the degree bar.
2. **Arm-3 pre-registration** (semantic encoder; locked margins; +instruments: WVS/ANES are
   plausible algorithmic grabs; split resampling; the income-and-drugs-neighbors interpretability
   check).
3. **Functional-gap harness + imputer showdown** (MCAR/MAR negative controls; oracle split;
   network-vs-GBM imputers on real observed cells).
4. **Coverage certification** of the three-factor bound across the generator family.
5. **Part-B learned generator** (existing spec, folded into §6 framing).
6. Anchor-ledger v1 for the dangerous-dozen classes (incl. the on-disk NHANES self-report-vs-
   measured-weight anchor as proof of concept).

---

*One sentence for the advisor: Lacuna stops claiming to detect what cannot be detected, measures
what can be, prices what can't, certifies the prices on ground truth it controls — and the
neural networks in it are exactly the components that demonstrably beat their shallow nulls,
no more and no fewer.*
