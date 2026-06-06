# Detectable Compared to What? — the comparison class, the "unknown" class, and the two-stage design

*Conceptual registration, not implementation. **No code.** Status: **for PI review.** Records three
connected foundational points raised 2026-06-06: (1) "detectable" is always relative to a *named
comparison class*, never "air"; (2) a fourth output class, "unknown"; (3) a two-stage architecture
(MCAR-vs-not first, δ-prior only if not-MCAR). Governed by `NORTH-STAR.md` (§2 identification, §3½
manifold/M3, §5 oracle, §6 coverage). Engages critically; flags prior art and open questions.*

## 1. "Detectable compared to what?" — never against air; against a named plausible class

**You are right, and this recovers M3 (the manifold near-injectivity claim, §3½) that we let fade.** The
key correction: **we never compare a mechanism against "the infinity of mathematically-possible MAR
twins."** That infinity is the *worst case* Molenberghs uses to prove non-identifiability in the abstract;
it is **not** the comparison class Lacuna uses. Lacuna deliberately restricts to a **plausible** class —
that restriction *is* the manifold prior.

**Concretely, what our oracle already compares against (this is the precise answer to "compared to what?"):**
The `lod_oracle` / profiled-MAR-null does **not** compare B against air or against all of math. It compares
B (the MNAR mechanism, δ>0) against the **best-fitting MAR mechanism within a specific plausible family** —
the *profiled MAR-on-observed-predictors*, `σ(β₀ + β₁'·z_p)`, re-fit to mimic B's footprint at matched rate.
So the comparison class is **already a named, plausible class**, and that is what makes the verdicts mean
something:

- **own-value self-censoring → "flat" is credible *because the competitor is plausible*.** The best
  MAR-on-predictors (missingness driven by observed demographics — the *textbook* plausible MAR) reproduces
  the footprint. We are **not** saying "B is indistinguishable from a contrived universe"; we are saying "B
  is indistinguishable from a *genuinely plausible* MAR mechanism." That is a real, not artefactual,
  non-identifiability.
- **top-coding → "detectable" is credible *because its only MAR mimic is off-manifold*.** No
  MAR-on-predictors can reproduce a sharp truncation-at-τ; the MAR mechanism that could would have to depend
  on the target value itself (→ it *is* MNAR) or truncate at exactly τ for contrived reasons (→ off-
  manifold). So top-coding is distinguishable *within the plausible class* — which is exactly M3: its
  adversarial MAR-twin lies off the manifold, where nature does not build.

**Two operative comparison classes, both named, never "air":**
1. **Oracle:** the profiled MAR-on-observed-predictors family (the strongest plausible MAR competitor in a
   parametric family).
2. **Learned model:** the support of **`P_prior`** — the model's "detectable" means "distinguishable from
   the *other idioms/δ in the training mix*." As the survey-idiom vocabulary grows (item nonresponse, skip
   logic, top-coding, social-desirability…), this class becomes a richer, more realistic manifold, and
   detectability becomes relative to *that* — never to all-of-math.

**"If we only have B, compared to what?"** — We never have *only* B. We have B's footprint (from the one
real dataset) **plus** a comparison class **supplied by the prior/oracle** (the plausible alternatives). The
indeterminacy claim is always *"B's footprint is indistinguishable from {plausible alternatives}"*, computed
via the oracle/prior — not read off a single dataset against nothing.

**Binding consequence:** every "(non-)detectable" claim must state its **comparison class explicitly** —
"indistinguishable from the best plausible MAR-on-predictors / from the other idioms in `P_prior`," never an
unqualified "non-identifiable." The unqualified Molenberghs floor (vs all of math) is the *wrong* class for
a survey tool and should not be how we phrase results.

**Open research question (honest):** is *profiled MAR-on-observed-predictors* the **right** plausible class?
It is one parametric family. A richer plausible-MAR family (nonlinear in predictors, auxiliary-variable-
driven) might mimic more footprints (lowering detectability) or none (robustness). Defining and justifying
the plausible comparison class is itself a substantive contribution — and is precisely the manifold (M3)
question we should re-foreground.

## 2. The fourth class — "unknown"

**Agreed, and we already have the machinery — it just needs to be a first-class *label*.** For a
dissertation we cannot and must not claim to enumerate every mechanism science knows. So the honest output
space is **MCAR / MAR-ish / MNAR-ish / UNKNOWN**, where **"unknown" = the footprint is off the training
manifold ⇒ we likely have not learned this mechanism ⇒ escalate.** This is exactly the **OOD / abstention**
head already in the Level-1 design (§3, §6) — now elevated from a confidence flag to a **named class**.

Critical distinction (already encoded by the two-headed output, detectability ⟂ OOD):
- **in-manifold flat** → return the prior, *confidently uncertain*, **do NOT** label "unknown."
- **off-manifold** → **"unknown"**, abstain/escalate ("we don't know — likely a mechanism we haven't
  learned"). **Most common in natural missingness**, whose true mechanism is unobservable and may be novel.

So "unknown" is **not** "MNAR we can't quantify"; it is "outside what `P_prior` spans." That keeps the
honesty the project requires.

## 3. Two-stage design — MCAR-vs-not first (the identifiable part), δ-prior only if not-MCAR

**This is a strong reframing and it recovers old-Lacuna's genuine strength.** Split the pipeline:

- **Stage A — MCAR vs not-MCAR.** This is the **identifiable** axis (§2 solid ground: MCAR-vs-structured
  leaves *testable* restrictions on the observed-data law). It is the one thing the old Lacuna was actually
  good at. Output: a calibrated MCAR-departure score.
- **Gate:** if **MCAR** → missingness is ignorable → **no sensitivity analysis needed** (the column does not
  "deserve" one — directly answers the canonical triage question). Stop.
- **Stage B — only if not-MCAR** → the δ-prior + detectability + triage (the new column-primary work),
  reported **relative to the named comparison class** (§1).
- **"Unknown"** (§2) can fire at **either** stage (off-manifold footprint).

This elevates MCAR-departure from a **Level-1 Stage-2 *auxiliary*** to the **first gate** — a meaningful
reprioritization, and the right one: lead with what is *identifiable*, reserve the manifold-relative δ-prior
for columns that are actually non-MCAR.

**Why not Little's MCAR test (you're right):** Little's (1988) assumes the data are **multivariate normal**;
its χ² null calibration relies on it. Real survey data is overwhelmingly **non-normal** (skewed income,
ordinal Likert, counts, mixed types), so Little's can be mis-calibrated (wrong Type-I error, low power)
exactly where we operate. **Lacuna's variant is distribution-agnostic:** a learned MCAR-departure detector
trained on semi-synthetic MCAR-vs-structured masks across many *real, non-normal* survey X learns the
discriminator **without** the normality assumption, and is **calibrated** on the held-out ladder.

**Honesty / prior art (so we don't overclaim):** distribution-free MCAR tests already exist — e.g.
**Jamshidian & Jalal (2010)** (nonparametric MCAR test via homoscedasticity of imputed groups) and Kim &
Bentler. So "nonparametric MCAR detection" is not unprecedented. **Lacuna's contribution** is the specific
combination: a **learned, calibrated, survey-manifold-relative** detector, semi-synthetically validated, and
**integrated as Stage A of the δ-prior triage** (and reporting a *score* + abstention, not just a p-value).
We should cite Little, Jamshidian-Jalal, and position Lacuna as the learned/integrated variant — not as the
first distribution-free MCAR test.

## 4. What this changes (registered; deferred, no code)

- **Detectability is comparison-class-relative.** Every (non-)detectability claim names its class (profiled
  MAR-on-predictors / `P_prior` support); drop unqualified "non-identifiable." Re-foreground M3 (the
  plausible-manifold comparison) as a substantive part of the thesis, and treat "what is the right plausible
  comparison class?" as an explicit research question.
- **Fourth class "unknown"** = OOD/abstain, elevated to a first-class output label; distinct from in-
  manifold-flat; the honest default for natural missingness.
- **Two-stage architecture:** **Stage A MCAR-vs-not (the identifiable gate, distribution-agnostic, Little's-
  replacement, old-Lacuna's strength) → Stage B δ-prior + triage only if not-MCAR.** Elevate MCAR-departure
  from a Level-1 Stage-2 auxiliary to the **first gate**. Per-column output becomes
  **{MCAR-departure, δ-prior, detectability, unknown/abstain}**.
- **Framing of results:** "self-censoring is not detectable" → restated as **"own-value self-censoring is
  indistinguishable from the best *plausible* MAR-on-predictors at matched rate"** — a claim about a named,
  plausible competitor, not about air.

These update the framing of the Level-1 / cross-domain specs and will be folded in on approval. No code.
