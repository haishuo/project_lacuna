# Architecture-Fitness Analysis — is the Lacuna encoder the right machine for δ-estimation?

*Analysis + proposal. Status: **for PI review**. No model was trained or tuned for this document; no
`lacuna/` model code or charter was changed. The one numerical artifact is an explicitly-labelled
illustrative pooling probe (`scripts/illustrative_pooling_probe.py`, §3), not a Lacuna run. Governed
by `NORTH-STAR.md` (§2 identification line, §3 reframed estimand, §3½ manifold, §6 coverage, §8
validation ladder) and the closed `DECISION-MEMO-P2.2c-consolidation.md` (the three detectability
levels).*

---

## 0. The question, sharply

P2 cut the original Lacuna pipeline at the seam `evidence = encoder(...)` (`assembly.py:363`) and
stapled on a δ-bin head (`delta_head.py`, `conditioned_head.py`, `feature_only_head.py`) to estimate
a **continuous sensitivity parameter δ** — the own-value self-censoring slope, or the LOD/top-coding
log-odds jump — from the observed-data footprint. The original pipeline was built for a *different*
question: 3-class MCAR/MAR/MNAR **classification**. The PI's concern, in his analogy: *are we feeding
image data to an MLP when the task needs a CNN?*

The single sharpest empirical clue (DECISION-MEMO §6): hand-computed **observed-marginal order
statistics** of the target column (`consequence_features.py`: quantiles, max, tail mass, skew) carry
the transferable δ signal — a logistic regression on them reaches out-of-family binary δ=0-vs-δ2.5
AUC ≈ 0.68–0.73 — while the **encoder-based model floors at base-rate**, and a Bayes-optimal **oracle**
proves the LOD signal is present in principle (BE→0). The signal lives in the *shape of the observed
target marginal*; hand order-stat features capture it; the learned encoder does not.

This document asks whether that gap is an **architectural** fact (the encoder cannot, or is strongly
biased against, computing the functions δ-estimation needs) and, if so, what to do about it.

**Bottom line up front (verdict: PARTIALLY valid; see §7).** The encoder is well-matched to what it
was designed for — cross-**column** structure of the *missingness mask* — and structurally
ill-matched to what δ-estimation needs — within-**column**, across-**row** *order-statistic shape of
the observed values*. But the defect is **not** a strict representational impossibility (the pooling
primitive can approximate quantiles in isolation; §3). The binding architectural problems are two
concrete, fixable design choices: (i) two-stage pooling **mean-collapses the per-column across-row
marginal before any order statistic can be read**, and (ii) the P2 conditioning head then takes a
plain **mean** over the target column's row-reps (`conditioned_head.py:67`), averaging the very tail
δ lives in. **And** even a perfect distributional architecture is capped by the oracle and by an
independently **low out-of-family transfer ceiling** (§6) — so architecture is **necessary but not
sufficient**.

---

## 1. What δ-estimation actually requires of an architecture

Write the two idioms as functions of the observed data so we can ask, concretely, what must be
computed. In both, X is real survey data; we impose holes; δ is known (the semi-synthetic ladder,
NORTH-STAR §8.1); the missingness rate is **matched across δ** so the *count* of holes carries no
information and the signal is forced into the **shape** of what survives.

### (a) Own-value smooth self-censoring
`P(miss | t) = σ(β₀ + β₂·z_t)`, δ ≡ β₂, MAR ⇔ δ=0. Larger t is censored more often, so the
**observed** target marginal is the true marginal reweighted by `σ(−β₂ z_t)` — a smooth deficit in
the upper tail. To recover δ an estimator must, in effect, compare the **observed conditional/marginal
distribution of t** against what it would be under MAR at the same rate: the signal is a smooth
distortion of the observed marginal's **upper-tail shape**, ideally read **conditional on the observed
predictors** (because a predictor correlated with t mimics part of the deficit — the proxy-absorption
confound, P2.2b). This idiom is the demonstrated **flat-likelihood** case (NORTH-STAR §8.2): at matched
rate on real X the distortion is near-invisible even to the oracle-equivalent.

### (b) LOD / top-coding (step censoring)
`P(miss | t) = σ(β₀ + β₁·z_p + δ·1[z_t > τ])`, δ = the log-odds **jump** at a fixed threshold τ
(`lod_generator.py`, `lod_oracle.py`). Above τ a constant extra censoring probability applies, so the
observed target marginal has a **truncated / depleted upper tail above τ** — a sharp edge, not a smooth
shift. To recover δ: detect and quantify the **truncation of the observed upper tail** — concretely, the
**tail mass above τ**, the **upper quantiles** (q90/q95/q99), the **max**, and the **gap between high
quantiles** (`consequence_features.py` computes exactly these). The oracle proves this carries strong δ
information in principle (BE→0; DECISION-MEMO §2).

### The required function class, distilled
Both idioms reduce to: **compute order-statistic / ECDF / tail-shape functionals of the observed
target column across rows, optionally conditional on observed predictors, and compare them to a
matched-rate MAR reference.** Four properties:

1. **Order statistics across rows.** Quantiles, max, tail mass, inter-quantile gaps of the *observed
   values* of one column — not their mean. Matched rate deliberately removes the mean-rate cue, so the
   discriminative content is in the high-moment / extreme-order region of the marginal.
2. **Per-column, within-column.** The relevant distribution is one column's values across all rows;
   it must **not** be blended with other columns before the shape is read.
3. **Conditional on predictors (for idiom a).** The comparison reference is the predictor-conditional
   expectation of the target marginal; raw marginal shape alone is confoundable by proxy correlation.
4. **Permutation-invariant over rows.** Rows are exchangeable; the functional is a set function of the
   observed column values. (This part the encoder gets right — see §2.)

The matched-rate design is what makes this **hard for an averaging machine specifically**: it strips out
exactly the cue (rate = a mean) that mean-pooling is built to read, and leaves only the cue (tail shape)
that mean-pooling is built to discard.

---

## 2. What the current architecture computes (inductive biases, from the code)

`LacunaEncoder` (`encoder.py`) is a hierarchical set-transformer / DeepSets-family encoder:

- **TokenEmbedding** (`encoder.py:96`): each cell becomes a token
  `[value, is_observed, mask_type, feature_id]` (`tokenization.py`, `TOKEN_DIM=4`). The scalar value
  passes through a **linear** projection `Linear(1, h/4)`, **scaled by the observed indicator** (missing
  ⇒ value contributes 0; `encoder.py:162`). So the raw value enters only as a linear feature plus
  learned obs/mask/position embeddings.
- **TransformerLayer × n** (`encoder.py:200`): self-attention **row-wise** — each row is a separate
  sequence of ≤`max_cols` feature tokens (`x.view(B*max_rows, max_cols, h)`, `encoder.py:267`).
  Attention is **over columns within a row**. This is precisely a *cross-column dependency* detector —
  "does feature j's representation depend on feature k's value in the same row" — which is the textbook
  MAR signature. The docstring says as much (`encoder.py:8–16`).
- **RowPooling** (`encoder.py:394`): collapse the ≤`max_cols` feature tokens of each row into **one row
  vector** (attention/mean/max over columns). After this step the per-column identity of a value is
  gone — each row is a single blended `hidden_dim` vector.
- **DatasetPooling** (`encoder.py:461`): collapse `max_rows` row vectors into one `evidence` vector
  (attention/mean/max over rows), then project to `evidence_dim`.

Two facts matter for δ:

**(A) The two-stage order destroys the per-column across-row marginal.** Pooling is **features→row,
then rows→dataset**. The object δ lives in — the distribution of *one column's values across rows* — is
never formed: by the time we aggregate across rows (where a marginal would be summarized), each row has
already been reduced to a blend of all its columns. The architecture is built to summarize **rows as
units** and **columns as interacting tokens within a row**; it is not built to summarize **a single
column as a distribution across rows**. That is a cross-column machine being asked a within-column
question.

**(B) The default aggregators are averages.** Attention pooling (`encoder.py:341`) is a
**content-weighted mean** (`softmax` weights · values, `einsum("bs,bsd->bd")`, `encoder.py:389`); mean
pooling is a plain average. Both are first-moment operators by construction. The only built-in
non-averaging aggregator is `max` pooling, which is not the default and is not selected in the P2 runs
(the configs use `attention`). Nothing in the pooling stack is an explicit sorting / quantile / ECDF
operator.

**The original design already knew the encoder was not enough — and bolted on the *complementary*
feature set.** In the full `LacunaModel` (`assembly.py:284`), the MoE gate is fed not only by `evidence`
but by an explicit `MissingnessFeatureExtractor` (`missingness_features.py:635`) that hand-computes
**missing-rate statistics across columns and cross-column missingness correlations**
(`missingness_features.py:14–16`) — because "reconstruction errors alone cannot distinguish MCAR from
MAR" (`missingness_features.py:6–9`). This is the decisive tell: **the original architecture
compensated for the encoder with explicit hand-features of the *missingness mask*** — exactly the
statistics the *classification* task needs. Those features are about **where the holes are** (mask
structure across columns), **not about the shape of the observed values** (the target marginal). δ-
estimation needs the latter. So:

- The encoder's inductive bias (cross-column mask interactions, first-moment pooling) is **aligned**
  with MCAR/MAR/MNAR classification — its design target.
- P2 **dropped the MoE and the `MissingnessFeatureExtractor`** (the running log; `delta_head.py:6–11`),
  keeping the encoder and adding a δ head. It removed the *one* component the original design used to
  inject explicit pattern statistics — and then, much later, re-added a *different* explicit feature
  block (`consequence_features.py`) at the head. The P2 architecture is therefore the encoder being
  asked, alone, to surface a statistic family (observed-value order statistics) that even the original
  design did not trust the encoder to surface for its *own* (mask-structure) statistics.

**(C) The P2 conditioning head averages the tail away.** `TargetConditionedDeltaModel._pool_target`
(`conditioned_head.py:56–67`) is the one place the target column's identity is recovered: it gathers
the target column's per-row token-reps and then does a **masked MEAN over rows**
(`(tgt*rm).sum(1)/rm.sum(1)`, `conditioned_head.py:67`). So the head receives the **mean of the target
column's representation across rows** — a first moment — exactly when the signal is in the tail. Whatever
the transformer layers encode per row, the across-row aggregation that reaches the head is an average.
The consequence-feature experiment (`consequence_features.py`, concatenated at the head,
`conditioned_head.py:79–87`) was an attempt to inject the missing order statistics — and is, in effect,
a hand-built partial ECDF front-end. It moved the in-distribution number (LR on those features AUC 0.94)
but did not transfer (§6).

---

## 3. The representational gap — can averaging pooling represent order statistics?

The naive strong claim — "attention/mean pooling **cannot** compute order statistics" — is **false**,
and an honest report must say so. DeepSets/set-transformers are universal approximators for
permutation-invariant functions given sufficient latent width (Zaheer et al. 2017; the standard
construction recovers an order statistic by summing soft threshold indicators `1[x>t]` over a grid of
t — an ECDF — then inverting). A GELU MLP inside the transformer FF can build soft indicators, and
**content-weighted attention pooling can place its weight on the largest elements**, softly selecting
the tail. So *strict representability at the primitive level is not the wall.*

**Illustrative probe (`scripts/illustrative_pooling_probe.py`; ILLUSTRATIVE ONLY, not a Lacuna run).**
To put numbers on the *inductive bias*, I trained the project's actual `AttentionPooling` layer (plus a
per-element MLP and a linear head) to regress set→scalar targets on sets of 64 scalars whose mean and
shape vary independently. Test R² (mean control / p90 / max):

| set→scalar regressor | mean | p90 | max |
|---|---|---|---|
| **AttentionPooling** (the encoder primitive) | 1.000 | **0.971** | **0.986** |
| Mean pool (pure average) | 1.000 | 0.898 | 0.704 |
| ECDF / quantile front-end | 0.997 | 0.980 | 0.887 |

Read this carefully and skeptically:

- **The encoder's pooling primitive *can* approximate p90 and the max** (R² 0.97–0.99) when the task is
  *exactly* that, single-level, with direct supervision on the order statistic and abundant data. This
  **refutes the strong "cannot represent" hypothesis.** Content-weighted attention is genuinely better
  than a plain mean at the max (0.986 vs 0.704), because it can softly argmax.
- **Plain mean pooling is materially worse at the extremes** (max R² 0.704) — and the P2 conditioning
  head (§2C) uses *exactly* plain mean pooling over rows. So the specific aggregator on the path that
  reaches the δ head is the weakest one for this job.
- The ECDF front-end matches or beats the learned poolers with a trivial linear head and no pooling to
  learn — it makes the order statistic the *input*, not something to be discovered.

So the gap is **not** "impossible to represent." It is the gap between *can-be-fit-in-isolation* and
*is-the-path-of-least-resistance-and-transfers*. Three reasons the in-isolation success does **not**
carry over to the encoder on the δ task:

1. **Wrong level / mean-collapse (the real architectural defect).** The probe pools a *bare set of the
   target scalars* in one stage. The encoder never has that set: it pools features→row→dataset (§2A),
   and the conditioning head's across-row aggregation is a **mean** (§2C). The order-statistic-capable
   operator (attention pooling) is applied **over columns within a row** and **over already-blended row
   vectors** — never over the clean across-row marginal of one column. The capable primitive is wired to
   the wrong axis.
2. **Supervision is on δ-bins, not on quantiles.** The probe supervises directly on p90. The δ model is
   supervised on a 7-way (or coarse) bin label via RPS. It must *discover* that "compute the observed
   upper-tail shape" is the useful intermediate, with no architectural prior pointing there, from a weak
   label. Universality says the function exists in the hypothesis space; it says nothing about whether
   SGD finds it from this supervision.
3. **The metric is out-of-family transfer, not in-distribution fit.** The probe is in-distribution. The
   δ headline is leave-datasets-out (NORTH-STAR §5, §8.1). A soft-ECDF the encoder learns to fit the
   *training* columns' shapes need not transfer to *held-out* columns/datasets — and indeed the
   hand-feature LR, which hard-codes the order statistics, itself transfers only weakly (AUC ~0.68; §6).

**Conclusion of §3.** The encoder is not *incapable* of order statistics in the strict sense, but its
inductive bias is **first-moment, cross-column, row-as-unit**, and the δ signal is **tail-shape,
within-column, column-as-distribution**. The architecture makes the right computation an
uphill, non-transferring special case rather than the natural thing it computes — and the one place P2
recovers the target column, it averages. That is the MLP-vs-CNN mismatch, made concrete: not "the MLP
can't represent edges," but "the MLP has no convolutional prior, so it won't learn edge detectors that
transfer the way a CNN's do."

---

## 4. Is the original architecture even the right *family*?

DeepSets/set-transformers compute permutation-invariant functions of the form `ρ(pool_i φ(x_i))`.
Order statistics are the **known hard edge** of sum/mean-pooling DeepSets: representable in principle
(via the ECDF-of-soft-indicators construction) but requiring either large latent width or an explicit
sorting/quantile pooling to be sample-efficient and stable. This is exactly the regime where the
literature adds **sorted / order-statistic pooling, quantile/ECDF layers, or set functions with
higher-moment pooling** (e.g. learnable-quantile or "featurewise sort" poolings) precisely because
mean/attention pooling is a poor *inductive* match for rank-based targets.

For Lacuna specifically:

- **Permutation-invariance over rows: in-class and correct.** Rows are exchangeable; a set encoder is
  the right symmetry. Keep it.
- **Order-statistic / tail functionals over rows: at the hard edge of the class, and mis-wired here.**
  The function is in the closure of the family but not where the current pooling's bias sits, and (§2)
  it is computed on the wrong axis behind a mean.
- **Two-sample / distribution-to-reference comparison (idiom a): outside the current architecture
  entirely.** Recovering δ for smooth self-censoring needs the observed target marginal compared to its
  **predictor-conditional MAR reference** — a two-distribution comparison. The encoder has no such
  module; the MoE/reconstruction machinery that could have supplied a "predict-then-compare" reference
  (the reconstruction heads predicted held-out values — a natural MAR reference) was **dropped in P2**.
  This is the part of the function class that is genuinely *absent*, not merely *biased against*.

So: the **set-encoder family is the right symmetry**, but the **specific pooling and staging are the
wrong instance** of it for δ, and the **conditional-reference comparison is missing**. The answer is not
"abandon set-transformers"; it is "add the order-statistic / ECDF pooling and a conditional-reference
branch that the δ functionals require, on the right (within-column across-row) axis."

---

## 5. Concrete architecture proposals (ranked; rationale, cost, validity)

All proposals must (a) stay on the semi-synthetic → held-out ladder (NORTH-STAR §8.1), (b) be scored
out-of-family (§5), and (c) respect identification: **a better architecture only computes the
*observable footprint* statistics better; it never beats the oracle** (§2 theorem; §6 caveat). None of
these is authorized for implementation here — they are for PI selection.

### Proposal A (best bet) — explicit ECDF / order-statistic pooling on the within-column across-row axis
Add a pooling branch that, for the **supplied target column** (and optionally each observed predictor),
forms the **across-row order statistics of the observed values directly**: a fixed or learnable set of
quantiles / tail-mass / inter-quantile gaps (a differentiable ECDF layer), computed on the clean column
marginal **before** any cross-column blending, then fed to the head alongside `evidence`.
- *Rationale.* This is the §3 ECDF front-end, but wired at the correct axis and kept inside the learned
  model. It supplies the exact functional family §1 requires and that §3's probe shows trivially
  captures the signal. It also subsumes/repairs the `consequence_features` experiment, whose weakness was
  not the features but that they were appended at the head while the encoder still mean-collapsed
  everything upstream and the across-row pooling stayed a mean.
- *Cost.* Low–moderate. Isolated to `lacuna/survey/` (a new pooling module + head wiring); no tokenizer
  change; reuses `consequence_features` math as the deterministic core. Fully ladder-evaluable.
- *Caveat.* See §6 — the LR-on-order-stats OOF ceiling is itself ~0.68 binary; this fixes the *encoder's*
  failure to reach that ceiling, it does not raise the ceiling.

### Proposal B — replace the conditioning head's mean with order-statistic pooling
Minimal surgical fix: change `_pool_target` (`conditioned_head.py:67`) from masked **mean** over rows to
**multi-statistic** pooling (mean + a few quantiles + max) of the target column's per-row reps.
- *Rationale.* Directly removes the §2C "averages the tail away" defect with the smallest possible
  change; §3 shows attention/sort pooling reaches p90/max R²≈0.97 where mean hits 0.70.
- *Cost.* Very low (one module). *Caveat.* Pools *learned reps* (already cross-column-contaminated by
  RowPooling), so weaker than A which pools the raw observed values; best treated as a cheap ablation to
  run alongside A, not instead of it.

### Proposal C — conditional-reference (two-sample) branch for smooth self-censoring
Add a "predict-then-compare" module: estimate the predictor-conditional reference distribution of the
target (a lightweight reconstruction, echoing the dropped v1.0 reconstruction heads) and feed the
**observed-vs-reference tail deficit** to the head.
- *Rationale.* This is the only proposal that addresses idiom (a)'s genuine *missing* function (§4) and
  the proxy-absorption confound (P2.2b) by construction.
- *Cost.* Moderate–high (a conditional density/quantile estimator + comparison). *Caveat.* Idiom (a) is
  the demonstrated flat-likelihood idiom (§8.2); even a correct two-sample branch may find the deficit
  near-zero at matched rate — which would then be *correct* high-entropy behavior, not a failure. Build
  only after A/B show the architecture can read the *detectable* (LOD) idiom.

### Proposal D — keep the encoder for cross-column structure; add a parallel distributional branch
Architecturally: a **two-branch** model = (encoder for cross-column mask/structure) ⊕ (distributional
branch = A/C for within-column shape), fused at the head. Restores the v1.0 spirit (encoder + explicit
complementary features) but with the *value-shape* features δ needs instead of the *mask-structure*
features classification needed.
- *Rationale.* Honest division of labour matching §2's finding that the encoder is *good at one thing*.
- *Cost.* Moderate. *Caveat.* Same §6 ceiling.

**Ranking:** A (highest value/cost, addresses the proven LOD case, ladder-clean) > B (cheap ablation,
run with A) > D (clean architecture once A works) > C (needed for idiom a, but gated behind the
detectable idiom and the §8.2 flat-likelihood reality).

---

## 6. Honest caveat — architecture is necessary, not sufficient

A better architecture cannot rescue δ-estimation by itself, for reasons the closed P2.2c arc already
established (DECISION-MEMO §6, the three levels):

- **The oracle is the ceiling.** Even a perfect machine cannot beat the Bayes-optimal LLR test on the
  observed data (NORTH-STAR §2; `lod_oracle.py`). For own-value self-censoring at matched rate the
  oracle itself is near-flat — **no architecture recovers a signal that is not in the observed-data
  law.** Proposals A–D only help where the oracle says signal *exists* (LOD).
- **The out-of-family transfer ceiling is independently low.** The hand order-statistic features — which
  *are* the §3 ECDF front-end, hard-coded — transfer only weakly: LR binary δ=0-vs-δ2.5 OOF AUC
  **~0.68–0.73**, and 3-bin OOF ≈ base-rate (DECISION-MEMO §4–§5). An architecture that perfectly
  computes those features inherits **that** ceiling. So Proposal A's realistic target is "**reach the
  ~0.68 LR ceiling out-of-family that the encoder currently misses**," not "solve δ." Whether 0.68 is
  product-useful is a separate, PI-level call (likely: useful as a *coarse presence-of-truncation*
  detector with calibrated abstention, not as a calibrated δ-magnitude estimator).
- **Mapping to the three detectability levels.** Architecture work targets the **learned-channel** level
  only — moving it from "base-rate, ≤ LR" up toward "reaches the feature-level LR ceiling." It cannot
  move the **oracle** level (theorem) or the **feature** level (that ceiling is a property of the
  observable footprint and the OOF transfer of these idioms, not of the model). The detectability
  *spectrum* (LOD ≫ own-value) is real at the oracle, faint at the feature level; the most a new
  architecture can do is make the learned channel finally *reflect* the faint feature-level spectrum
  instead of flooring below it.

Therefore: a distributional architecture is the right way to test **"can the learned channel reach the
feature-level ceiling for a detectable idiom?"** — an open, North-Star-faithful question (§3
generalization axis, "estimable & improvable"). It is **not** a path to beating non-identifiability, and
must not be sold as one.

---

## 7. Recommendation (PROPOSAL for PI approval — not an implementation)

**Verdict: the current architecture is PARTIALLY valid for δ-estimation.**

- **What it is valid for / good at:** permutation-invariance over rows, and **cross-column structure of
  the missingness mask** — the inductive bias the MCAR/MAR/MNAR classifier was built around, and the
  reason the encoder is the right *family* (a set encoder) even if the wrong *instance*.
- **What it specifically cannot (or is strongly biased not to) represent for δ:** the **within-column,
  across-row order-statistic / tail-shape of the observed target marginal**, because (i) two-stage
  features→row→dataset pooling **mean-collapses the per-column marginal before it can be summarized**
  (§2A), (ii) the default aggregators are **first-moment** averages (§2B), and (iii) the P2 conditioning
  head recovers the target column only to take a **mean over rows** (§2C), averaging away the tail δ
  lives in. This is not strict non-representability (the pooling primitive fits p90/max in isolation,
  §3) — it is a decisive **inductive-bias and wiring** mismatch that makes the needed functional an
  uphill, non-transferring special case. The conditional-reference (two-sample) comparison idiom (a)
  needs is **absent** outright, having been dropped with the reconstruction heads in P2.

- **Single best-bet architectural direction:** **Proposal A — an explicit ECDF / order-statistic pooling
  branch on the within-column across-row axis** (the supplied target column's observed values), fused
  with the existing `evidence`, with **Proposal B** (replace the head's mean-over-rows with
  multi-statistic pooling) as the cheap companion ablation. This injects the exact functional family §1
  requires, repairs the precise defect §2 identifies, is isolated to `lacuna/survey/`, changes no
  tokenizer and no charter, and is fully evaluable on the semi-synthetic held-out ladder.

- **The test it should be approved to answer (not "solve δ"):** *On the **detectable** LOD/top-coding
  idiom, does a distributional architecture move the **learned channel** off base-rate up to the
  feature-level LR OOF ceiling (~0.68 binary), while own-value self-censoring correctly stays flat?* A
  yes demonstrates the detectability spectrum **in the learned channel** (the one level P2.2c never
  reached) and validates the architecture thesis. A no — learned channel still floors even with the
  order statistics wired correctly on the right axis — would be strong evidence that the bottleneck is
  the **low OOF transfer ceiling itself**, not the architecture, and would redirect effort to
  domain-randomization / the detectability-map reframe (DECISION-MEMO §9) rather than more architecture.

- **What this proposal explicitly does not claim:** it does not beat non-identifiability (§6, the oracle
  is the ceiling); it does not raise the feature-level transfer ceiling; and it is not a license to
  resume training before the PI selects a direction. Per the standing instruction and the consolidation
  memo, **no experiment runs until this proposal is approved.**

---

### Appendix — evidence trail
- Code: `lacuna/models/encoder.py` (TokenEmbedding, row-wise TransformerLayer, RowPooling/AttentionPooling,
  DatasetPooling); `lacuna/models/assembly.py:284–465` (v1.0 `LacunaModel`, MoE, reconstruction);
  `lacuna/data/missingness_features.py` (the v1.0 mask-structure hand-features the MoE used);
  `lacuna/survey/{delta_head,conditioned_head,feature_only_head,consequence_features,lod_oracle}.py`.
- Findings: `DECISION-MEMO-P2.2c-consolidation.md` (§6 three levels), `feasibility-p2p2c-lod-oracle-findings.md`
  (oracle BE→0), `-lod-ladder-findings.md` (encoder floors; 95th-pct moves with δ),
  `-consequence-features-findings.md` (in-dist AUC 0.94, OOF gap), `-transfer-gate-findings.md` +
  `-coarse-ab-findings.md` (corrected OOF AUC ~0.68; 3-bin ≈ base-rate), `PHASE-SUMMARY-P2.2-P2.2b.md`,
  `NORTH-STAR.md` §2/§3/§6/§8.
- Illustrative probe: `scripts/illustrative_pooling_probe.py` (not a Lacuna run; §3 table).
