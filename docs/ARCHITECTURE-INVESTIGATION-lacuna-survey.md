# Architecture Investigation — what should Lacuna-Survey actually be?

*Architecture research, not software development. **No code, no experiments, no implementation plan.**
Status: **for PI review.** Triggered by the Stage-0 result (a column-primary distribution representation
recovers OOF LOD signal the row-primary BERT backbone loses). Governed by `NORTH-STAR.md` and the
evidence chain: `ARCHITECTURE-FITNESS-delta-estimation.md`, `ARCHITECTURE-CHANGE-AUDIT-distributional-
stream.md`, `probe-encoder-representation-findings.md`, `probe-stage0-column-primary-findings.md`.*

*This document answers six questions (A–F) before any line of code. It does not assume the current
backbone survives, and it does not pitch a rewrite; it lets the inference object drive the conclusion.*

---

## A. What is the actual inference object?

Not the model. The scientific object. It is **layered**, and conflating the layers is what produced the
abandoned arc.

- **Layer 1 — the missingness CONSEQUENCE (estimable from data).** The **deviation between the observed-
  data law and the law a matched-rate MAR mechanism would produce** — concretely, the distortion of each
  column's observed (conditional) distribution relative to its MAR reference. This is a property of the
  observed-data law: it is *identifiable in principle* (the MCAR-vs-structured axis, §2 solid ground).
  **This is the object the network must learn to compute.** It is a *two-distribution comparison*
  (observed vs reference), not a classification and not a dataset-summary.

- **Layer 2 — the IDENTIFICATION BRIDGE (consequence → δ), degenerate.** δ is a parameter of the
  mechanism, and the map consequence→(mechanism, δ) is **many-to-one** (Molenberghs §2): the same
  footprint is consistent with a family of (mechanism, δ). This layer is **not** a learnable function of
  the observed-data law; it is resolved only by the **manifold-restricted prior** (§3½) — the bet that
  nature reuses a small vocabulary of survey idioms, placing ~zero mass on the adversarial MAR-twins.
  The architecture does not *compute* this layer; it *carries* it as a regularizer/prior.

- **Layer 3 — the terminal OUTPUT.** A **calibrated prior over the sensitivity parameter δ**, plus an
  explicit **detectability / coverage state** (how informative the consequence is) and **abstention**
  off-manifold. Wide/high-entropy where the consequence is flat (correct, §4.2); sharp where diagnostic
  (e.g. LOD); "escalate to manual review" off-span (§4.3).

**The single sentence:** *the inference object is the observed-vs-reference distributional deviation (the
missingness consequence), together with the calibrated, abstaining prior over the sensitivity parameter
it induces under a survey-manifold mechanism prior.* The **latent detectability state** the PI named is
real and belongs at Layer 3 as a first-class output — the model must *know* when it is in a flat-
likelihood regime.

**Consequence for architecture:** the network's hard, load-bearing job is **Layer 1 — compute the
observed-vs-reference deviation.** Everything the abandoned arc did wrong followed from mistaking the
object for "classify the dataset" (Layer-2 confusion) or "pool the dataset into an evidence vector"
(which represents neither the per-column distribution nor the deviation).

---

## B. What information must any successful architecture preserve?

A complete inventory. Each item is tagged with whether the **current backbone preserves it** (from §8 /
Stage 0 / the fitness analysis).

**Within-column (the locus of δ):**
1. **Per-column observed empirical distribution** at full resolution — ECDF / quantiles / order
   statistics, *not* moments. — *Backbone: NO (averaged away; §8 R²≤0, Stage 0 0.55 vs 0.74).*
2. **Tail behavior** specifically — upper/lower truncation, depletion (value-localized censoring lives
   here). — *Backbone: NO.*
3. **Scale-invariant** within-column representation (standardize within observed). — *Partially (value
   is linearly projected; not distribution-level).*

**Cross-column / relational (the reference):**
4. **Predictor-conditional distribution / expectation of the target** — the MAR reference, and the
   antidote to the proxy-absorption confound. — *Backbone: weakly (cross-column attention exists, but
   collapsed by pooling before a conditional distribution is formed).*
5. **Cross-column dependency structure** (correlations) — for the reference and for MAR signatures. —
   *Backbone: YES (this is its design strength).*

**Mask / topology:**
6. **Mask topology** — per-column miss rates, co-missingness across columns, missingness-vs-observed
   dependence (the MCAR/MAR-identifiable anchor). — *Backbone: YES (token channels + attention; this is
   what it was built for).*
7. **Matched-rate invariance** — the representation must *not* key on overall missing rate (matched /
   uninformative; the leakage trap). — *Enforced by the generator + leakage gate, not the architecture.*

**Comparison (Layer 1 itself):**
8. **Observed-vs-reference DEVIATION as a first-class quantity** — not reconstructable from the marginal
   alone. — *Backbone: NO (no reference module anywhere; the v1.0 reconstruction heads that could have
   been a reference were dropped in P2).*

**Invariances:**
9. **Permutation-invariance over ROWS** (exchangeable observations). — *Backbone: YES.*
10. **Column identity preserved; predictor-SET exchangeability** — the target is distinguished, the
    predictor columns are an exchangeable set for the reference. — *Backbone: partial (position embeds).*

**Context / priors (given per human-parity §4.8; mostly downstream):**
11. Regime/scope (survey) — given; restricts the manifold.
12. Column semantics/metadata — a legitimate *prior* source (§8.3), gated by held-out calibration.
13. The estimand — downstream reporting layer (§3¾), not in the network.

**Output-side:**
14. **Detectability / uncertainty state** — computable signal of how informative the consequence is. —
    *Backbone: NO explicit representation.*

**The diagnosis in one line:** the backbone preserves the *mask/cross-column* items (5, 6, 9) — its
original design target — and **destroys the within-column distribution and the deviation** (1, 2, 8) —
which is exactly where δ lives. That is the mismatch, now empirically nailed.

---

## C. Plausible architecture families (clean-sheet; backbone not assumed to survive)

| family | naturally represents | cannot represent | strengths | weaknesses | mechanism-general? |
|---|---|---|---|---|---|
| **BERT / row-wise transformer (current)** | cross-column dependence within a row; mask topology | within-column across-row distribution / order stats (pooling averages them) | strong for the *classification* task; variable width | row-as-sequence + mean/attention pool destroys distributional shape; deviation not native | general for mask idioms; **mismatched for value-distribution idioms** |
| **DeepSets** ρ(Σφ(xᵢ)) | permutation-invariant set fns; with quantile pooling, per-column distributions | order stats under *plain sum* pooling; cross-column relational structure | simple, cheap, proven (Stage-0 φ is this); row-permutation-invariant | single-set; needs extension for relational reference | yes (within-column); not relational |
| **Neural Statistician** | a distribution → latent embedding (set→summary), hierarchical, uncertainty-aware | cross-column reference unless composed; two-sample comparison | principled distribution encoder; matches Layer-1; native uncertainty | heavier; training stability; still needs a relational layer | yes |
| **Set Transformer (ISAB/PMA)** | attention set-encoding; higher-order interactions; *over columns* → the relational/reference layer | order stats natively (attention pool ≈ weighted mean; §3 probe: possible but not the bias) | flexible relational modeling over the predictor set | same averaging bias for within-column extremes; params | yes (relational) |
| **Distribution / ECDF encoders** (quantile-fn, learnable histogram, characteristic-fn nets) | empirical distributions explicitly; tails | relational/conditional alone | directly native to Layer-1; transferable (Stage-0 φ ≈ raw-ECDF); scale-invariant | per-column; reference needs a separate module | yes |
| **Two-sample / distribution-comparison nets** (deep MMD, conditional-discrepancy) | the **deviation** between two distributions — *directly Layer-1's object* | the reference itself (needs a generator/conditional model) | native to "observed-vs-reference deviation" — the actual inference object | needs the reference as input; less standard; trickier to train | yes — the most direct match |
| **Graph / relational (columns=nodes, co-missingness/dependency=edges; GNN)** | cross-column dependency graph; mask topology; variable width | within-column distribution (must be injected as node features) | natural for topology + relational structure | within-column distribution still needs a distribution node-encoder; heavier | yes (relational/topology) |
| **Hybrid: per-column distribution encoder → set/graph over columns → deviation head** | the **entire** §B inventory | the identification limit (no architecture beats it) | matches the inference object layer-for-layer; each part a proven family | most components; integration cost; the *reference* module is the risky/unproven part | yes, strongly |

**Reading:** no single classic family covers the object, because the object is **two-level** —
*distribution-of-each-column*, then *relate-across-columns to form a reference and a deviation.* The
families compose: a **distribution encoder** (DeepSets-with-quantile-pooling / Neural Statistician /
ECDF-net) for Layer-1 within-column, a **set-transformer/graph over columns** for the relational
reference, and a **two-sample/deviation head** for the consequence itself. The current BERT backbone is
the one family whose *primary axis* (row-as-sequence, average-pool) is actively wrong for Layer-1.

---

## D. Minimum architecture that could represent the inference object (conceptual)

**Load-bearing (essential — remove any and the object cannot be represented):**
1. **Within-column distribution encoder φ** — order-statistic/ECDF-preserving, row-permutation-invariant,
   scale-invariant. *Stage-0 proves this is the substrate that carries the signal.* Without it the signal
   is destroyed before anything downstream.
2. **A reference + deviation mechanism** — some way to form the (predictor-conditional) MAR reference and
   the **observed-vs-reference deviation**. This is what makes the system *δ-inference* rather than
   *marginal description*. **Conceptually load-bearing for any value beyond the raw-ECDF baseline** — and
   (Stage-0 caveat) the *only* thing that could justify a neural architecture over a one-line raw-ECDF
   classifier. **Its empirical payoff is unproven and historically fragile** (predictor-referencing
   failed once: `transfer_features` 0.523).
3. **Calibrated δ-prior output + detectability/abstention head** — the governance product (Layer 3).

**Optional / enhancing (improve coverage; not required for a minimal representation):**
4. **Cross-column relational encoder** (set-transformer / graph over columns) — needed for a *multi-
   predictor* conditional reference; a minimal version can use a single best predictor.
5. **Mask-topology stream** — the MCAR/MAR-identifiable anchor and an abstention input; the detectable-
   idiom δ task can begin without it.
6. **Metadata / semantics prior** — deferred (§8.3), gated by held-out calibration.

**The crux, stated plainly:** the *minimum that justifies a rewrite* is **φ + deviation mechanism +
calibrated output.** φ **alone** equals the raw-ECDF baseline (Stage-0: 0.735 = 0.735) — so φ-alone is a
learned reimplementation of a one-liner, **not** worth a rewrite. **Component 2 (the deviation module) is
the load-bearing differentiator and simultaneously the unproven, highest-risk piece.** Whether Lacuna-
Survey should be a neural architecture at all hinges entirely on whether component 2 earns its keep.

---

## E. Salvage analysis (keep / modify / replace / remove, with justification)

| component | verdict | justification |
|---|---|---|
| **Tokenization** (`tokenization.py`, per-cell 4-tuple, row-major) | **Replace** (for the δ path) | row-major per-cell tokens serve row-wise attention; a column-primary architecture needs **column-major observed-value sets + column metadata**. Keep the *concept* (value, observed, mask_type), reorganize column-first. |
| **Encoder** (`encoder.py`, BERT backbone) | **Replace as the δ spine** (optionally **demote** to an auxiliary mask/cross-column stream) | §8 + Stage 0: it does not preserve the within-column signal. It is genuinely good at cross-column/mask structure, so it *may* survive as a secondary topology stream — but **not** as the representation spine. |
| **Pooling** (Row/Dataset, mean/attention) | **Replace** | averaging is the *specific* defect; needs order-statistic/quantile/ECDF pooling on the within-column axis. |
| **Fixed consequence features** (`consequence_features.py`, 17-d value-ECDF) | **Keep** (as baseline / φ sanity) | it *is* the raw-ECDF signal (0.735); useful as the baseline a learned φ must beat, and as a deterministic fallback. |
| **rep-ECDF stream** (`distributional_stream.py`, the patch) | **Remove** from the δ path; **salvage the primitive** | the patch was order-statistic pooling on the *wrong substrate* (encoder reps). The `masked_quantile_pool` primitive is reusable inside φ (Stage-0 already reuses it on raw values). |
| **transfer_features** | **Remove** | predictor-referencing failed the no-training gate (0.523). |
| **Generators** (`delta_generator`, `lod_generator`, answer sheets, matched-rate solve) | **Keep** | semi-synthetic supervision is the ground truth; architecture-agnostic. |
| **Evaluation** (held-out leave-datasets-out ladder, A/B, metrics) | **Keep** | the eval spine is sound and idiom/architecture-agnostic. |
| **Leakage gate** (`leakage.py`) | **Keep** | essential discipline; architecture-agnostic. |
| **Oracle** (`lod_oracle`, profiled MAR null) | **Keep** | the pre-train ceiling discipline (§5); architecture-agnostic. |
| **Calibration / loss** (RPS, temperature, ECE, coverage, δ-bins) | **Keep**, **Modify** to add detectability/abstention head | output-layer machinery is architecture-agnostic; Layer-3 needs an abstention/coverage addition. |
| **MoE / reconstruction / 3-class / decision** (`assembly.py`) | **Remove** from δ path (already bypassed); **keep the reconstruction *idea*** | the reconstruction heads' "predict-then-compare" is conceptually the reference module — keep the idea, not the code. |

**The shape of the conclusion:** **keep the entire scientific scaffold** (generators, ladder, leakage,
oracle, calibration, δ-bins) and **replace the representation spine** (tokenization-for-δ, encoder,
pooling). A "rewrite" here is a **representation rewrite on a retained, validated scaffold — not a from-
zero project rewrite.** That bounds the cost and the risk, and it is the honest scope of what Stage 0
justifies.

---

## F. What evidence would justify a full rewrite? (explicit criteria)

The decision splits into **two gates**, and the distinction is the whole point.

**Gate I — is the backbone fundamentally mismatched as the δ spine?**
*Criterion:* a representation *outside* the backbone family recovers OOF signal the backbone cannot, in
the *same* held-out regime, convergent across probe types and seeds.
*Status:* **MET.** §8 (old reps ~0.52, preservation R²≤0, random *and* trained) + Stage 0 (column-primary
φ 0.735 vs backbone 0.548, +0.19, seed-stable, same split). The backbone is mismatched **as the δ
representation spine.** → *Replace the spine.* (This does not condemn the scaffold, §E.)

**Gate II — does a neural distributional architecture earn its complexity over the trivial baseline?**
*Criterion:* the φ + **deviation** architecture beats the raw-ECDF baseline **and** the marginal-φ OOF
(while own-value stays materially lower), i.e. component 2 (§D) adds real signal.
*Status:* **NOT YET TESTED.** Stage 0 showed φ-alone *equals* raw-ECDF (no marginal-learning headroom);
the deviation module is unproven and historically fragile. **This gate decides neural-rewrite vs ship-
the-baseline.**

**Therefore the two things this exercise can now say — and it says the first, with a scoped boundary:**

> **(1) The current backbone IS fundamentally mismatched as the δ-inference representation spine, and a
> representation rewrite to a column-primary distribution-native architecture is justified (Gate I met).**
> The scaffold (generators, eval ladder, leakage, oracle, calibration) is **salvageable and should be
> kept**; the mismatch is the representation spine, not the science.

It does **not** yet say a *full neural Option-C* is justified, because **Gate II is untested**: φ-alone
is only a learned raw-ECDF, and the deviation module that would justify a neural architecture over a one-
line raw-ECDF-plus-sensitivity-layer governance tool has not been shown to add OOF signal.

**The cleanest decision criterion going forward (still analysis, no rewrite):**
- **If** a single bounded **conditional-deviation analysis** (does observed-vs-predictor-reference
  deviation add OOF signal over the marginal raw-ECDF, own-value staying lower?) is **positive** → Gate
  II met → a full neural Option-C rewrite is justified; proceed to a staged build spec.
- **If negative** → Gate II fails → the right product is the **minimal column-primary baseline (raw-ECDF
  / φ) + the calibrated sensitivity-reporting layer + abstention** — a *light* rewrite of the spine, not
  a heavy neural architecture — and further neural effort is unwarranted (DECISION-MEMO §9 regime view).

Either way, **the BERT backbone does not return as the δ spine.** The open question is *how heavy* the
replacement must be, and that is a Gate-II question answerable by one more analysis — **not** by writing
the architecture yet.

---

## Recommendation (for PI; no code)

1. **Record Gate I as met:** the row-primary BERT backbone is mismatched as the δ representation spine →
   replace it with a column-primary distribution-native spine; **keep the validated scaffold** (§E).
2. **Do not write the architecture yet.** The heavy-vs-light decision (full neural Option-C vs minimal
   φ/raw-ECDF + sensitivity layer) is **Gate II**, and it is decidable by **one bounded analysis** (the
   conditional-deviation probe), consistent with the standing "analysis before code" discipline.
3. **Approve, as the next architecture-phase step, that single analysis** (or, if the PI prefers to
   settle the object/spec first, approve this document and defer the Gate-II analysis). No implementation
   until Gate II is resolved and a staged build spec is approved.

The Stage-0 result was enough to pause implementation and earn the *substrate* decision. It is **not**
enough to earn the *full neural architecture* decision — and saying so is the discipline that the
abandoned arc lacked.
