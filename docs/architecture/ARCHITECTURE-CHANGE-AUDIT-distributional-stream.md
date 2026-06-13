# Architecture Change Audit — did we re-architect Lacuna, or patch it?

*Audit only. No code, no runs. Status: **for PI review.** Tone: surgical honesty — this document does
not defend the distributional-stream implementation. Governed by `NORTH-STAR.md` and
`ARCHITECTURE-FITNESS-delta-estimation.md`. Written while the Gate-2 GPU A/B is still running; its
result is **not** used here and must **not** be read as evidence about "a distributional architecture"
until this audit is accepted (see §9).*

## 0. One-line verdict

**It was a head/branch patch, not a re-architecture.** The BERT-style backbone, the tokenization, and
the row→column information flow are **unchanged**. We replaced one across-row pooling of the target
column (mean → learned quantiles) and concatenated a 28-dim summary near the head. A null A/B therefore
**cannot** falsify "a distributional architecture fails for δ" — it only tests one head-adjacent readout
bolted to the *same* backbone the fitness report flagged.

---

## 1. What exactly changed?

**Files changed (this arc):**
- **NEW** `lacuna/survey/distributional_stream.py` — `masked_quantile_pool` + `RepECDFPooling`
  (one `Linear(hidden→4)` + deterministic quantile/max pooling over rows).
- **MODIFIED** `lacuna/survey/conditioned_head.py` — added `rep_ecdf_pooling`/`n_shape_probes`; split
  `_gather_target` out of `_pool_target`; `_pool_target` now branches mean-vs-ECDF; factory args.
- **MODIFIED** `lacuna/survey/train.py` — `TrainConfig.rep_ecdf_pooling`/`n_shape_probes` (+guard),
  model construction wiring, three manifest fields.
- **NEW** tests + runner/audit scripts (not architecture).

**Classes/functions changed:** `TargetConditionedDeltaModel.__init__`, `._pool_target`, new
`._gather_target`; `create_target_conditioned_model`; `TrainConfig`; the model-build block of
`train_delta_prior`. **Object added:** `RepECDFPooling` (≈ 530 params: a `Linear(hidden,4)`).

**Old components UNCHANGED (byte-identical):**
- `lacuna/models/encoder.py` in full — `TokenEmbedding`, `TransformerLayer`, `RowPooling`,
  `AttentionPooling`, `DatasetPooling`, `LacunaEncoder`. **The backbone was not touched.**
- `lacuna/data/tokenization.py` — `TOKEN_DIM=4`, `[value, is_observed, mask_type, feature_id]`.
- The two-stage pooling **order** (features→row→dataset). `DeltaBinHead`, RPS loss, `delta_bins`,
  `leakage`, `manifest`, the fixed `consequence_features` (17-dim value-ECDF).

**Old components still in the training path:** the **entire encoder** (token embed → 4× row-wise
TransformerLayer → RowPooling → DatasetPooling → `evidence`), whose `evidence` is concatenated to the
head; and the encoder's `token_representations` (post-transformer, pre-pool) which the head gathers the
target column from. The fixed 17-dim value-ECDF.

**Old components bypassed — but bypassed since the *start* of P2, NOT by this change:** MoE,
reconstruction heads, `MissingnessFeatureExtractor`, 3-class posterior, `BayesOptimalDecision`. The
distributional stream did not remove these; they were already absent from `TargetConditionedDeltaModel`.

> Net change of *this* arc: the across-row aggregation of the **target column's encoder reps** went
> from a masked **mean** (`conditioned_head.py` old `_pool_target`) to **6 quantiles + max** of a
> learned 4-d projection, concatenated near the head. Everything upstream is the old encoder.

---

## 2. True re-architecture, or head/branch patch?

| dimension | changed? | evidence |
|---|---|---|
| Backbone representation of the data | **No** | `encoder.py` unchanged |
| Core tokenization | **No** | `TOKEN_DIM=4`, same 4 channels |
| Row/column information flow | **No** | row-wise attention over column tokens, then pool — unchanged |
| Order in which within-column distributions are formed | **Barely** | the ECDF pools the target's *per-row encoder reps* across rows — but those reps are produced by the **same** row-wise attention that already blended each row's columns; the raw column marginal is formed only **outside** the net (`consequence_features` in `collate`) |
| Extra features/pooled summaries near the head | **Yes — this is the change** | head input = `[evidence(64); target order-stats(28); value-ECDF(17)]` |

**Verdict: head/branch patch.** We changed the target *readout* and appended summaries. We did not
change how the backbone represents data, how rows/columns flow, or where the within-column marginal is
formed inside the network.

---

## 3. What scientific object does the new architecture represent?

- **Empirical distributions — partially, peripherally.** A raw-value ECDF exists only as the *fixed*
  `consequence_features` computed in `collate` (outside the network); the learned `RepECDFPooling`
  computes order statistics of a learned projection's values across rows. Neither is a first-class
  distribution object the architecture is built around — both are **appended vectors**.
- **ECDF / quantiles / order statistics — yes, but as head-side summaries**, not as the backbone's
  representation.
- **Mask topology — represented only by the *old* encoder** (via `is_observed`/`mask_type` token
  channels + cross-column attention), unchanged. The stream adds nothing here.
- **Observed-vs-reference deviation — NO.** The conditional/MAR-reference branch (1B) was excluded by
  decision. There is no reference-comparison object anywhere.
- **Honest summary:** the model is still **global evidence + target pooling + appended features**:
  `[evidence ; target-across-row-order-stats ; fixed value-ECDF] → MLP → δ bins`.

---

## 4. Is it still BERT-like?

**Yes.** Backbone = token embeddings + pre-norm multi-head self-attention + hierarchical pooling; the
"sentence" is a row, the "tokens" are its feature cells, attention is **over columns within a row**
(`TransformerLayer.forward` reshapes to `[B·max_rows, max_cols, hidden]`). It is BERT-over-columns-per-
row plus two-stage pooling.

- **Why a BERT-like backbone is appropriate — for the ORIGINAL task, not this one.** Cross-column
  attention is the natural detector of "missingness in column j depends on observed values in column k"
  = the MAR signature (the encoder docstring says exactly this). For **δ-prior / consequence inference**
  the relevant object is the **across-row distribution of a single column** and its deviation from a
  reference — which row-wise cross-column attention does **not** natively form. So the backbone's
  inductive bias is **mismatched** to the new target, as the fitness report argued.
- **What we actually did, in analogy:** kept the MLP (BERT-over-columns) and bolted a histogram readout
  onto its activations — we did **not** build the CNN (a distribution-native backbone).

---

## 5. What did not change but maybe should have?

- **Tokenization** — unchanged. A δ architecture may want column values exposed as a *sequence/sorted
  set per column*, not per-cell 4-tuples.
- **Row/column axis handling** — unchanged (row-as-primary). A δ architecture likely wants
  **column-as-primary** distribution objects.
- **Global evidence pooling** — unchanged (attention = first-moment mean over rows).
- **Target conditioning** — only the readout changed (mean→ECDF); still head-side gather, not a
  backbone that conditions on the target.
- **Loss / model scale / regime / dataset diversity** — loss unchanged (RPS, fine); scale/regime/seeds
  changed only for the GPU *run*, not architecturally.
- **The load-bearing unexamined assumption: that the old encoder should be reused at all.** The patch
  *presupposes* the BERT backbone is the right substrate and only the readout needs fixing. **That
  assumption has not been earned** (see §8). This is the crux of the PI's concern, and it is correct.

---

## 6. What would a true from-scratch Lacuna-Survey architecture look like? (sketch — do not build)

Design **from the inference target backward.**

- **Input:** survey table `X[n,d]` + observed mask `R[n,d]` (+ column metadata/semantics, eventually).
- **Explicit intermediate objects (the point):**
  1. **Mask topology** — per-column miss rates, co-missingness across columns (revive the *concept* of
     `MissingnessFeatureExtractor`, which v1.0 already had as an explicit object).
  2. **Per-column empirical distributions** — for each column, a permutation-invariant-over-rows
     **distribution embedding** of its observed values (ECDF/histogram/neural-statistician per column).
  3. **Conditional / reference distributions** — for a target column, the distribution expected under
     **MAR given observed predictors** (a predictor-conditional reference).
  4. **Missingness consequence** — the **observed-minus-reference deviation** per column. *This is the
     object δ actually lives in*, and nothing in the current architecture represents it.
- **Backbone (two axes):** (a) a **within-column distribution encoder** (sorting/quantile/ECDF pooling
  or neural-statistician) → per-column distribution embedding; (b) an **across-column set encoder** (set
  transformer over columns) letting the target attend to predictor embeddings + mask topology → forms
  the reference and the deviation.
- **Output:** calibrated δ-prior (ordered bins or continuous + uncertainty) **with abstention/coverage**.
- **Evaluation:** held-out semi-synthetic survey tasks (the ladder), leakage-gated, oracle-anchored.

**Architectural inversion vs current:** *column-as-primary distribution objects + explicit
observed-vs-reference deviation*, instead of *row-as-sequence + cross-column attention then mean-pool*.
The analogy is **not BERT** — it is closer to a **Neural Statistician / DeepSets-over-distributions with
a two-sample-deviation head**, with a set-transformer-over-columns for the conditional reference.

---

## 7. Viable architecture options

| | Represents | Cannot represent | Cost | Sci. risk | Mechanism-general? | Lacuna-LOD risk |
|---|---|---|---|---|---|---|
| **A. Minimal patch to current encoder** (e.g. swap mean→max/quantile pooling) | same as now + a bit more tail | per-column raw distribution natively; reference deviation | trivial | low effort / low ceiling | yes | low |
| **B. Current encoder + distributional stream (WHAT WE DID)** | global evidence + target order-stats (learned) + fixed value-ECDF | conditional-reference deviation (excluded); native per-column dist (only target, only *post-blend*) | low (done) | **confounded** — a patch on a possibly-mismatched backbone; a null can't separate backbone vs readout | yes | low |
| **C. New backbone around per-column ECDFs/histograms** (column-primary: dist-encoder per column + set transformer over columns + reference-deviation module) | per-column distributions, conditional reference, observed-vs-reference deviation, mask topology as its own stream | the identification limit (oracle ceiling) — no architecture beats it | **high** | medium-high (more parts) but **directly targets the inference object** | yes — distribution-native, not idiom-specific | **low** (represents distributions/deviations, not truncation edges) |
| **D. Full rewrite / new Lacuna-Survey backbone** | everything in C + reconsiders tokenization, column metadata, mask-topology graph, abstention | identification limit | **very high** | high (scope/time), highest payoff; the only option that *earns* the re-architecture claim | yes | low |

Note: **B is the current state and it is a patch** (§2). A/B on B alone cannot decide reuse-vs-rewrite.

---

## 8. Evidence that would justify reuse vs rewrite (explicit criteria)

The missing diagnostic is a **representation probe** on the *unchanged backbone* (not yet run):
compare (a) **LR on raw-value ECDF features** (the baseline signal), (b) **a linear probe on the
encoder's `token_representations`/`evidence`** for the held-out δ, (c) **the full trained model**.

- **Keep the current encoder** when **(b) ≈ (a)**: the backbone reps already linearly encode the
  order-stat signal ⇒ the substrate is fine; fix the readout.
- **Replace the head only** when **(b) high but (c) low**: signal is in the reps, the head fails to
  exploit it.
- **Add streams** when signal is in raw values but **(b) ≪ (a)** *and* a stream is shown live+used
  (Gate-1) *and* moves the learned channel (Gate-2). (This is the test we set up — but see §9.)
- **Declare the backbone mismatched / start from scratch (C or D)** when **(b) ≪ (a)** persists *and*
  even a live+used stream at full scale across seeds does not reach the baseline: the backbone is
  **destroying the signal before the head**, and a better readout on a lossy substrate cannot recover it.

**Crucial gap:** the current arc **did not run probe (b).** Without it we cannot attribute any A/B
outcome to backbone-vs-head. So no reuse-vs-rewrite decision is currently *earned*, in either direction.

---

## 9. Did the latest implementation test the architecture hypothesis? (honest)

**No — not the deep one.** It tested a narrow sub-hypothesis: *"does replacing the target mean-pool with
a learned order-statistic pool (and keeping the fixed value-ECDF) move the learned channel?"* The
backbone the fitness report fingered — row-wise cross-column attention + two-stage mean-collapse pooling
— was **unchanged**. Therefore:

> A null Gate-2 result does **not** show "a distributional architecture fails for δ." It shows "this
> head-adjacent ECDF branch, on the unchanged BERT backbone, does not move it." It was a **head/branch
> patch**, and it must not be reported as a re-architecture.

Gate 1 confirmed the patch is *live and used* (not dead plumbing) — that is real and worth keeping — but
"the branch is used" is not "the architecture hypothesis was tested."

---

## 10. Recommended next step (recommendation, not a run, not code)

1. **Do not scale the patch as if sufficient,** and do not let the Gate-2 A/B stand as evidence about
   "a distributional architecture." Record the A/B as a fact about Option B only.
2. **Earn the reuse-vs-rewrite decision with the §8 representation probe** (analysis, not a new
   architecture): does the *unchanged* encoder's representation linearly contain the order-stat δ signal
   that raw-value ECDFs carry? This single diagnostic separates "backbone destroys the signal" (→ C/D)
   from "head/readout fails to use it" (→ A/B). It is the cheapest decisive evidence and it is missing.
3. **In parallel, write a from-scratch design spec for Option C** (column-primary per-column
   distribution encoder + set-transformer-over-columns + explicit observed-vs-reference deviation +
   abstention), designed from the inference target backward (§6) — as a *spec for approval*, on a clean
   architecture branch, **not** an implementation.
4. **Decide reuse-vs-rewrite from the probe + the spec**, not from the patch's A/B.

**Primary recommendation:** treat the current work as a *patch* (Option B), not a re-architecture; **do
not interpret the A/B as architecture evidence**; the earned next step is (a) the backbone
representation probe and (b) an Option-C from-scratch design spec — both analysis/spec, decided before
any new build. The re-architecture decision the PI is asking for has **not yet been earned**, and this
audit's job is to say so plainly.
