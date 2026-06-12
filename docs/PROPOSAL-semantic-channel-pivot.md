# PROPOSAL — The Semantic-Channel Pivot (pre-registration-shaped; spec only, no code)

*Companion to `docs/T-review-findings.md`. Status: PROPOSAL for PI/advisor decision. Nothing
here is built, downloaded, or run. If adopted, the locked sections below become a
pre-registration in the same discipline as `7714f0c` (frozen bars, frozen nulls,
implementation-before-results, honest failure branches).*

---

## 0. Why this pivot, in one paragraph

The T-review (fail/fail/fail, locked verdict) established that on the numeric-matrix channel,
wherever signal exists it is feature-capturable, and wherever features fail, networks fail
identically — the binding constraint is identification, not representation. Yet statisticians
make confident MNAR judgments daily ("income ⇒ refusal bias ⇒ MNAR"). The review's measurements
show that judgment is **~all prior, ~no likelihood**: the information lives in the *semantics of
the item* — the channel every blindfolded-matrix experiment deliberately amputated. The
hypothesis under review therefore changes from "a network can detect MNAR from the matrix better
than statistics" (measured: false) to **"the missingness-relevant information is encoded in
natural language and human disclosure behavior, and a trained model can learn its structure"**
(unmeasured, well-posed, real labels exist).

## 1. The hard requirement (binding, from the PI)

The dissertation requires a **trained neural network** that is **non-decorative** — trained by
us (gradient descent on weights, our objective, our labels; pretrained initialization is
acceptable and standard), not merely an off-the-shelf model we call. If no such component clears
its bar, the project does not serve as a dissertation and is sunsetted (archived, not deleted —
the pre-registered negatives and the labeled corpus are what make the sunset defensible).

**Operationalization:** the non-decorative test is the **Arm 3 − Arm 2 margin** (§3). Beating
the keyword null with a frozen off-the-shelf embedding (Arm 2 > Arm 1) is commodity capability
and does NOT satisfy the requirement. Only the *trained* representation beating the *frozen*
one, out-of-instrument, demonstrates that our training is load-bearing.

**Pre-build checkpoint (do first, costs one conversation):** confirm with the advisor that
"fine-tuned pretrained transformer with a novel behavioral objective + demonstrated necessity"
meets the degree's ML bar. If the bar is "from-scratch architecture," this design changes
materially and that must be known before any corpus work.

## 2. Hypothesis (three clauses, each separately falsifiable)

**H-semantic:** the item-nonresponse behavior of a survey item (refusal propensity; refusal-vs-
don't-know composition; refuser covariate signature) is predictable from its natural-language
description (label, question text, response options, codebook context), out-of-instrument.

**H-structure:** the predictive structure includes *missingness-risk classes that cut across
topical semantics* — e.g., income and illicit-drug-use are neighbors in risk-space (sensitive
self-report) while income and interest-rate are not, despite topical proximity. Off-the-shelf
embeddings encode topic; nothing in their pretraining encodes "these two attract refusals from
similar people." This is the representation-learning claim.

**H-trained (the load-bearing clause):** capturing H-structure requires task-supervised
representation learning — a behavior-trained encoder beats both the curated-keyword null and the
frozen-embedding arm on held-out instruments by a locked margin.

## 3. Design — the three-arm showdown (nulls frozen before any trained result)

| arm | what it is | what beating it would prove |
|---|---|---|
| **Arm 1 — curated null** | hand-built topic taxonomy + keyword/regex/bag-of-words features + LR/GBM, assembled maximally (the committee's "couldn't I just…" objection, locked as in T1's anti-handicap guard) | paraphrase robustness matters (commodity) |
| **Arm 2 — frozen embeddings** | pinned open-weights text encoder (reproducibility: a frozen artifact, not an API), no training; shallow head on embeddings | semantics beats keywords (still commodity) |
| **Arm 3 — behavior-trained encoder** | the same (or smaller) encoder **trained by us**: contrastive/metric or multi-task objective against observed nonresponse behavior; LoRA or full fine-tune; checkpoints, curves, ablations | **the dissertation claim** — trained representation is load-bearing |

**Primary locked bar (to be numerically fixed after the corpus-feasibility scope, before any
Arm-3 training):** Arm 3 (mean−SE over 5 seeds) beats Arm 2 on the primary task,
leave-instrument-out, by a margin to be locked in the final prereg; Arm 3 must also beat Arm 1
(sanity, not the headline). Recorded prior that Arm 3 clears it: **~50%** — genuinely uncertain,
not rigged. Discipline inherited wholesale: frozen nulls, 5 seeds mean−SE, degenerate seeds
reported, bit-identical eval items across arms, no tuning after results, all slices reported.

**Failure branches (recorded now):**
- Arm 3 ≈ Arm 2: training adds nothing ⇒ the hard requirement is unmet ⇒ **sunset branch** (per
  §1), with the consolation that "nonresponse behavior is frozen-semantics-predictable" is a
  real survey-methodology finding on real labels.
- Arm 2 ≈ Arm 1: topic keywords carry the signal ⇒ doubly negative, same branch, same realness.
- Either way the result is about the world (labels not authored by us) — unlike the
  generator-circular June arc, parity here is still science.

## 4. Supervision — real labels, three types, all documented (not authored by us)

1. **Refusal vs don't-know codes** (the workhorse): item-level sentinel codes distinguishing
   *refused* from *don't know* from *no answer*. Sources on hand or public: **GSS** (decades,
   thousands of items, codes preserved — likely the richest single source), **ESS** (~500 items
   × ~30 countries; the 77/88/99 codes we previously treated as cleaning nuisances ARE the
   labels), NHANES questionnaire modules (7777/9999), SCF, PISA. Refusal-vs-DK composition is a
   finer target than refusal rate (sensitivity vs cognitive burden) and a model separating them
   is learning something the folk taxonomy doesn't state.
2. **Documented routing** (negative/structural class): skip-logic gates printed in the
   instrument — items missing by design that the model must NOT call sensitive.
3. **δ-tier anchors (thin; calibration only):** bracket follow-ups (SCF/CPS "is it above or
   below X?" — direct refuser-vs-respondent value-gap evidence) and record-linkage validation
   studies from the methodology literature. Dozens of anchors, not thousands ⇒ claims tier down
   accordingly (§6).

## 5. Confound controls (pre-registered; the lessons of the corpus arc apply directly)

- **Leave-instrument-out splits, block-aware** (a survey = one block; waves/countries of the
  same survey = the same block — the slice-inflation trap, again).
- **Near-duplicate/paraphrase dedup across surveys** before splitting (or the splits silently
  leak paraphrases and Arm 2/3 get free wins).
- **Mode and house effects** (face-to-face vs self-administered changes refusal for the same
  item): block on them or condition on them; decide and lock in the final prereg.
- **Position/length covariates** (item order, questionnaire fatigue) included in ALL arms so
  they can't masquerade as semantics.

## 6. Output contract (what Lacuna emits; no naked probabilities)

Per column, a **provenance card** — never "90% MNAR":

> *Semantic reference class:* sensitive self-reported financial quantity (induced cluster #k)
> — class-assignment confidence: **measured** OOF accuracy.
> *Expected nonresponse behavior:* elevated refusal share, refusers skew [signature] —
> **measured** calibration on held-out instruments.
> *δ-prior range:* [a, b] — **cited** ledger anchors (bracket/linkage studies; count visible).
> *Matrix-channel evidence:* MCAR-gate / skip / top-coding findings (the channels the T-review
> validated) — including **channel conflicts** ("text says routine; mask shows a gated block")
> as first-class outputs.
> *Suggested analyst action:* sensitivity analysis with the stated δ-range, or none, with basis.

Reference classes are **induced** (clustered in the learned space, validated for behavioral
coherence OOF), not hand-coded — hand-coding "income/wealth/illegal/health" would assume the
answer to H-structure and recreate the expert-system mistake. The folk taxonomy lives in Arm 1
as the null, not in the architecture. Every number on the card has a provenance type (measured /
cited / induced+validated). The comparison class for "defensible" is current practice — the
uncited, uncalibrated lab-coat prior. Molenberghs is never beaten; he is budgeted for, with
sources, as width in the δ-range.

## 7. Dissertation framing (PhD in Machine Learning and Data Science)

- **Part I (data science):** the measurement framework (matched-rate semi-synthetic truth, the
  oracle ladder, frozen feature nulls, pre-registered showdowns) and its finding — the
  expert MNAR judgment decomposes into ~all prior / ~no likelihood; the matrix channel is
  statistics-sufficient (the T-review). This part *motivates and disciplines* Part II.
- **Part II (machine learning):** the behavior-trained encoder — representation learning over
  item semantics supervised by real disclosure behavior, with H-trained as the load-bearing
  claim and the Arm 3 − Arm 2 margin as its proof. The generators and the matrix channel are
  retained as the calibrated Stage-A spine (MCAR-gate 1.000; skip/top-coding detection), not
  discarded.
- The North Star is tightened, not abandoned: "infer missingness mechanisms from data" — where
  "data" now means the dataset as it actually exists (matrix + instrument + codebook), the form
  in which every human expert ever receives it.

## 8. Sequencing (proposed; nothing authorized until PI/advisor sign-off)

1. **Advisor checkpoint** (§1) — what counts as "trained." One conversation; gates everything.
2. **Corpus feasibility scope** (~days, read-only): can we extract item text + response options
   + sentinel codes at scale from GSS/ESS/NHANES/SCF/PISA codebooks? Deliverable: item counts,
   label coverage, dedup estimate, per-survey block map. The SCF-acquisition playbook applies.
3. **Final pre-registration** (locks Arm-1/Arm-2 nulls, primary metric, margins, splits) —
   committed before any Arm-3 training, reviewed by the PI.
4. **Arms 1–2** (nulls measured first), then **Arm 3** (5 seeds), then verdict — mechanically,
   from the locked bars, whatever it says.

*Spec only. The one question this proposal exists to answer, stated plainly: is there a trained
neural network that is demonstrably load-bearing in Lacuna? The T-review answered "not in the
matrix." This design either finds it in the semantics or licenses the sunset.*
