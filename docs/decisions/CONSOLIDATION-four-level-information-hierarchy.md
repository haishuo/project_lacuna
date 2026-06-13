# Where Does the Information Disappear? — The Four-Level Hierarchy (consolidation)

> **STATUS (2026-06-10): amended by the adversarial review AND resolved by the measured empty cell.**
> Read with `ADVERSARIAL-REVIEW-four-level-hierarchy.md` (lattice-not-chain; truth and mechanism
> assumptions are non-nested substitutes) and `conditional-without-truth-cell-findings.md` (the empty
> cell, **measured: MIXED, unanimous**). **§0½ below is the result of record** and overrides the
> original §B headline and 2×2 wherever they conflict.

## 0½. THE MEASURED RESULT OF RECORD (2026-06-10; PI-accepted)

The truth-bottleneck / conditional-bottleneck fork is **replaced by a measured decomposition** (own-value,
binary δ=2.5, matched rate 0.3, n≈384, current survey role-B corpus, H/S/R feature family, LR LODO —
**eval-only; no runtime claim**):

| station | own-value OOF AUC | what it has |
|---|---|---|
| **marginal footprint floor (φ)** | **≈ 0.57** | observed marginal only |
| **conditional observed-view mid-rung** | **≈ 0.77** (0.74–0.80; ρ mean 0.506) | observed view + conditional structure; **no truth** |
| **truth-counterfactual apex** | **0.93–0.99** | the deleted values; no mechanism knowledge |
| **oracle apex** | mechanism-informed, **cell-dependent** (mean high, bimodal: 0.08–1.00 at δ=2.5) | the mechanism family; never the realized truth |

- **Own-value is: marginally weak, conditionally visible, and still partially identification-limited.**
- **The lab-coat fraction is a two-term deployment-gap decomposition:** the **representation share**
  (φ → conditional observed-view, ≈ 51% of the G1 increment) and the **identification share**
  (conditional observed-view → truth-counterfactual, ≈ 49%).
- **Scoped replacement for every "own-value is flat" statement:** *own-value is flat to marginal
  footprint features, NOT flat to conditional observed-view features, and highly visible under
  truth-counterfactual evaluation.*
- **`transfer_features` (0.523) is SUPERSEDED for top-coding by the matched protocol** (no-truth
  conditional reaches 0.81–0.83 on the same corpus/splits): it remains only as historical evidence that
  **cross-era feature results are not portable** across corpora/feature families.
- **The no-truth conditional station detects *selection* more strongly than it distinguishes
  *mechanisms*:** idiom separation collapsed without truth (0.54–0.68 vs 0.68–0.83 with truth) — the
  truth-referenced PIT shape was the mechanism reader.

> **DO NOT OVERCLAIM (binding):**
> - Do **not** say this proves runtime recoverability — the mid-rung is an eval-station measurement
>   under semi-synthetic conditions; no runtime pathway is validated.
> - Do **not** say the identification wall disappears — ≈ half the gap remains truth/assumptions-only,
>   and mechanism *identification* (idiom separation) collapsed without truth.
> - Do **not** say conditional features "solve" own-value — 0.77 ≪ 0.93–0.99, and the result is δ=2.5
>   / rate-0.3 / corpus-scoped.
> - Do **not** design G2 yet.
> - Do **not** amend architecture beyond this conceptual synthesis.

*Conceptual synthesis — **specification/analysis only. No experiments, no G2 design, no architecture
proposals, no runtime proxies, no implementation** (binding; PI 2026-06-08). Treats G1 as a scientific
result and asks where mechanism/δ information is lost between omniscience and deployment. Synthesizes:
P1 feasibility, P2.2c oracle + decision memo, Stage-0/Stage-1, D2 regime map, D3 transfer test, the
detectability review (E-JUSTIFY/E-FALSIFY), and G1 (`G1-imputation-channel-findings.md`). Governed by
`NORTH-STAR.md` and `MASTER-lacuna-survey-architecture.md`.*

---

## 0. Headline

G1's important content is not "the imputation channel works." It is:

> **The major information collapse occurs between Level 2 (truth, no mechanism) and Level 3 (observed
> marginal footprint) — not between Level 1 (oracle) and Level 2.** Mechanism knowledge is nearly
> redundant *given truth*; almost everything is lost when *truth* is removed — and a previously archived
> negative result (`transfer_features`) shows the loss is specifically **truth**, not conditional
> structure. The bottleneck decomposes into **two different walls** for the two idioms (§E), and this
> changes what the dissertation's core claim should be (§G).

## A. The four-level hierarchy, formalized

Each level is an **epistemic station**: what the observer knows beyond the observed data of one
semi-synthetic example (column + mask), at matched rate.

| level | observer knows | the estimator | its quantity | computable… |
|---|---|---|---|---|
| **1. Oracle** | true **mechanism family + δ** (profiled over nuisance) + observed data | Neyman–Pearson LLR vs best plausible MAR | profiled Bayes error → `I_oracle` | eval only (needs the generating cell) |
| **2. Imputation counterfactual** | **truth at the punched cells**; *no* mechanism knowledge | generic MAR imputer on observed rows; paired vs matched-rate MCAR | ΔB, ΔPIT, ΔB_top, ΔW1, Δcov80 | eval only (needs the answer sheet) |
| **3. Marginal footprint (φ)** | observed data only; fixed/learned **marginal** statistics of the column | consequence-features / raw-ECDF / φ encoder + LR | feature-level OOF AUC | **runtime-computable** |
| **4. Deployed transferable channel** | observed data only; must **transfer** across domains and run unattended | trained Level-1 φ-spine posterior | AUC/RPS/ECE/entropy, `I_gain`, coverage-state | runtime (the product) |

Strict information ordering by construction: each level's inputs are a superset of the next's
(1 knows the mechanism that generated 2's truth; 2's truth determines 3's observed data; 4 consumes 3's
statistics under added constraints). So measured performance can only fall going down — the scientific
question is **where it falls, and by how much.**

## B. What each transition loses — and the decisive 2×2

**1 → 2 loses: the mechanism family** (the parametric form of selection, hence the ability to integrate
per-row likelihood evidence optimally). **Empirically near-lossless given truth** (G1 vs E-study oracle:
0.90–0.99 vs ≈1.0 on the binary slice). Mechanism knowledge is *nearly redundant when you can see the
counterfactual* — even a linear imputer suffices in-family.

**2 → 3 loses TWO things at once**: (i) the **counterfactual truth**, and (ii) **conditional**
(target-given-predictors) access — φ reads only the marginal. These can be disentangled because the
project has already (inadvertently) run the missing cell of the 2×2:

| own-value (measured, matched protocol — §0½) | **conditional access** | **marginal only** |
|---|---|---|
| **with truth** | **Level 2: 0.90–0.99 OOF** (G1) | *(not run; trivially informative — the punched cells' truth-marginal is directly visible)* |
| **without truth** | **0.74–0.80 OOF — the mid-rung** (empty-cell run; MIXED verdict) | **Level 3: 0.574 own-value / 0.633 top-coding** (G1 baseline) |

**[SUPERSEDED by §0½]** ~~"The binding loss at 2→3 is TRUTH, not conditionality."~~ The measured form:
the 2→3 collapse decomposes into a **representation loss** (marginal-only φ discards conditional
observed-view signal: 0.574 → ≈ 0.77, ≈ 51%) **and** a **truth loss** (≈ 0.77 → 0.93–0.99, ≈ 49%).
`transfer_features` (0.523) was top-coding, cross-era, and is superseded for that idiom by the matched
no-truth-conditional measurement (0.81–0.83); it survives only as evidence that cross-era feature
results are not portable. (Level 1 remains the other substitute: the oracle reads observed data with
mechanism knowledge.)

**3 → 4 loses: little content, but adds constraints and failure modes.** The φ-spine was *designed* to
match feature-level information (Stage-0: φ 0.735 ≈ raw-ECDF 0.735), and in-distribution it does. What
the deployed level adds: the **transfer requirement** (D3: coverage-structured, pool-level; held-out
performance tracks pre-registered regime coverage at r = −0.91), **estimation instability** (repeated
seed collapses), and **loss of reliability self-knowledge** (confidently wrong off-manifold; ECE rises
while entropy falls; `I_gain` ⊥ `I_oracle`, Spearman ≈ −0.03). Levels 1 and 4 do not even *rank* cells
consistently (the detectability review's sign inversion) — the hierarchy's levels are not a single
scalar dial.

## C. The hierarchy populated (binary δ0-vs-2.5 slice, matched rate, this corpus)

| level | top_coding | own_value | provenance |
|---|---|---|---|
| 1 oracle | 0.60–0.99 across δ; BE→0 at large δ on 6/6 real-fitted X | **0.75–0.98 across δ** — *more* distinguishable than top-coding at moderate δ | P2.2c LOD oracle; P1; E-study §4 |
| 2 counterfactual | 0.88–0.95 OOF (+0.26–0.31 over φ) | **0.90–0.99 OOF (+0.30–0.42 over φ)**, incl. wealth held-out 0.98 | G1 |
| 2½ conditional, no truth | — | 0.523 OOF (dead) | `transfer_features` (P2.2c) |
| 3 marginal/φ features | 0.633 (LR LODO); 0.72–0.75 raw-ECDF/Stage-0 in the favorable config | **0.574** (LR LODO); 0.59–0.65 at the δ=2.5 extreme | G1 baseline; Stage-0; probe studies |
| 4 deployed channel | 0.64–0.74 **when regime-covered**; 0.46–0.58 uncovered/mismatched; calibration collapses off-manifold | ≈ base-rate (flat); the founding P2.2 result | D3; Stage-1 M1; P2.2c; E-study |

Two readings jump out: **(i)** the 1→2 step is small for both idioms; **(ii)** the 2→3 step is
enormous for own-value (0.99 → 0.57) and large for top-coding (0.95 → 0.63 at the LR level), while the
3→4 step is modest and *coverage-governed* for top-coding and already-at-floor for own-value.

## D. Four concepts the project has conflated — now separable

| concept | definition (this project's operational form) | measured at | top_coding | own_value |
|---|---|---|---|---|
| **Signal existence** | the observed-data law under MNAR(δ) differs from the best *plausible* MAR (named class) | Level 1 | **YES** | **YES** (P1; E-study — *not* flat at the Bayes level) |
| **Identifiability** | the difference is attributable from observed data **without counterfactual truth**, given stated assumptions; always class-relative | Levels 1 vs 2/3 | YES — and *assumption-light*: the footprint is structural (truncated tail), readable from the marginal | **only assumption-driven**: readable via an assumed mechanism family (oracle) but not by generic conditional (0.523) or marginal (0.574) statistics |
| **Transferability** | an estimator trained on some domains works on unseen domains | Level 3→4 | **conditional on footprint-regime coverage** (D3; pool-level) | n/a — nothing to transfer (floor at level 3) |
| **Deployability / runtime observability** | the estimator's required inputs exist at runtime | Level 2 vs 3/4 | yes (marginal stats are runtime-computable) | **NO for the informative quantity** — the counterfactual is runtime-unobservable by definition |

G1 is what disentangles them: own-value now has **signal existence ✓, truth-conditional recoverability ✓
(0.99, assumption-free, estimator-class-robust), identifiability ✗ (without assumptions), deployability ✗**.
Before G1, "own-value is flat" blurred all four into one negative.

## E. So what is the central bottleneck? — Two walls, not one

The question "is Lacuna's bottleneck lack of signal / transfer / identification / runtime observability?"
has **no single answer — it is idiom-dependent, and G1 lets us assign each idiom its wall:**

- **Own-value (diffuse, sensitive-item selection): TWO PARTIAL WALLS, measured (§0½, supersedes the
  original single-wall claim).** Signal exists; truth recovers it almost perfectly with *any* imputer.
  The deployment gap then decomposes: a **representation wall** (marginal-only φ discards conditional
  observed-view signal — ≈ 51% of the gap, 0.574 → ≈ 0.77; own-value is *flat to marginal features, not
  flat to conditional observed-view features*) **and** a residual **identification wall** (≈ 49%,
  ≈ 0.77 → 0.93–0.99 — readable only through truth or an assumed mechanism, i.e., exactly the untestable
  assumptions sensitivity analysis is *about*; and mechanism *identification* collapses without truth —
  the no-truth station detects *selection*, not *which* mechanism). The representation share is in
  principle addressable from the observed view (eval-only statement; **no runtime claim**); the
  identification share is moved only by assumptions (`P_prior`, mechanism families) or external
  information (metadata; deferred).
- **Top-coding (structural, truncation-type selection): a TRANSFER/COVERAGE wall at 3→4.** The footprint
  is marginal-readable and learnable; what limits deployment is footprint-regime coverage of the training
  pool (D3) plus estimation stability and reliability self-knowledge (E-study). **This wall is moved by
  data — specifically regime-cluster coverage — and by support-aware governance**, not by assumptions.
- **Runtime observability** is the *general form* of the own-value wall (Level 2 quantities are
  eval-only for every idiom), but it only *binds* where Levels 3/4 are at floor. For top-coding the
  runtime-observable shadow suffices; for own-value it does not.
- "Lack of signal" is now **ruled out** for both idioms; "lack of estimator power" is ruled out by G1's
  estimator-class robustness (even linear works given truth, in-family).

**The two walls map onto the two existing architectural commitments** — and this is the synthesis's
practical payoff: the **φ-spine + coverage gate** is the correct response to the *transfer wall*
(structural idioms), and the **named prior `P_prior`** is the correct response to the *identification
wall* (diffuse idioms). The project did not have this division of labor explicit before G1.

## F. Per-idiom state of knowledge

| | **top_coding** | **own_value** | provenance |
|---|---|---|---|
| L1 signal vs plausible MAR | ✓ strong (BE→0 at large δ) | ✓ strong — *exceeds* top-coding at moderate δ, n=384 | P2.2c, P1, E-study |
| L2 truth-recoverable, no mechanism | ✓ 0.88–0.95 OOF | ✓ 0.90–0.99 OOF (incl. wealth 0.98) | G1 |
| L2 estimator-class sensitivity | low (all 4 distinct classes pass) | low, except linear on heavy tails (wealth 0.596) | G1 |
| conditional-no-truth (**mid-rung**, matched protocol) | ✓ **0.81–0.83** (supersedes `transfer_features` 0.523 — cross-era, non-portable, historical only) | ◐ **0.74–0.80** (MIXED; ρ ≈ 0.51; detects *selection* > *mechanism* — idiom sep 0.54–0.68) | empty-cell run |
| L3 marginal footprint | ◐ 0.63 LR / 0.72–0.75 best-config | ✗ 0.57 (*flat to marginal features only* — see mid-rung row) | G1, Stage-0 |
| L4 deployed, in-coverage | ◐ 0.64–0.74 | ✗ base-rate | D3, Stage-1, P2.2c |
| L4 deployed, out-of-coverage | ✗ 0.46–0.58 + confidently wrong | ✗ | D3, E-study |
| L4 reliability self-knowledge | ✗ (`I_gain` ⊥ oracle; ECE↑ entropy↓ off-manifold) | ✗ | E-study |
| **the wall(s)** | **transfer/coverage (3→4)** | **TWO partial walls: representation ≈ 51% + identification ≈ 49% (§0½)** | this doc + empty-cell run |
| what moves the wall(s) | regime-cluster data; coverage governance | representation share: conditional observed-view features (eval-only, no runtime claim); identification share: assumptions (`P_prior`), external info | this doc + empty-cell run |

*(Future idioms slot into the same rows; the MCAR/MAR null occupies L1 trivially and anchors every
paired design.)*

## G. Implications for the dissertation narrative

1. **The core claim should change.** From *"can a learned system estimate calibrated δ-priors from
   data?"* (answer so far: weakly, for one idiom, in-coverage) to: **"a measured four-level information
   decomposition of missingness-mechanism inference — locating, per idiom, exactly where the information
   dies."** That is a stronger, more general, and fully evidenced contribution: every cell of §F carries
   a number from a pre-registered or oracle-gated study.
2. **The lab-coat fraction becomes a measurement with an upper story.** Old form: "for flat idioms,
   sensitivity analysis is domain knowledge in a lab coat." New, sharper form: **the information needed
   to govern sensitive-item sensitivity analysis *exists* — abundantly, assumption-free — in the
   counterfactual (L2 = 0.99), and is *unreachable at runtime for identification reasons only* (L3 =
   0.57).** The lab coat is not covering an absence of information; it is covering an absence of
   *identification*. This is the epistemology-of-sensitivity-analysis contribution in quantitative form.
3. **The two-wall structure justifies the architecture retrospectively and prospectively:** φ-spine +
   coverage gate ↔ transfer wall; named prior ↔ identification wall. The dissertation can present the
   architecture as *derived from* the information decomposition rather than as engineering history.
4. **Acquisition strategy inherits the split:** corpus expansion (regime clusters) addresses only the
   transfer wall; it is the correct investment for structural idioms and *provably beside the point* for
   diffuse ones. This sharpens the grant narrative: fund coverage for what coverage can fix; fund
   assumption-elicitation/validation (priors, metadata) for what it cannot.
5. **Honest negative results become load-bearing:** `transfer_features` (0.523) graduates from "a failed
   feature idea" to **the experiment that isolates truth-loss as the binding deficit**; the E-study's
   oracle-rank inversion becomes the demonstration that the levels are not a single dial.
6. **G2's value question is now precise** (not designed here): a runtime proxy could at best move
   *top-coding-like* idioms within the transfer wall; it cannot cross the identification wall for
   own-value — Level 2's inputs do not exist at runtime. Any future G2 decision should be made *against
   this ceiling*, which is exactly what the PI paused to establish.

## H. Caveats (scope of the synthesis)

All quantitative statements: binary δ0-vs-2.5 slice, matched rate 0.3, n≈384, this corpus (4 domains, 40
continuous targets), LR-level for L2/L3 comparisons, fitted-Gaussian oracle X-models, top-coding-trained
subjects for the L4 calibration facts. The 1→2 near-losslessness is at the δ-extreme; intermediate-δ
resolution and additional idioms (skip-logic, item-nonresponse) are unmeasured rows, not refutations.
None of the §G implications depend on the exact numbers — only on the *ordering* of the collapses, which
is large and consistent across imputers, domains, and idioms.

---

*No new experiments, designs, proxies, or implementations are proposed. This document fixes the
four-level hierarchy as the project's organizing scientific frame and records that G1 has already
changed the core claim: the question is no longer whether the information exists, but at which level it
dies — and the answer is different for the two idioms.*
