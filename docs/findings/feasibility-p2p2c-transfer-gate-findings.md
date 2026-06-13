# P2.2c — Transfer-Robust Features — No-Training LR Gate — Findings

*Branch `p2/delta-prior-rearchitecture`. Spec: `feasibility-p2p2c-transfer-features-note.md`.
No model was trained (the gate failed). Includes a **correction** to the prior consequence-A/B
findings. Stop-for-review.*

## Result: GATE FAILED — the transfer-robust features do NOT transfer (and are worse than the old ones)

No-training logistic-regression gate, binary δ=0 vs δ=2.5, train on the train pool / test on the
held-out pool (dataset selection **decoupled from δ** — see the bug note below):

| feature set | binary AUC in-dist | **binary AUC OOF** | per-held-out-dataset OOF | 7-bin OOF acc |
|---|---|---|---|---|
| old 17 marginal | 0.876 | **0.734** | cps1985 0.865 / workinghours 0.676 | 0.217 |
| **new transfer (10)** | 0.668 | **0.523** | cps1985 0.464 / workinghours 0.581 | 0.150 |

**Gate (proceed to neural A/B only if transfer-feature OOF AUC ≥ 0.65): FAIL** — new-transfer OOF
AUC 0.523 < 0.65. Per the pre-registered decision rule: **stop, do not run the neural A/B, do not
blame the encoder.** The transfer-robust hypothesis (predictor-conditional / predictor-ruler
features generalize better) is **rejected** — they generalize *worse* (0.523 vs 0.734). Referencing
the predictor added column-specific structure (the predictor identity/correlation varies across
datasets) rather than removing it.

## A bug I found and fixed — it corrects the PRIOR round's conclusion

While building the gate I found a **phase-locking confound** in the gate's data generation (and in
my earlier inline diagnostics): `delta = deltas[i % len(deltas)]` together with
`raw = pool[i % len(pool)]` locks each test dataset to a single δ when both lists have length 2
(binary δ + 2 test datasets) — so the classifier was separating **datasets**, not δ. Fixed by
choosing the dataset from an independent RNG stream.

**Consequence (intellectual honesty):** the prior consequence-A/B findings reported the 17 features'
binary OOF AUC as **0.43 ("direction flips, doesn't transfer")**. That number was the bug. The
**corrected** value is **0.734 — the 17 features DO transfer for coarse δ.** The 7-bin numbers in
that doc were never affected (grid length 7 vs pool length 2 are coprime) and stand. The neural A/B
itself was **not** affected (the example sources pick datasets randomly, independent of δ), so the
"both arms floor" model result remains valid.

## The corrected, honest picture of LOD transfer

- **Coarse δ transfers out-of-family.** δ=0 vs δ=2.5 is recoverable on held-out datasets from the
  simple 17 marginal features (OOF AUC 0.734) — *not* a flat / non-transferring story.
- **Fine 7-bin δ does NOT transfer well.** OOF 7-bin accuracy is 0.217 (old) / 0.150 (new) — barely
  above chance 0.143. The fine resolution exceeds what generalizes.
- **The 7-bin neural model floored** despite the coarse signal transferring — consistent with: the
  7-bin RPS task washes out a signal that is essentially **coarse**, and the model sits near the weak
  7-bin transferable ceiling.
- **My predictor-referencing features made it worse**, not better.

So the bottleneck is **not** "consequence features can't transfer" (coarse δ transfers fine). It is
that the **7-bin formulation is finer than what transfers out-of-family**, and (separately) the model
does not convert the coarse-transferable signal into a calibrated 7-bin prior.

## Decision-rule outcome + recommendation

Per the gate's pre-registered rules, the transfer-feature iteration is **rejected** (LR does not
transfer; STOP; no neural A/B; do not blame the encoder). Per the standing instruction —
*"if this fails, pause and reassess whether LOD transfer requires a different modeling formulation
rather than piling on more features"* — I recommend **pausing feature engineering** and, for PI
decision, **reassessing the FORMULATION to match what actually transfers**:

1. **Coarse / curriculum δ target.** Coarse δ transfers (binary OOF 0.734). A **binary (δ=0 vs
   large-δ) or 3-bin** δ-prior — using the existing 17 marginal features (which already transfer
   coarsely) — is the natural test of the spectrum in the held-out learned channel. This is a
   *formulation* change (already-built coarse-bin idea from the deferred rung 2), not more features.
   Decision rule to pre-register: does a coarse δ-prior beat its base rate out-of-family while
   own-value stays flat?
2. **Model-integration check (if coarse still floors).** The 17 features transfer coarse δ (0.734)
   but the model floored — a possible integration issue (features swamped by encoder evidence). A
   features-only / features-emphasized head on the coarse target would isolate this.

I do **not** recommend more feature variants — the gate just showed added feature engineering made
things worse, and coarse δ already transfers with the simple features.

## Discipline / status

Frozen feature list; no-training gate run before any training; both feature sets reported in-dist +
OOF + per-dataset + coefficient signs; gate threshold (0.65) pre-declared. Bug found, fixed,
committed, and its effect on prior claims disclosed and corrected. No model trained. Full suite green
(1335 passed, 1 skipped). `runs/p2p2c-transfer-gate.json` saved.

**Bottom line:** transfer-robust features rejected (OOF 0.523 < gate). Correcting a phase-locking
bug shows the *simple* features already transfer **coarse** δ out-of-family (0.734); the unmet target
is **fine 7-bin** transfer. Recommend pausing feature work and reassessing the **formulation**
(coarse/curriculum δ) rather than adding features. Stop for review.
