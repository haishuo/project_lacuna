# P2.2c — Consequence-Feature A/B — Findings

*Branch `p2/delta-prior-rearchitecture`. Note: `feasibility-p2p2c-consequence-features-note.md`;
prior A/B: `feasibility-p2p2c-lod-ladder-findings.md`. Stop-for-review.*

> **CORRECTION (2026-06-05, see `feasibility-p2p2c-transfer-gate-findings.md`).** The **binary**
> δ0-vs-δ2.5 LR AUCs in this doc (in-dist 0.94 / **OOF 0.43 "direction flips"**) were produced by a
> **phase-locking bug** (dataset index and δ index both cycled mod-2, locking each test dataset to one
> δ — the LR separated *datasets*, not δ). The **corrected** binary numbers (independent dataset
> sampling) are **in-dist 0.876 / OOF 0.734** — i.e. the 17 features **DO transfer for coarse δ**; the
> "generalization gap / direction flips" framing below is **wrong for the binary contrast** and is
> superseded. UNAFFECTED and still valid: the **7-bin** OOF numbers (≈ chance — fine δ does not
> transfer well; grid len 7 vs pool len 2 are coprime) and the **neural A/B "both arms floor"** result
> (the example sources sample datasets randomly, independent of δ). Net corrected story: coarse δ
> transfers out-of-family; FINE 7-bin δ does not; the 7-bin model floored at the weak 7-bin
> transferable ceiling. Read the body below with the binary-AUC claims replaced accordingly.

## Result: both arms still floor out-of-family — but the diagnostics reveal a GENERALIZATION gap, not a wall

Same held-out (leave-datasets-out) A/B, target-conditioned, **+ the fixed 17 consequence features**
on the head input. Both arms float at the uniform RPS floor:

| arm | held-out RPS | (uni−RPS)/SE | bin | adj | entropy | leakage |
|---|---|---|---|---|---|---|
| own-value | 0.1903 | +0.02 | 0.136 | 0.421 | 2.806 | clean |
| LOD | 0.1906 | −0.02 | 0.171 | 0.436 | 2.805 | clean |

Adding the consequence features did **not** make LOD learn out-of-family. But *why* is now precisely
pinned down by no-training diagnostics (logistic regression on the 17 features alone).

## Diagnostics (no training) — the features carry strong signal that DOES NOT TRANSFER

| probe (LR on the 17 features) | in-distribution | leave-datasets-out |
|---|---|---|
| binary δ=0 vs δ=2.5, AUC | **0.942** | **0.431** (below chance!) |
| 7-bin accuracy (chance 0.143) | 0.230 | 0.208 |

- **In-distribution the signal is strong and linearly accessible** (AUC 0.94 for the extreme
  contrast). The consequence features DO expose the LOD truncation — NORTH-STAR §6 is *supported*
  at the in-distribution level.
- **Out-of-family it collapses** (AUC 0.43 — the feature→δ direction literally *flips* across
  datasets). The map from these features to δ is **dataset-specific** and does not transfer.
- The trained model's held-out bin-acc (0.171) ≈ the LR's out-of-family accuracy (0.208) ≈ chance.
  So **the model is at the *transferable* ceiling — it is NOT under-exploiting the features.** There
  is simply nothing out-of-family to exploit with this representation.

## Diagnosis (and two honest self-corrections)

The held-out floor is **a generalization / transfer gap**, not non-identifiability and not a broken
model: the LOD signal is present and in-distribution-learnable, but the consequence features encode
it in a **column-distribution-specific** way (z-scored within-observed shape statistics mean
different things for a skewed wage column vs an hours column), so a decision boundary learned on the
train datasets does not carry to the held-out ones.

Two hypotheses I floated mid-investigation, both corrected by the diagnostics:
1. *"z-scoring destroyed the signal"* — **wrong**: the z-scored features beat my robust-IQR
   alternative (0.23 vs 0.14 7-bin) and reach AUC 0.94 in-distribution. The signal is there.
2. *"the model under-exploits the features"* — **wrong**: the model matches the LR's out-of-family
   accuracy (~0.2 ≈ chance). It exploits what is transferable, which is ~nothing.

## What this means (and why it is the GOOD kind of negative)

- **The detectability spectrum EXISTS** — now at three levels: oracle (LOD BE→0 ≫ own-value),
  *and* in-distribution learned (LOD features AUC 0.94 ≫ own-value, which was flat even
  in-distribution in the P2.2b cardinality probe).
- **The held-out (out-of-family) failure is a GENERALIZATION gap** — exactly the axis NORTH-STAR §3
  deliberately moved the project onto: *"identification failure is fatal and unfixable, while a
  generalization gap is estimable and improvable."* This is the improvable kind of problem, not a
  wall.
- We have **not** demonstrated the spectrum in the *out-of-family* learned channel (the §8.1 headline
  metric). We HAVE demonstrated it at the oracle and in-distribution levels, and localized the
  remaining gap precisely to **transfer of the feature→δ map across survey columns**.

This also re-reads the whole P2.2c arc cleanly: own-value = signal absent even in-distribution
(flat-likelihood idiom); LOD = signal present and in-distribution-learnable, but the representation
of it does not generalize across datasets at matched rate.

## Recommended next (await review — bounded, NOT encoder redesign)

The fix is **transfer-robust consequence representation**, on the §6 Level-1 coverage axis
(*"diversity/domain-randomization training and consequence-features"*):
1. **Reference-normalized consequence features** — encode the observed target *relative to a
   per-column MAR reference* so the feature→δ map is column-agnostic (e.g. compare the observed
   target's tail to the predictor's, or to a MAR-resampled baseline; rank/ECDF-based features;
   observed-vs-MAR-expected tail deficit). The goal: a feature whose relationship to δ is the same
   across columns.
2. **Domain-randomization** — more train datasets / wider column diversity so a transferable map can
   be learned (§6 Level-1).
3. Re-run the same held-out A/B. **LOD learns out-of-family while own-value stays flat** would finally
   demonstrate the spectrum in the held-out learned channel.

This stays bounded (features + training diversity), not an encoder/tokenization/loss redesign. The
decision of record is the PI's: iterate on transfer-robust features, or accept the current
localization (spectrum proven; out-of-family transfer is the open, improvable problem).

## Discipline / status

Feature list frozen before the run; manifest records `consequence_features_enabled=True` + the frozen
17-feature schema; both arms leakage-clean and `kind=main`. No encoder/tokenization/loss/generator
change — only the fixed consequence vector was concatenated. The 95th-pct diagnostic stayed external
(not a training target). Full suite green (1328 passed, 1 skipped). `runs/p2p2c-cons-*.json` saved.

**Bottom line:** consequence features make the LOD signal linearly accessible *in-distribution*
(AUC 0.94) — §6 supported — but the feature→δ map does **not transfer across datasets** (OOF AUC
0.43), so the held-out model floors. This is a **generalization gap** (NORTH-STAR §3, *improvable*),
not non-identifiability. The spectrum is real; transferable representation is the open problem.
Stop for review.
