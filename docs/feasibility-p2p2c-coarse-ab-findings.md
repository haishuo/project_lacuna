# P2.2c — Coarse / Curriculum δ A/B — Findings

*Branch `p2/delta-prior-rearchitecture`. Spec: `PROPOSAL-P2.2c-coarse-delta-formulation-audit.md`.
Binary diagnostic + 3-bin main A/B. Stop-for-review.*

## Result: the coarse milestone is NOT met — the held-out learned δ-prior does not beat base-rate, and underperforms the LR

Held-out (leave-datasets-out) A/B, existing 17 marginal features. **Baselines (LR on the 17
features):** binary δ0-vs-δ2.5 OOF AUC **0.679** (in-dist acc 0.733); coarse3 OOF acc **0.477**
(in-dist 0.507; chance 0.333; majority-class ≈ 0.5).

| run | model | family | RPS | (uni−RPS)/SE | **(base−RPS)/SE** | bin-acc | adj | AUC | leak |
|---|---|---|---|---|---|---|---|---|---|
| binary (MAIN) | features-only | LOD | 0.2491 | +0.57 | +0.57 | 0.529 | 1.00 | **0.581** | ✅ |
| binary (MAIN) | features-only | own-value | 0.2500 | −0.05 | −0.05 | 0.500 | 1.00 | 0.499 | ⚠ False |
| 3-bin (MAIN) | features-only | LOD | 0.1980 | +2.45 | **−0.21** | 0.464 | 0.857 | — | ✅ |
| 3-bin (MAIN) | features-only | own-value | 0.1988 | +2.40 | **−0.29** | 0.457 | 0.850 | — | ✅ |
| 3-bin (diag) | encoder+features | LOD | 0.2010 | +1.92 | **−0.45** | 0.450 | 0.836 | — | ✅ |
| 3-bin (diag) | encoder+features | own-value | 0.2021 | +2.24 | **−0.68** | 0.450 | 0.857 | — | ✅ |

**The `(uni−RPS)/SE = +2.4` "beats uniform" is a base-rate artifact** — the 3-bin marginal is
imbalanced (grid → bins {0; weak×2; strong×3}), so predicting the marginal beats uniform. The honest
baseline is **base-rate RPS, and no arm beats it** (−0.2 to −0.7 SE). The models learned to predict
the marginal, nothing discriminative.

## What this says (sober, decision-rule-aligned)

1. **No spectrum separation in the learned channel.** 3-bin LOD (0.1980) ≈ own-value (0.1988) — the
   learned δ-prior does not distinguish the signal-bearing idiom (LOD) from the flat one. The
   milestone *"LOD coarse learns OOF while own-value stays flat"* is **not** achieved.
2. **The learned model does not beat the LR.** Features-only binary LOD AUC 0.581 < LR 0.679; 3-bin
   neural sits at base-rate while the LR clears chance (0.477). Per the pre-registered rule —
   *"if features-only fails to exceed the LR baseline, we have not demonstrated that a learned model
   adds value over the simpler formulation."* That is the outcome. (Encoder+features ≈ features-only
   ≈ base-rate, so this is not merely an encoder-integration problem — both are at the floor.)
3. **The transferable signal is WEAK — weaker than the 0.734 single draw suggested.** This run's LR
   binary OOF AUC is 0.679, and **coarse3 (3-bin) does not transfer even for the LR** (OOF acc 0.477,
   below majority-class ~0.5). So the transferable structure is essentially the *binary extreme*
   contrast (δ=0 vs strong), at modest strength — not a calibrated 3-bin magnitude.
4. **A faint spectrum whisper remains** at the feature/LR level: binary LOD AUC 0.68 vs own-value
   ≈ chance. But it is too weak for the learned channel to convert into a calibrated coarse prior
   that beats base-rate out-of-family.

This matches the decision rule *"binary learns but 3-bin fails ⇒ the recoverable signal is
presence/absence of strong truncation, not calibrated δ magnitude"* — and adds that even the binary
recovery by the **learned** model is weak (0.581), below the LR (0.679).

## Honest status of the detectability-spectrum claim

- **Oracle level:** spectrum real (LOD BE→0 ≫ own-value).
- **Feature/LR level, out-of-family:** spectrum faint (binary LOD AUC ~0.68 vs own-value ~chance);
  3-bin does not transfer.
- **Held-out LEARNED δ-prior:** spectrum **not** demonstrated — LOD ≈ own-value at base-rate; the
  learned model adds no value over the LR.

So the corrected, fully-qualified headline: **the detectability spectrum is real at the oracle level
and faint at the OOF feature level, but is NOT yet demonstrated in the held-out learned channel —
the transferable LOD signal is weak (binary-extreme only) and the learned δ-prior does not beat
base-rate.** No over-claim.

## Flags / discipline notes

- **own-value binary `leak=False`** (one arm). It is `kind="ablation"` (binary is a diagnostic) so it
  did not block; the arm is at chance (AUC 0.499) regardless. To audit before any reuse: with only
  δ∈{0,2.5} the δ→rate leakage statistic is small-sample; re-check at higher n before trusting any
  own-value binary number.
- Features frozen (the existing 17; no new features, no transfer set). No encoder/tokenization/loss
  change. Both 3-bin main arms leakage-clean and manifest-validated (`kind=main`, num_bins=3).
  Full suite green (1346 passed, 1 skipped). `runs/p2p2c-coarse-*.json` saved.

## Recommendation (await review) — pause and reassess, per the standing instruction

We are at the point pre-specified for reassessment (*"if this fails, pause and reassess whether LOD
transfer requires a different modeling formulation rather than piling on more features"*). The
evidence now:
- own-value = flat boundary;
- LOD = oracle-separable, but the **held-out learned recovery is weak** (binary-extreme only, AUC
  ~0.68 at the LR ceiling; 3-bin and the learned model do not beat base-rate).

Options for the PI (I do **not** recommend more features or more bins):
1. **Accept the qualified result and stop the LOD-learning push:** the spectrum is real at the
   oracle/feature level but the held-out learned δ-prior does not yet recover it usefully; record
   this as the boundary of the footprint-learning channel on real survey X for these idioms.
2. **Domain-randomization** (many more train survey datasets) to strengthen transfer — but the LR
   ceiling (binary ~0.68, 3-bin ≈ base-rate) suggests limited headroom; weigh cost vs expected gain.
3. **Step back to the strategic level** (North Star §3/§8): the project's defensible contribution may
   be the *oracle/identifiability map* + calibrated wide priors + sensitivity reporting, rather than
   a learned δ-estimator that beats base-rate out-of-family on these survey idioms.

**Bottom line:** coarse formulation did not rescue it — the held-out learned δ-prior sits at
base-rate for both LOD and own-value and underperforms the LR; the transferable LOD signal is weak
(binary-extreme). The detectability spectrum is real at the oracle level but **not demonstrated in
the held-out learned channel**. Stop for review and a strategic reassessment.
