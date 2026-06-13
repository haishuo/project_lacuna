# Stage 0 — Column-Primary Representation Probe: findings

*Diagnostic findings. Status: **for PI review.** Analysis/probe only — no architecture was built; no
BERT-backbone patch; no new mechanism; no metadata channel (the PI's Stage-0 constraints). Governed by
`NORTH-STAR.md` (§4.9 weak-proxy discipline, §5 metrics) and the Option-C spec
(`PROPOSAL-OptionC-column-primary-architecture-spec.md` §7–§8, "Stage 0"). Script:
`scripts/probe_stage0_column_primary.py`; JSON: `runs/probe-stage0-column-primary.json`.*

## 0. Question

Can a **distribution-native, column-primary** representation of the *observed target values* recover the
raw-ECDF LOD signal that the old row-primary BERT backbone loses
(`probe-encoder-representation-findings.md`: old reps ~0.52 vs raw-ECDF ~0.72)? This is the cheap,
decisive gate that *earns or refuses* the Option-C rewrite **before** any rewrite.

## 1. Result (binary LOD δ0-vs-δ2.5; held-out leave-datasets-out; train=8 / test=2; 3 seeds; leakage-gated)

| representation (OOF AUC) | LOD | own-value control |
|---|---|---|
| 1. raw-ECDF LR (fixed column-primary, the reference) | 0.735 | 0.646 |
| 2. **old encoder reps** linear probe (δ-trained, frozen; row-primary) | **0.548** | 0.565 |
| 3. **learned column-primary φ** (DeepSets over observed values + quantile pool) | **0.735 ± 0.002** | 0.591 ± 0.049 |

φ = per-value MLP → masked-quantile pool (12 quantiles + max, reusing the tested
`distributional_stream.masked_quantile_pool` on **raw** values) → small MLP → 2 logits; **target column
only**, no BERT, no LOD detector; trained end-to-end, early-stopped on the val datasets. Leakage gate
passed for both idioms.

## 2. Reading

1. **The column-primary representation recovers what the backbone loses.** φ **0.735** vs old-encoder
   **0.548** — a **+0.19** gap, essentially zero-variance across 3 seeds (0.737/0.732/0.735), in the
   *same* held-out transfer regime the BERT backbone fails. This is the decisive, pre-registered
   "EARNED" condition: φ ≫ old-encoder and φ ≈ raw-ECDF on LOD.
2. **Transfer regime is NOT the bottleneck.** φ transfers to held-out datasets at 0.735 — the same OOD
   leave-datasets-out split where the backbone sits at 0.548. So the §8 result was not "the task doesn't
   transfer"; it was "the *row-primary* representation doesn't transfer the within-column signal." The
   Stage-0 "NOT EARNED → blame transfer regime" branch is ruled out.
3. **Convergent with §8.** Old reps don't carry/preserve the signal (§8: probe ~0.52, preservation
   R²≤0); a distribution-native rep does (Stage 0: 0.735). The representational *family* is the binding
   constraint, and column-primary is the fix.

## 3. Honest caveats (do not oversell the rewrite)

1. **φ only *matches* the fixed raw-ECDF (0.735 = 0.735); it does not exceed it.** There is **no
   within-column *learning* headroom** at the target marginal — a plain raw-ECDF logistic regression
   already captures this signal. Consequently a *neural* column-primary architecture cannot justify its
   complexity on the target marginal alone; its incremental value must come from the part Stage 0 did
   **not** test — the **across-column predictor-conditional reference / deviation** module (Option-C
   Stage 2). **That payoff is unproven, and predictor-referencing has failed once before**
   (`transfer_features` OOF AUC 0.523, DECISION-MEMO §3). So Stage 0 earns the *substrate*, not the full
   architecture's marginal value over a simple raw-ECDF deployment.
2. **own-value is not perfectly flat at the binary δ=2.5 extreme** (raw 0.646, φ 0.591). Strong
   self-censoring leaves *some* marginal footprint at the extreme; the **spectrum holds** (LOD 0.735 ≫
   own-value 0.591) but the correct statement is "LOD materially more detectable," not "own-value at
   chance." This matches the coarse-A/B finding that *binary presence/absence of strong truncation*
   transfers while fine δ-magnitude does not.
3. **Weak-proxy discipline (§4.9).** The decisive direction here is a **positive** (a representation
   *recovers* signal — §4.9 permits positive findings from any probe), and it is stable across seeds and
   convergent with §8, so the "EARNED" reading is sound. The caveats above bound *what* was earned.

## 4. Decision bearing

Per the pre-registered Stage-0 criterion (φ materially exceeds old-encoder **and** approaches/exceeds
raw-ECDF on LOD, own-value materially lower): **the column-primary direction is EARNED.** The old
backbone was a credible binding constraint and a distribution-native substrate removes it.

**But "earned" is scoped:** it earns the column-primary **substrate** (φ), not yet the full Option-C
value proposition. The genuinely new, still-unproven question is **Stage 2** — does an across-column
predictor-conditional **reference/deviation** module add signal *beyond* the marginal raw-ECDF (which φ
already equals)? That is where Option C either justifies a neural architecture over a one-line raw-ECDF
baseline, or does not.

**Recommended next step (for PI; spec/analysis, not a full build):** write the **Stage-1+2 spec** as one
increment — (a) φ + minimal δ-head (already ≈ raw-ECDF, include only as the calibrated, abstaining,
δ-bin version on the full grid), and (b) the **across-column reference-deviation module**, pre-registered
against the sharp question *"does conditional deviation beat the marginal φ / raw-ECDF OOF, while
own-value stays lower?"* If the PI prefers, gate the build behind a **Stage-1.5 analysis probe** of the
reference idea (a cheap conditional-deviation feature vs the marginal, no full architecture) given the
prior predictor-referencing failure. **No implementation until that spec is approved.**
