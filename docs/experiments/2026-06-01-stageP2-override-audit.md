# Stage P2 — the prior×likelihood combiner + the override audit

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md, Stage P2.
- **Branch:** `experiment/metadata-prior`.
- **Module:** `lacuna/priors/metadata_prior.py` (`aggregate_column_priors`, +5 tests). **Script:**
  `scripts/stageP2_override_audit.py`. **Report:** `runs/stage0_general_baseline/stageP2_override_audit.json`.

## What P2 builds

The combiner: per-column priors are aggregated to a dataset prior by **missing-cell weight**
(`aggregate_column_priors`, the ADR-0007 by-cell denominator), then pooled with the composition head's
likelihood Dirichlet by **summing evidence** (`combine_prior_likelihood`). The **override audit** then
injects controlled priors (flat / favouring each mechanism, at the table's capped strength r=0.70) onto the
real likelihood over 640 semi-synthetic datasets with known realised composition, and measures per axis:

> **override-survival ratio = (posterior shift caused by the prior) / (prior-alone shift)** — low = the
> data overrides the prior; high = the prior decides.

## Result 1 — combine in RAW evidence space, not calibrated (a design finding the audit forced)

| combine in… | data α₀ | MCAR-axis survival | split-axis survival |
|---|--:|--:|--:|
| τ-calibrated likelihood | 4.7 | 0.80 | 0.86 |
| **raw likelihood evidence** | 11.6 | **0.44** | 0.56 |

The Stage-D temperature (τ=4.99) flattens the data to α₀≈4.7 — *weaker* than the capped prior (α₀≈6.7) — so
combining there lets the prior **dominate both axes** (survival 0.80/0.86): a wrong prior would corrupt even
the well-identified MCAR axis (commitment 1 violated). Combining with the **raw** head evidence (α₀≈11.6),
where the data's differential informativeness lives (sharp on MCAR, flat on the split — Stage C), restores
the seam: the data overrides the prior **more on the identifiable axis (0.44)** than on the non-identifiable
split (0.56). **Conclusion: pool prior + likelihood in raw evidence space; calibrate the combined posterior
afterwards** (Stage P3). The override differential is directionally correct but modest — the raw data's
total evidence mass dilutes the prior on *both* axes — and is sensitive to the prior-strength↔data-evidence
balance (a joint strength/temperature fit is a P3 refinement).

## Result 2 — the value: a correct prior halves the error where the data is at chance

Split-MAE (the non-identifiable MAR-vs-MNAR axis) on the 456 structured datasets, raw-evidence combine:

| | split-MAE |
|---|--:|
| likelihood-only (flat prior) | 0.247 |
| **+ CORRECT prior** | **0.158**  (−36%) |
| + WRONG prior | 0.416 |

This is the prior×likelihood thesis, demonstrated: on the axis where the composition arc was stuck at
~chance (Stage F), a **correct prior cuts the split error by a third** — it supplies signal exactly where
the data is silent. And the honest cost: a **wrong** prior *inflates* the error to 0.416 and **the data
cannot rescue it** on this axis (it is non-identifiable — the data has nothing to override with). That is
precisely why the prior must stay **auditable** and be **reported as a separate channel** (commitments 1, 3):
its value on the non-identifiable axis is real but *uncheckable from data*.

## Interpretation

The two channels behave as ADR-0008 designed: the **data overrides the prior where the data is informative**
(the identifiable MCAR axis — the safety property), and the **prior decides where the data is silent** (the
non-identifiable split — the value property, and the risk). The asymmetry is the honest seam, now operational:
the instrument can lean MAR-vs-MNAR using the metadata prior *without* claiming to identify it from data, and
a strong contrary data signal still wins on the part the data can actually measure.

## Caveats / what P2 does not yet do

- **Injected, oracle priors** (correct/wrong/flat by construction) — this tests the *combiner mechanics*, not
  the metadata prior's real accuracy. The real metadata-authored prior on real anchors is Stage P3.
- **Combined-posterior calibration is deferred to P3** (here we combine in raw evidence and report the
  Stage-D temperature only as the over-flattening diagnostic). A combined-posterior temperature should be
  re-fit; the prior strength and that temperature ideally fit jointly.
- **Modest override differential (0.44 vs 0.56)** — directionally correct, bounded by the raw data's total
  evidence mass diluting the prior on both axes; not a clean 0-vs-1 separation.
- Semi-synthetic, survey-scoped, as throughout the arc.

## Reproduce

```
python scripts/stageP2_override_audit.py     # loads the Stage-C/D instrument; injects controlled priors
```

## Files

- Combiner + aggregation: `lacuna/priors/metadata_prior.py`; tests `tests/unit/priors/test_metadata_prior.py`.
- Audit: `scripts/stageP2_override_audit.py`; report `runs/stage0_general_baseline/stageP2_override_audit.json`.
