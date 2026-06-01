# Stage P3 — the metadata prior × data likelihood on real survey anchors

- **Date:** 2026-06-01
- **ADR:** docs/decisions/0008-metadata-prior-likelihood.md, Stage P3 (the real-data payoff).
- **Branch:** `experiment/metadata-prior`.
- **Script:** `scripts/stageP3_anchors_combined.py`. **Report:** `runs/stage0_general_baseline/stageP3_anchors_combined.json`.

## What P3 runs

The **deployed pipeline, end-to-end on real data**: a local model (Qwen2.5-14B) reads each anchor's
analyst-metadata for its dominant missing variable — domain + what it is + design, **never the mechanism
label or a mechanism-naming citation** (circularity guard) — and labels its semantic class; the frozen P1
table turns that into a prior; the composition head supplies the data likelihood; the two are pooled in
**raw evidence space** (Stage P2). 14 anchors, posterior averaged over 8 row-subsamples for n>128.

## 🟥 The honest framing (there is NO mechanism ground truth on real data) 🟥

The anchor consensus labels are themselves elicited judgement. Validating the prior against them is
**semi-circular** (it mostly confirms the model encoded the same textbook the consensus came from). So the
headline check is the **FACT tier**, where the metadata states a *design fact* the data channel cannot see
(rotation, skip-logic); the **GUT tier** (sensitive-item → MNAR) is reported as weaker face validity. A
wrong prior amplifies error on the non-identifiable axis and the data can't rescue it (Stage P2) — so the
**disagreement report is the deliverable**, not a hidden correction.

## Result 1 — FACT tier: the prior reads the design and fixes the footprint's blind spot (directionally)

Qwen labelled both PISA anchors `planned_random` and UCLA `skip_gated`, *mechanism-blind*. Combining:

| anchor | consensus | data-only (M/A/N) | combined (M/A/N) | note |
|---|---|---|---|---|
| pisa2018_gbr_rotation | MCAR | .05/.55/.40 (MAR) | .20/.47/.34 (MAR) | f_MCAR lifted 0.05→0.20, not flipped |
| pisa2022_deu_rotation | MCAR | .60/.17/.23 (MCAR) | .70/.13/.18 (MCAR) | reinforced ✓ |
| survey_ucla_textbooks | MAR | .06/.52/.42 (MAR) | .05/.60/.35 (MAR) | reinforced ✓ |

**The PISA MCAR-by-design fix — the documented Stage-E blind spot — works directionally:** mean PISA f_MCAR
**0.33 → 0.45**. The footprint reads rotated booklets as *structured* (value-independence is invisible to
it); the metadata prior supplies the value-independence and pulls f_MCAR up. PISA-2022 stays MCAR; **but
PISA-2018 only lifts 0.05→0.20 and does not flip** — the data reads it so strongly structured that the
*capped* prior (P(MCAR)≤0.70) cannot fully override it.

## Result 2 — GUT tier: the prior leans the sensitive items toward MNAR (semi-circular face validity)

Consensus argmax across the 11 gut-tier anchors: **3/11 data-only → 5/11 combined**. On the 5 NHANES
sensitive-item MNAR anchors specifically, **4/5 land MNAR after the prior** (vs 2/5 data-only — `duq_drug`
and `whq_weight` flip MAR→MNAR). This is the prior agreeing with the textbook gut feeling — real face
validity, but it mostly confirms the model encoded the same domain knowledge the consensus came from, so it
is *weaker* evidence than the fact tier. The CRAN MAR anchors stay hard (the data reads several as MCAR; MAR
is the arc-long hardest axis and the CRAN MAR consensus is itself the shakiest).

## Result 3 — the disagreements (the deliverable), surfaced not hidden

Prior-vs-data argmax disagreements: `survey_chile`, `survey_nhanes_demographics`, `nhanes_whq_weight`,
`nhanes_duq_drug`, `survey_bfi`, `survey_cars93`, `pisa2018_gbr_rotation`. Two worth calling out:

- **`survey_chile` — the "reasonable-but-wrong" prior, exactly as predicted.** Income → the prior leans
  MNAR (sensitive disclosure); the data reads MCAR; the textbook consensus is MAR (Fox: demographic-driven).
  All three disagree. The prior's MNAR lean is a defensible gut feeling that is *wrong here* — and the
  instrument shows it (disagreement 0.88) rather than asserting it.
- **`survey_nhanes_demographics` — the data resists the prior.** The income prior leans MNAR, but the data
  reads strongly MCAR (0.64) and the combined stays MCAR (the data overrides — the safety property on a
  strong data signal). Honest heterogeneity: the *other* income anchor (`nhanes_inq_income`) had a
  structure-bearing data read and the prior pushed it to MNAR ✓.

## The refinement P3 exposes: strength should scale with the prior's EPISTEMIC tier

P1 capped every prior at P(favored)≤0.70 for overridability. P3 shows that is right for the **gut-feeling
tier** (income/drug/depression — a wrong one must stay overridable, cf. `chile`) but **too weak for the
FACT tier**: a *design fact* (rotated booklet → MCAR, skip-logic → MAR, LOD → detection-MNAR) is more
reliable than the footprint's structural misread and *should* override it — which is exactly why PISA-2018
only half-fixed. The clean next step is **tiered strength**: a strong, near-overriding prior for the fact
tier (the metadata states the design), a capped/overridable prior for the gut tier (it states a hunch). This
operationalises the user's distinction: some "priors" are facts the data can't see, others are honest hunches.

## Verdict

The prior × likelihood instrument works **end-to-end on real data**: the metadata prior reads the design
(mechanism-blind), fixes the footprint's MCAR-by-design blind spot directionally, leans the sensitive items
toward consensus, the data overrides it where the data is strong, and every prior-vs-data disagreement is
reported. Perfection isn't attainable (no ground truth; the non-identifiable axis is a theorem) — but the
instrument is **honest about which channel is doing the work and where they disagree**, which was the goal.

## Update — tiered strength implemented (the refinement, same day)

Acting on the refinement above: the fact tier (`planned_random`, `skip_gated`, `lab_lod`) was raised to
P(favored)=0.85 (strong enough to override a misreading data channel); the gut tier stayed capped at 0.65
(overridable). Re-running P3:

| metric | capped (uniform 0.70) | **tiered (fact 0.85 / gut 0.65)** |
|---|--:|--:|
| FACT-tier consensus argmax (combined) | 2/3 | **3/3** |
| PISA mean f_MCAR (data 0.33) | 0.45 | **0.58** |
| PISA-2018 combined argmax | MAR (not fixed) | **MCAR ✓ (flipped)** |
| GUT-tier consensus argmax (combined) | 5/11 | 5/11 (unchanged) |
| `survey_chile` (the overridable disagreement) | data resists ✓ | data resists ✓ (unchanged) |

The tiered strength **fully fixes the PISA MCAR-by-design blind spot** (2018 flips; mean f_MCAR 0.33→0.58)
**without touching the gut tier's overridability** — exactly the intended, epistemically-motivated change.

**New honest risk it exposes:** a strong fact tier **amplifies misclassification *into* the fact tier**.
`survey_bfi` (personality items) was mislabelled `skip_gated` by the model, and the strong MAR prior then
drove the combined read hard to MAR (it matched consensus, but for the wrong reason). With the capped prior
this barely moved; with the strong one it dominates. **Mitigation / next lead:** gate the fact-tier strength
on classifier confidence or a corroborating signal, so the strong prior applies only when the fact-tier
label is itself reliable. (The gut tier is unaffected — its cap already bounds misclassification damage.)

## Caveats

- **Raw-evidence reads** (for the P2-correct combination); these differ from Stage E's *calibrated* reads,
  and **calibration of the combined posterior is still deferred** (P3 reports direction, not calibrated
  region probabilities).
- **Semi-circular gut-tier validation** (the prior encodes the same domain knowledge as the consensus).
- **Bounded-prior partial fix** on the strongest-structured PISA case → motivates tiered strength (above).
- Survey-scoped; small/sparse anchors give extreme raw reads.

## Reproduce

```
python scripts/stageP3_anchors_combined.py    # phase 1 Qwen labels (freed), phase 2 likelihood + combine
```

## Files

- Script: `scripts/stageP3_anchors_combined.py`; report `runs/stage0_general_baseline/stageP3_anchors_combined.json`.
- Prior: `lacuna/priors/metadata_prior.py` + `scripts/metadata_prior/prior_table.json`; anchors `lacuna_survey/anchors.py`.
