# Stage 1 — Can the architecture read out the mechanism per column?

- **Date:** 2026-05-29
- **ADR:** docs/decisions/0006, Stage 1.
- **Encoder init:** RUN-071 (`stage0_general_baseline`, current-code 10-feature baseline).
- **Code:** `lacuna/models/column_head.py`, `lacuna/data/mixed_batch.py`,
  `scripts/stage1_column_head.py`. Eval on 7 held-out datasets × mixed compositions.
- **Status:** complete; gives a precise Stage 2 lead.

## Question

Add a per-column readout head on the encoder's per-token representations and train it on
mixed-mechanism datasets with per-column labels. Can the architecture recover a per-column
MCAR/MAR/MNAR posterior at all? Two modes: **frozen probe** (head only — is the signal already
in the baseline reps?) and **fine-tune** (encoder + head end-to-end).

## Results (held-out mixtures, supervised columns only)

| Mode | per-col acc | MCAR rec | MAR rec | MNAR rec | ECE | composition L1 |
|---|--:|--:|--:|--:|--:|--:|
| frozen probe | 0.425 | 0.467 | **0.724** | **0.040** | 0.032 | 0.448 |
| fine-tune | 0.473 | 0.457 | **0.741** | **0.184** | 0.053 | 0.476 |

Confusion [true × predicted]:
- frozen:    MCAR `[1117, 1245, 32]`, MAR `[651, 1904, 76]`, MNAR `[547, 1665, 92]`
- fine-tune: MCAR `[1093, 394, 907]`, MAR `[407, 1950, 274]`, MNAR `[405, 1474, 425]`

## Interpretation

1. **Per-column readout recovers MAR — the class the dataset-level model erased on mixtures.**
   MAR recall is 0.72–0.74 per-column, well-calibrated (ECE 0.03–0.05). Recall that Stage 0
   showed the *dataset-level* posterior drives MAR to the argmax essentially never on mixtures
   (it collapses to MNAR). So moving to per-column readout genuinely *resurrects* MAR.

2. **Per-column MNAR is the new hard problem.** MNAR recall is 0.04 (frozen) → 0.18 (fine-tune);
   MNAR columns are predominantly misread as MAR in both modes. Fine-tuning the encoder helps
   only modestly and does not solve it.

3. **The failure is the mirror image of Stage 0.** Dataset-level on mixtures collapses toward
   **MNAR** (MAR erased); per-column readout from encoder reps collapses toward **MAR** (MNAR
   erased). Two different readout geometries, two opposite single-class attractors.

4. **Diagnosis — the head is structurally missing the MNAR signal (precise Stage 2 lead).**
   The per-column head reads *only* the encoder's token representations. But MNAR here is
   *self-censoring* — missingness depends on a column's own (unobserved) value — and the
   dataset-level model detects that through the **reconstruction-error pathway**
   (`MNARSelfCensoringHead` natural errors feed the MoE gate), *not* the encoder evidence. The
   column head never sees those per-column reconstruction errors, so it lacks the very feature
   that separates MNAR from MAR. This explains why MNAR recall is poor in *both* frozen and
   fine-tuned modes (the encoder reps simply don't carry a strong per-column self-censoring
   signal), and why fine-tuning the encoder alone can't fully fix it.

5. **Composition estimate (Q2) is not yet solved.** Composition L1 stays ~0.45–0.48, bottlenecked
   entirely by the missed MNAR mass (mis-assigned to MAR). Per-column MAR/MCAR composition is
   recoverable; MNAR composition is not, yet.

## Answer to the Stage 1 question

**Yes — the architecture can read out per-column mechanism, but with a sharp asymmetry.** The
encoder's representations support per-column **MAR (and partially MCAR)** readout that is
well-calibrated and far better than the dataset-level model manages on mixtures. **Per-column
MNAR, however, is not readable from encoder representations alone** — frozen or fine-tuned —
because the discriminative self-censoring signal lives in the reconstruction-error pathway the
head does not consume. That is an attributable result, not a wash.

## Next (Stage 2 lead)

Wire the **per-column reconstruction errors** (the reconstruction heads already produce
`per_cell_errors [B, R, C]`; pool per column) — and optionally per-column missingness features —
into the per-column head, mirroring the dataset-level gate's three-signal input. The Stage 1
result predicts this is what unlocks per-column MNAR. This is a Stage 2 design step under ADR-0006.

## Caveats / scope

- Control/encoder is the current-code 10-feature baseline (RUN-071), not the 16-feature
  dissertation model (not loadable by current code).
- Mixtures use the three canonical generators (MCAR/MAR/MNAR-self-censoring) in the clean-MAR
  regime — a documented subset of the registry.
- Pooling over rows is a masked mean; attention pooling is untried. Fine-tune was 15 epochs at
  lr 1e-3 — not exhaustively tuned, so the *level* of MNAR recall may shift, but the structural
  recon-signal gap (point 4) is the more likely bottleneck than training budget.
