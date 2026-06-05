# P2.2c — Coarse / Curriculum δ Formulation — Implementation Audit / Spec

*Branch `p2/delta-prior-rearchitecture`. Governed by `NORTH-STAR.md` (§3 generalization axis, §8.1
ladder) and the corrected transfer evidence (`feasibility-p2p2c-transfer-gate-findings.md`).
**SPEC ONLY — no code until approved.** Tests whether the learned channel can recover and report the
**coarse, transferable** LOD δ signal out-of-family, using the EXISTING 17 marginal features — no new
features, no new representation.*

## 0. Premise (the corrected evidence this builds on)

- own-value self-censoring = flat idiom (no transferable signal).
- LOD/top-coding = **coarse δ transfers out-of-family** (binary δ0-vs-δ2.5 LR OOF AUC **0.734** from
  the existing 17 marginal features); **fine 7-bin δ does NOT transfer** (OOF acc ~0.22 ≈ chance).
- ⇒ The bottleneck is **resolution**, not signal or feature-transfer. Match the target to what
  transfers: start coarse.

## 1. Target formulation (coarse → curriculum)

- **Step 1 — BINARY diagnostic:** δ=0 vs large-δ. δ-grid = `{0.0, 2.5}` via the existing LOD
  generator; label = `1[δ>0]`. The cleanest "can the learned channel recover the transferable
  coarse signal?" question. (Binary is a labeled DIAGNOSTIC, never the P2 main objective —
  charter; reported as AUC, not as a Lacuna δ-prior.)
- **Step 2 — 3-BIN ordinal (the main coarse run):** `{δ=0} | weak (0<δ≤τ] | strong (δ>τ)`, **τ=1.0**
  (from the deferred rung-2 proposal; confirm against the grid). δ-grid spanning all three, e.g.
  `{0.0, 0.4, 0.75, 1.25, 1.75, 2.5}` → bins `{0,1,1,2,2,2}` (stratified for balance). Ordinal,
  RPS-scored — a real (coarse) δ-prior, not a binary classifier.
- **7-BIN remains the eventual target, NOT the immediate success criterion.** A curriculum: prove
  coarse first.

## 2. Why this is not backsliding

- Binary/3-bin is **not** the final Lacuna output — the destination is a calibrated δ-prior +
  sensitivity report. This is a **curriculum / feasibility rung**: demonstrate the learned channel
  can recover the *transferable* coarse signal before demanding fine δ resolution. It is the honest
  analogue of "establish the easy case, then add difficulty" — and it is scoped: coarse now, fine
  later, 7-bin still the goal.
- It is also charter-compliant: a 3-bin ordinal δ-prior is a *coarse calibrated prior*, reported with
  uncertainty; binary is explicitly a labeled diagnostic.

## 3. A/B (unchanged structure)

Same held-out leave-datasets-out A/B, same pools/config, **two arms**: LOD/top-coding vs own-value
self-censoring (boundary). **Pre-registered:** LOD should learn **coarsely** out-of-family; own-value
should remain **flat**. **If own-value also learns, treat as leakage/artifact until proven
otherwise** — audit whether a feature encodes δ/rate directly (own-value at matched rate, single
column, is a *flat* idiom per P2.2b; coarse own-value learning would be suspicious). The δ→rate
leakage gate runs on both arms.

## 4. Features (frozen, existing — NO new features)

Use the **existing 17 marginal consequence features** (`consequence_features.compute_consequence_features`)
— the set the corrected gate showed transfers coarse δ (OOF AUC 0.734). **Do NOT** add features.
**Do NOT** use the rejected transfer-robust set. **No new representation work** in this step.

## 5. Model (main vs diagnostic — both stated)

- **MAIN run — features-only learned head.** A small head on the fixed 17-feature vector ONLY:
  `LayerNorm(17) → MLP → K bins` (reuse `DeltaBinHead` with evidence_dim=17; new thin
  `FeatureOnlyDeltaModel`). No encoder. This isolates the question — *can a learned, calibrated model
  recover the transferable coarse signal out-of-family?* — and is directly comparable to the LR
  baseline (should match/beat it). It is a legitimate minimal learned δ-prior on the observable
  features.
- **DIAGNOSTIC run — encoder + features (existing).** The existing target-conditioned
  `DeltaPriorModel`/`TargetConditionedDeltaModel` with consequence features, output bins coarsened.
  Tests **model integration**: if the features-only head learns coarse δ but this floors, the encoder
  evidence is swamping the features (the §6 integration concern), not a signal problem.
- **No encoder / tokenization redesign.** Coarsening = a different `num_bins` + a coarse label map
  derived from the continuous `db.delta` (no generator/tokenization change).

## 6. Metrics

- **Binary (step 1):** AUC and accuracy (δ=0 vs large-δ), vs base-rate; ECE; leakage gate.
- **3-bin (step 2, main):** **RPS (3-bin)**, accuracy, **adjacent accuracy**, **mean entropy**,
  P(δ=0), **ECE / interval coverage** (3-bin), vs **base-rate** AND vs the **LR-on-17-features**
  baseline (in-dist + OOF). Headline = beats uniform-RPS(3) and base-rate by ≥2 SE out-of-family.
- All runs: leakage gate (`leakage_pass`) + validated manifest (records the coarse scheme, K,
  feature set = the 17, family).

## 7. Decision rules (pre-registered, from approval)

- **LOD coarse learns OOF AND own-value stays flat** → the **detectability spectrum is demonstrated in
  the held-out learned channel at coarse resolution** (the milestone).
- **LR learns but the (encoder+features) neural model fails** → **model-integration** issue (encoder
  swamps features); the features-only head is then the carrier.
- **Neither LR nor the model learns** → revisit the corrected gate assumptions (re-examine the gate /
  data).
- **3-bin learns but 7-bin does not** → adopt the **curriculum** (coarse now, fine later) explicitly.
- **Binary learns but 3-bin fails** → the recoverable signal is **presence/absence of strong
  truncation**, not calibrated δ magnitude; report Lacuna's coarse claim honestly as such.
- **own-value learns** → leakage/artifact until proven otherwise (audit features for direct δ/rate
  encoding).

## 8. Module plan (new code for the approved cycle; small)

- `lacuna/survey/coarse_bins.py` — binary + 3-bin schemes: `assign_binary_bin(δ)`,
  `assign_coarse3_bin(δ, τ=1.0)`, edges/centers, `NUM_*`. (Mirrors `delta_bins`.)
- `lacuna/survey/feature_only_head.py` — `FeatureOnlyDeltaModel` (LayerNorm + `DeltaBinHead` on the
  17-vector) + factory; deterministic init from injected RNG.
- `train` seam — `TrainConfig` gains `coarse_scheme ∈ {none, binary, coarse3}` and a `model_kind ∈
  {features_only, encoder_features}`; the label is mapped from `db.delta` to the coarse bin;
  `num_bins`/`uniform_rps` follow; metrics' bin-centers generalized to the active scheme. No change to
  generators, tokenization, loss, or the leakage gate.
- Runner `scripts/run_p2p2c_coarse_ab.py` — binary diagnostic + 3-bin main A/B (LOD vs own-value),
  features-only main + encoder-features diagnostic, with the LR-on-features baseline reported
  alongside; manifests validated.
- Tests for every new module (Rule 7); ≤500 LOC each; RNG-injected determinism; fail-loud;
  leakage-gated.

## 9. Constraints

No new features; no transfer-robust set; no encoder/tokenization redesign; no loss change; same A/B
(LOD vs own-value); same held-out leave-datasets-out validation; same leakage gate + manifest;
binary only as a labeled diagnostic; freeze the coarse scheme before running. Include the
LR-on-features baseline (in-dist + OOF).

## 10. Stop condition

This is the audit. **No code until approved.** On approval: implement the coarse-bin scheme +
features-only head + the train seam + tests (suite green) → run the **binary diagnostic** + the
**3-bin A/B** (features-only main; encoder-features diagnostic) with the LR baseline → report against
the decision rules → stop.

## Open decisions for review

1. τ for 3-bin (propose 1.0) and the δ-grid → bin mapping (balance).
2. Main = features-only head (propose) vs encoder-features as main (I propose features-only as main,
   encoder-features as the integration diagnostic).
3. Same 4/2/2 leave-datasets-out split as the prior A/B (propose).
4. Pass margin: beats uniform-RPS(K) and base-rate by ≥2 SE OOF (propose).
