# Lacuna — Changelog

Append-only ledger of substantive changes, so drift is visible in history. Governed by
`docs/NORTH-STAR.md` (charter) and `docs/PROPOSAL-survey-rewrite.md` (engineering plan). Newest first.
Format loosely follows Keep-a-Changelog. Each entry: date, type, what, why, and the charter/proposal
clause it serves.

---

## 2026-06-02 — Re-scope groundwork (docs only; no code changed)

### Added
- `docs/NORTH-STAR.md` — charter. Re-scopes Lacuna from a 3-class missingness classifier to a
  governance tool emitting a calibrated prior over a sensitivity parameter δ on a survey-specific
  manifold, with abstention. Pins: identification line (Molenberghs), reframed estimand, manifold
  hypothesis (M1/M2/M3), scope=survey-only, human-parity law (§4.8), estimand-at-reporting-layer
  (§3¾), guardrails, success metrics, coverage boundary, falsification condition.
- `docs/v1.0-generator-mechanism.md` — how the v1.0 generators fundamentally work (the
  `apply_to(X,rng)->R` contract, z-scored predictor view, MCAR/MAR/MNAR decision rules, registry).
- `docs/v1.0-realism-critique.md` — first-principles critique of the abandoned "mixed" hole-punching.
- `docs/PROPOSAL-survey-rewrite.md` — engineering governing doc: target architecture (wishlist),
  as-is codebase audit (5-subsystem salvage map), delta, and rewrite-scope verdict.

### Verified (not changed)
- v1.0 reproduction: real full MoE (`LacunaModel`, 900,747 params) via
  `scripts/train.py --config configs/training/survey.yaml` → best val_acc 0.9163 @ epoch 12
  (≈ 0.9175 target). Tag `v1.0-canonical` (2935cc8) confirmed correct. Suite green (1080 pass / 1 skip).

### Decided
- Manifold operationalization: HYBRID (enumerate survey idioms → generate → validate footprint
  coverage against real survey masks).
- Scope: survey-only (Lacuna-Survey); sibling regimes are future instruments.
- δ richness: B2 (canonical scalar δ) first, B1 (rich mechanism posterior) as documented escalation.
- Rewrite scope: gated by the codebase audit (this entry's PROPOSAL), not assumed up front.
- **First increment scoped to the own-value self-censoring axis only** (δ ≡ β₂, where scalar δ is
  exact), least-invasive-first. Attrition / unit-nonresponse / selection idioms explicitly deferred
  to abstention, with a documented **revision trigger**: if the self-censoring δ-prior passes the
  feasibility gate + out-of-family calibration, revise to a per-idiom multi-axis manifold covering
  them; if it fails where δ is exact, stop. (PROPOSAL §5.1 DECISION, §6 P0/P1.)
- Sequencing: a feasibility gate precedes any build — but it must be DECISIVE, not merely cheap.
  **Methodological invariant added (North Star §4.9):** no proxy weaker than the deployment model may
  license a negative/kill conclusion (a weak-proxy negative is confounded — the arc's exact error).
  Corrected P1 accordingly: the gate brackets the truth with (a) an information-ceiling Bayes oracle
  computed from the known generative model and (b) the real model RETRAINED FROM SCRATCH
  (frozen-encoder + stapled-head BANNED). A prior framing that sold "zero model training" as a virtue
  was wrong and is retracted.

### Added (P1 spec)
- `docs/PROPOSAL-P1-implementation-audit.md` — implementation audit for the P1 self-censoring
  feasibility probe (SPEC ONLY; no code, nothing run). Covers all 11 required items: generator path,
  δ formula (δ≡β₂), β₀ rate-matching solver, β₁/δ/rate/ρ/n sweep grid, pluggable X-model interface,
  analytic Bayes-oracle calculation (LLR + KL/Chernoff + n-sample error), model target/loss (current
  LacunaModel re-objectived to a binary δ head), **all layers trainable: YES**, **checkpoint loaded:
  NO**, expected runtime (oracle <10 min CPU; model arm ~1.5–4 h GPU), and saved artifacts/manifest.
- Decisions recorded: δ-axis approved (controlled scalar mechanism-family probe); oracle X-model
  hierarchy (exact synthetic ceiling + fitted-X-model real, pluggable, assumption never hidden);
  model arm = current architecture retrained from scratch (no frozen encoder / no stapled head / no
  checkpoint), final gate to be rerun on the final δ/composition head.
- Vocabulary locked: ceiling = analytic Bayes oracle (under stated X-model) ONLY; RF/MLP/handcrafted
  = baselines; frozen-head = transfer ablations; weak-proxy negatives are never impossibility evidence.

### Known debt surfaced by the audit (pre-existing, independent of re-scope)
- `lacuna/training/loss.py` (956 LOC) and `lacuna/training/checkpoint.py` (649 LOC) breach the
  500-LOC hard limit (CLAUDE.md Rule 4). Any rework must split, not extend.
- `lacuna/data/tokenization.py::tokenize_dataset` subsamples via global unseeded `np.random`
  (flagged `# NON-DETERMINISTIC`) — violates Rule 6; route through injected `RNGState` when reworked.
- `lacuna/core`, `lacuna/experiments`: no test files (Rule 7 gap).
