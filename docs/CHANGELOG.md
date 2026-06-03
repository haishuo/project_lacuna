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

### Known debt surfaced by the audit (pre-existing, independent of re-scope)
- `lacuna/training/loss.py` (956 LOC) and `lacuna/training/checkpoint.py` (649 LOC) breach the
  500-LOC hard limit (CLAUDE.md Rule 4). Any rework must split, not extend.
- `lacuna/data/tokenization.py::tokenize_dataset` subsamples via global unseeded `np.random`
  (flagged `# NON-DETERMINISTIC`) — violates Rule 6; route through injected `RNGState` when reworked.
- `lacuna/core`, `lacuna/experiments`: no test files (Rule 7 gap).
