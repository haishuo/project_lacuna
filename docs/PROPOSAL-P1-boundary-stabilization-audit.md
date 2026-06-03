# P1 Boundary Stabilization — Implementation Audit (for approval BEFORE running)

*Governed by `docs/NORTH-STAR.md` and the approved scope. **Purpose:** determine whether the high-ρ
boundary gap can be reduced **reliably** using **training-procedure changes only** — no architecture,
data-regime, generator, or oracle-parameter changes. SPEC ONLY — no code written, nothing run, until
this audit is approved.*

**Scoped statement of the problem (the carefully-phrased conclusion this experiment addresses):** the
boundary regime is learnable by the current architecture, but optimization is unstable — one seed
approaches the profiled ceiling while two seeds fall into a chance basin.

## 1. Exact boundary regime parameters (unchanged)
Synthetic-X `ConditionalGaussian.synthetic(ρ=0.9)`, 2 columns, rate 0.1, β₁=1.0, n=512.
- **H1 (MNAR):** β₀ = −3.344, β₁ = 1.0, β₂ = 1.0
- **H0 (profiled best-fit MAR):** β₀′ = −3.274, β₁′ = 1.864, β₂ = 0.0
- **Profiled Bayes ceiling reference:** 0.251 ± 0.010 (held fixed; NOT recomputed).

## 2. Current baseline boundary results (to beat)
- Pilot (1 seed): test error 0.473, gap +0.222, early-stopped epoch 15.
- Multi-seed (seeds 42/43/44): **0.309 / 0.480 / 0.458**, mean 0.416 ± 0.076, gaps +0.058/+0.229/+0.207.
- **Chance basin ≈ 0.47–0.48; escape on 1 of 3 restarts** (0.309). Recipe: Adam lr 3e-4, no warmup, no
  decay, patience 10, max 40 epochs.

## 3. Proposed optimizer / schedule changes (training-procedure ONLY)
Same model (full LacunaModel, max_cols=2, fresh init, all trainable), same data regime, same loss
(MCAR-safe binary). Changes target the early-training chance-basin trap:
- **Lower initial LR:** 3e-4 → **1e-4** (instability is evident; smaller early steps avoid the collapse).
- **LR warmup:** linear warmup over **300 steps** from 0 → 1e-4.
- **Decay:** **cosine** decay 1e-4 → 1e-5 over the max schedule. (Plateau/ReduceLROnPlateau is the
  documented alternative if cosine underperforms — but we run ONE recipe this pass, no per-run tuning.)
- **Longer patience:** 10 → **20**; **max_epochs 40 → 80** (room for warmup+cosine).
- Unchanged: Adam, grad-clip 1.0, batch 32, counts 8000/2000/4000, the binary head/loss.
This is a single fixed recipe applied to all restarts — no recipe search against the data.

## 4. Number of restarts / seeds
**10 restarts**, fresh seeds **{100–109}** (distinct from 42/43/44 to avoid any coupling). Each restart
= fresh model init + fresh **train+val** draw. A **single fresh held-out TEST set** (seed block 9000,
distinct from all prior runs) is shared across restarts for comparable final eval. (Synthetic data ⇒
"fresh held-out test" is exact; no overlap with the multi-seed test.)

## 5. Early-stopping / patience rule
Monitor **validation binary error** each epoch; stop when it does not improve (> 1e-4) for **20**
consecutive epochs, or at 80 epochs. Per restart, retain the **best-val** weights (in-memory restore —
NOT a loaded checkpoint).

## 6. Validation-selection rule
The reported per-restart model is selected by **minimum validation error** (ties broken by validation
loss), **never** by test performance. The shared test set is touched **once** per restart, for the
final number only. The epoch selected is recorded.

## 7. Expected runtime
Boundary is n=512; ~0.03–0.05 s/step, 250 steps/epoch; with patience 20 / max 80, ~15–50 epochs per
restart ⇒ **~3–8 min/restart ⇒ ~40–80 min for 10 restarts** on GPU. Data-gen is vectorized (~0.2 s).

## 8. Manifest fields (required before any metric is read)
`arm="model"`, `kind="main"`, `checkpoint_loaded=false`, `all_layers_trainable=true`,
`trainable_param_count`, `split_scheme` (per-restart fresh train/val seed blocks + shared fresh test
seed block 9000), `grid` (boundary params + restart seeds + **the full optimizer/schedule config**),
`xmodel`, `beta0_solver="FIXED recorded oracle params"`, `wall_clock_seconds`, `metrics`, `calibration`.
`validate_manifest` enforces no-checkpoint + all-trainable (charter §4.9). Metrics not interpreted
unless the manifest validates.

## 9. Metrics to report
Per restart: val-selected **test error** ± binomial SE, **best-val error**, epoch selected, **gap**
(= test error − 0.251) + CI, escaped-basin flag, ECE. Aggregate over 10 restarts:
- **mean ± std** test error, **median** test error, **best (val-selected) test error**,
- **fraction of restarts that escape the chance basin** (escape := test error < 0.40),
- gap mean/median,
- leakage guards: any test error < 0.251 − 2·SE flagged as **suspicious (leakage, not victory)**;
  manifest validity; rate-match (reported per class).
Every restart is reported, not just the best.

## 10. Success criterion (operationalized from your statement)
> Stabilization succeeds if a **majority of restarts escape the chance basin** AND the
> **validation-selected test error moves meaningfully toward the profiled ceiling**, with **no negative
> gaps or leakage signals**.

Operational thresholds (declared up front, not post-hoc):
- **Escape majority:** ≥ **6/10** restarts with test error < 0.40.
- **Toward ceiling:** **median** val-selected test error ≤ **0.32** (vs baseline mean 0.416; ceiling
  0.251) — i.e. a meaningful move, not necessarily reaching the ceiling.
- **Clean:** no restart with test error < 0.251 − 2·SE (no leakage); manifest valid; rates matched.
- **Partial success** (report honestly, do not overclaim): escape fraction improves over the 1/3
  baseline but falls short of 6/10, or median improves but stays > 0.32 — would motivate a documented
  second procedure variant (e.g. plateau decay / more restarts), still procedure-only.

## Non-goals (explicit)
No architecture change; no richer MAR-null profiling; no real-X; no new baselines; no v1.0 transfer
ablation; no generator/oracle parameter changes; no test-based selection or repeated tuning against a
test set.

## Approval gate
On approval I will implement `scripts/run_feasibility_boundary_stabilization.py` (reusing
`train_regime` with an added warmup+cosine LR schedule and the restart loop) + a small scheduler unit
test, get the suite green, run the 10-restart boundary stabilization on GPU, and report the per-restart
table + aggregate + success-criterion evaluation. **Nothing runs until this audit is approved.**
