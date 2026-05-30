"""
lacuna.data.composition_blocks

Apply a JOINT / BLOCK missingness mechanism to a fixed set of columns, at a controllable rate
(ADR-0007, Stage B — the realism scaffolding).

Stage A found our per-column-independent synthetic missingness is trivially distinguishable from
real survey missingness (discriminator AUC 1.000): real masks have co-missingness (whole modules
go missing together), few dominant patterns, and monotone/dropout structure, all of which
per-column independence cannot produce. ADR-0007 re-admits the JOINT mechanisms to close that gap.
This module is the thin reuse layer that lets the composition sampler splice those joints in: it
applies one joint generator to EXACTLY a block's columns (via a sub-matrix view, so it can never
touch columns outside the block — keeping the by-cell tags unambiguous) and tunes it to a target
average per-column miss rate (so the by-cell composition stays controllable — ADR-0007 commitment 1).

Block kinds (each reuses an existing, registry-frozen generator — Coding Bible Rule 8, no rebuild):
  MNAR:
    - "refusal"   — MNARModuleRefusal: a battery refused by rows with high latent module value.
                    Sharp, all-or-nothing co-missingness (corr≈1, very few patterns).
    - "attrition" — MNARAttrition: progressive left-to-right dropout. Monotone staircase structure.
    - "latent"    — MNARLatentHealth: a latent factor drives the whole block. GRADED co-missingness
                    (corr≈0.2–0.5), more distinct patterns — the intermediate regime real data shows.
  MAR:
    - "skip"      — MARModuleSkip: a battery skipped by rows whose OBSERVED gate value is high. Same
                    sharp co-missing shape as refusal, but driven by observed data → MAR.

Rate control. `refusal`/`skip` realise an average per-column rate ≈ their `baseline` (one column is
held out as the demographic gate, so the block average is scaled by (w-1)/w). `attrition`/`latent`
pin a high internal rate, so they are ROW-GATED — the mechanism is applied to only an eligible
fraction `q` of rows (the rest stay fully observed in the block, as a non-panel subpopulation would)
— and `q` is set from the target rate. The realised rate of every kind is guarded by tests to land
near the requested target (the confound-control discipline shared with the column pools).

Determinism (Coding Bible Rule 6): all randomness flows through the injected RNGState. Fails loud
(Rule 1) on a block too narrow for the requested kind.
"""

from typing import Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import MAR, MNAR
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mnar.social import MNARModuleRefusal
from lacuna.generators.families.mnar.selection import MNARAttrition
from lacuna.generators.families.mnar.latent import MNARLatentHealth
from lacuna.generators.families.mar.survey import MARModuleSkip

MNAR_BLOCK_KINDS: Tuple[str, ...] = ("refusal", "attrition", "latent")
MAR_BLOCK_KINDS: Tuple[str, ...] = ("skip",)
_ALL_KINDS = MNAR_BLOCK_KINDS + MAR_BLOCK_KINDS

# Module width fraction for refusal/skip: keep ~1 column out of the battery as the always-observed
# demographic gate (the family caps the module at d-1 columns regardless; this names the intent).
_MODULE_FRAC = 0.9
# Empirical inflation of MARModuleSkip's realised rate over its baseline at gate_strength≈1.5
# (the gate-driven sigmoid over-deletes vs the nominal baseline). Calibrated in the Stage-B de-risk.
_SKIP_INFLATION = 1.2
# Mean per-column miss rate the (un-gated) attrition / latent mechanisms produce on a standard-normal
# block. Used to invert the eligible-row fraction q from a target rate. Calibrated in the de-risk.
_ATTRITION_PROFILE = 0.85
_LATENT_PROFILE = 0.5
_LATENT_STRENGTH = 2.5   # latent→missingness coupling; higher = stronger (but still graded) co-miss.


def block_class(kind: str) -> int:
    """Mechanism class id (MCAR/MAR/MNAR) a block kind stamps onto its co-deleted cells."""
    if kind in MNAR_BLOCK_KINDS:
        return MNAR
    if kind in MAR_BLOCK_KINDS:
        return MAR
    raise ValueError(f"unknown block kind {kind!r}; valid: {_ALL_KINDS}")


def _row_gate(r_block: torch.Tensor, q: float, rng: RNGState) -> torch.Tensor:
    """Restore (force fully observed) a random (1 - q) fraction of rows across the whole block.

    The eligible `q` fraction keep the mechanism's mask; the rest become a fully-observed
    subpopulation (a respondent never entered into the longitudinal panel / latent process). This
    is how the otherwise rate-pinned attrition / latent mechanisms are tuned to a target rate, and
    it preserves their structure (eligible rows stay monotone / co-missing; restored rows are
    trivially complete).
    """
    n = r_block.shape[0]
    eligible = rng.rand(n) < q
    out = torch.ones_like(r_block)
    out[eligible] = r_block[eligible]
    return out


def apply_block(
    kind: str,
    z_block: torch.Tensor,
    target_rate: float,
    rng: RNGState,
    *,
    strength: float = 1.5,
) -> torch.Tensor:
    """Apply a joint/block mechanism to a block's columns, tuned to a target avg per-column rate.

    Args:
        kind: one of `MNAR_BLOCK_KINDS + MAR_BLOCK_KINDS`.
        z_block: [n, w] z-scored predictor view of the block's columns (the sub-matrix the block
            owns). The mechanism only ever sees these columns, so it can only delete within the
            block — the cells it deletes are unambiguously this block's (and its class's).
        target_rate: desired average per-column missing fraction over the block (confound control).
        rng: explicit RNG (mechanism draws + any row-gating flow through spawned children).
        strength: value→missingness coupling strength (selection / gate / attrition value-dep).

    Returns:
        r_block: [n, w] boolean observed-mask for the block (True = observed).

    Raises:
        ValueError: on an unknown kind, a block too narrow for the kind, or target_rate not in (0,1).
    """
    if z_block.ndim != 2:
        raise ValueError(f"z_block must be 2-D [n, w], got ndim={z_block.ndim}")
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    n, w = z_block.shape

    if kind == "refusal":
        if w < 2:
            raise ValueError(f"'refusal' block needs w >= 2 (a battery + a gate), got w={w}")
        baseline = min(0.95, target_rate * w / (w - 1))
        gen = MNARModuleRefusal(0, "comp_mnar_refusal", GeneratorParams(
            module_frac=_MODULE_FRAC, baseline_refusal=baseline,
            selection_strength=strength, direction="high"))
        return gen.apply_to(z_block, rng.spawn())

    if kind == "skip":
        if w < 2:
            raise ValueError(f"'skip' block needs w >= 2 (a battery + a gate), got w={w}")
        baseline = min(0.95, target_rate * w / ((w - 1) * _SKIP_INFLATION))
        gen = MARModuleSkip(0, "comp_mar_skip", GeneratorParams(
            module_frac=_MODULE_FRAC, baseline_skip=baseline, gate_strength=strength))
        return gen.apply_to(z_block, rng.spawn())

    if kind == "attrition":
        gen = MNARAttrition(0, "comp_mnar_attrition", GeneratorParams(
            attrition_rate=1.0, value_dependence=strength * 0.5))
        r_full = gen.apply_to(z_block, rng.spawn())
        q = min(0.99, max(0.02, target_rate / _ATTRITION_PROFILE))
        return _row_gate(r_full, q, rng.spawn())

    if kind == "latent":
        gen = MNARLatentHealth(0, "comp_mnar_latent", GeneratorParams(
            latent_strength=_LATENT_STRENGTH, obs_strength=1.0))
        r_full = gen.apply_to(z_block, rng.spawn())
        q = min(0.99, max(0.02, target_rate / _LATENT_PROFILE))
        return _row_gate(r_full, q, rng.spawn())

    raise ValueError(f"unknown block kind {kind!r}; valid: {_ALL_KINDS}")
