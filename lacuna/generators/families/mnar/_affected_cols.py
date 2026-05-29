"""
lacuna.generators.families.mnar._affected_cols

Resolve which columns a per-column MNAR censoring/detection generator should affect.

The threshold (`censoring.py`) and detection-limit (`detection.py`) MNAR families apply a
per-column rule (e.g. "values above the 70th percentile go missing") to a SET of affected
columns. Historically that set was always a random `affected_frac` draw. For the column-level
missingness experiment (ADR-0006) we also need to target ONE specified column, exactly as the
self-censoring family already supports via `target_col_idx` — so that diverse MNAR subtypes
(threshold, detection limit) can be spliced into per-column mechanism mixtures, not just the
logistic self-censoring family.

This helper centralises that resolution so all 12 threshold/detection generators share one
implementation rather than duplicating it (Coding Bible Rule 3). Two modes:

  - `target_col_idx` ABSENT (default): preserve the legacy behaviour exactly — a random
    `affected_frac` fraction of columns, chosen via `rng.choice` (no behaviour change for the
    registry or existing tests).
  - `target_col_idx` PRESENT: affect EXACTLY that single column. Negative indices wrap from the
    end (`-1` = last column), matching the convention in `self_censoring.MNARLogistic`. Out-of-range
    positive indices are taken modulo `d` (same forgiving convention as the logistic family) so a
    caller composing mixtures over variable-width datasets cannot crash the generator.

Two entry points share the same target-resolution rule:

  - `resolve_affected_cols(params, d, rng)` -> 1-D int array of affected column indices. Used by
    families whose missingness is decided per-affected-column (threshold/detection/self-censoring/
    social/strategic/...): they iterate the returned indices.
  - `restrict_to_target_col(R, params, d)` -> mask R with non-target columns forced observed when
    `target_col_idx` is set, else R unchanged. Used by WHOLE-MATRIX families (latent confounding,
    progressive attrition) that compute a joint mask first and cannot enumerate affected columns
    up front. Applying it AFTER the joint computation keeps the default path byte-for-byte identical
    (it is a literal no-op when `target_col_idx` is absent) while still letting such a mechanism be
    spliced into a per-column mixture, targeting one column.

Contract: fails loud (Rule 1) on a structurally invalid request (d < 1) or a non-int target.

Determinism (Rule 6): all randomness flows through the injected RNGState; the targeted mode is
fully deterministic given d.
"""

import numpy as np
import torch

from lacuna.core.rng import RNGState
from lacuna.generators.params import GeneratorParams


def _resolve_single_target(params: GeneratorParams, d: int) -> int:
    """Resolve `target_col_idx` to a concrete 0 <= idx < d. Shared by both entry points.

    Negative indices wrap from the end; out-of-range positives wrap modulo d (matching
    `self_censoring.MNARLogistic`). Raises on a non-int (bool included — a True index is a bug).
    """
    target = params["target_col_idx"]
    if not isinstance(target, (int, np.integer)) or isinstance(target, bool):
        raise ValueError(
            f"target_col_idx must be an int, got {type(target).__name__}: {target!r}"
        )
    target = int(target)
    if target < 0:
        target = d + target
    return target % d


def resolve_affected_cols(params: GeneratorParams, d: int, rng: RNGState) -> np.ndarray:
    """Return the column indices a per-column MNAR generator should affect.

    See module docstring for the two modes. `affected_frac` (default 0.5) is only consulted in
    the legacy random mode; it is ignored when `target_col_idx` is set (targeting is exactly one
    column by construction). Returns a 1-D numpy int array in BOTH modes so callers can iterate
    (`for col in cols`) or index (`cols[i]`) identically to the legacy `rng.choice` result.
    """
    if d < 1:
        raise ValueError(f"resolve_affected_cols requires d >= 1, got d={d}")

    if "target_col_idx" in params:
        return np.array([_resolve_single_target(params, d)], dtype=np.int64)

    affected_frac = params.get("affected_frac", 0.5)
    n_affected = max(1, int(d * affected_frac))
    # Legacy behaviour, unchanged: random fraction of columns via the injected RNG.
    return rng.choice(d, size=n_affected, replace=False)


def restrict_to_target_col(R: torch.Tensor, params: GeneratorParams, d: int) -> torch.Tensor:
    """For whole-matrix MNAR families: keep missingness only in `target_col_idx` (others observed).

    No-op when `target_col_idx` is absent — the default whole-matrix mask is returned unchanged,
    so this cannot alter legacy behaviour. When set, every column except the resolved target is
    forced fully observed; the target column keeps its computed mask. Determinism is preserved:
    the joint mask was already drawn from the injected RNG; this only selects a column from it.
    """
    if "target_col_idx" not in params:
        return R
    if d < 1:
        raise ValueError(f"restrict_to_target_col requires d >= 1, got d={d}")
    target = _resolve_single_target(params, d)
    out = torch.ones_like(R)
    out[:, target] = R[:, target]
    return out
