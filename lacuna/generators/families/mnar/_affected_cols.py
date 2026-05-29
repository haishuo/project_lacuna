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

Contract:
    resolve_affected_cols(params, d, rng) -> 1-D LongTensor of column indices to affect.
Fails loud (Rule 1) on a structurally invalid request (d < 1).

Determinism (Rule 6): all randomness flows through the injected RNGState; the targeted mode is
fully deterministic given d.
"""

import numpy as np

from lacuna.core.rng import RNGState
from lacuna.generators.params import GeneratorParams


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
        target = params["target_col_idx"]
        if not isinstance(target, (int, np.integer)) or isinstance(target, bool):
            raise ValueError(
                f"target_col_idx must be an int, got {type(target).__name__}: {target!r}"
            )
        target = int(target)
        if target < 0:
            target = d + target
        target = target % d
        return np.array([target], dtype=np.int64)

    affected_frac = params.get("affected_frac", 0.5)
    n_affected = max(1, int(d * affected_frac))
    # Legacy behaviour, unchanged: random fraction of columns via the injected RNG.
    return rng.choice(d, size=n_affected, replace=False)
