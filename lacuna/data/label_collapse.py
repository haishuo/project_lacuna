"""
lacuna.data.label_collapse

Collapse a heterogeneous (mixed-mechanism) dataset's per-column ANSWER SHEET into a single
dataset-level MCAR/MAR/MNAR label for the single-label MoE, plus the full by-cell composition vector.

The improved generators punch holes heterogeneously across columns; we know the true mechanism of every
hole because we created it (`compose_mixed_missingness` returns `column_classes` + `per_column_miss_rate`).
This module turns that answer sheet into:
  - composition_vector: [p_MCAR, p_MAR, p_MNAR] = fraction of MISSING CELLS from each mechanism
    (the ADR-0007 by-cell denominator; OBSERVED columns contribute no missing cells);
  - a collapsed dataset-level label under one of three configurable rules:
      * "dominant"   : argmax of the by-cell composition.
      * "primary"    : the generator's intended primary mechanism (passed in); secondaries are
                       realism/noise and ignored for the label.
      * "thresholded": MNAR if p_MNAR >= mnar_threshold, else argmax over {MCAR, MAR}.

Both the collapsed label and the full composition are returned/saved so the MoE can train single-label
while we retain the answer sheet for analysis (per the experiment spec).

Determinism (Coding Bible Rule 6): pure, no RNG. Fails loud (Rule 1) on a dataset with no missing cells
(the label is undefined) or a bad rule / out-of-range threshold — never a silent default.
"""

from typing import Optional, Tuple

import numpy as np

from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.mixed_missingness import OBSERVED

N_MECH = 3
COLLAPSE_RULES = ("dominant", "primary", "thresholded")


def composition_vector(column_classes: Tuple[int, ...], per_column_miss_rate: Tuple[float, ...],
                       n_rows: int) -> np.ndarray:
    """By-cell composition [p_MCAR, p_MAR, p_MNAR] from the per-column answer sheet.

    Args:
        column_classes: per-column mechanism (OBSERVED / MCAR / MAR / MNAR), length d.
        per_column_miss_rate: realised missing fraction per column, length d.
        n_rows: number of rows (to turn rates into missing-cell counts).

    Raises:
        ValueError: on length mismatch, n_rows <= 0, or zero total missing cells (label undefined).
    """
    if len(column_classes) != len(per_column_miss_rate):
        raise ValueError(f"column_classes ({len(column_classes)}) and per_column_miss_rate "
                         f"({len(per_column_miss_rate)}) must have equal length")
    if n_rows <= 0:
        raise ValueError(f"n_rows must be > 0, got {n_rows}")
    counts = np.zeros(N_MECH, dtype=float)
    for cls, rate in zip(column_classes, per_column_miss_rate):
        if cls == OBSERVED:
            continue
        if cls not in (MCAR, MAR, MNAR):
            raise ValueError(f"invalid column class {cls!r}; expected OBSERVED/MCAR/MAR/MNAR")
        counts[cls] += float(rate) * n_rows
    total = counts.sum()
    if total <= 0:
        raise ValueError("dataset has no missing cells; dataset-level label is undefined")
    return counts / total


def collapse_label(composition: np.ndarray, rule: str = "dominant", *,
                   primary: Optional[int] = None, mnar_threshold: float = 0.5) -> int:
    """Collapse a by-cell composition [3] to a single MCAR/MAR/MNAR label under `rule`.

    Raises:
        ValueError: on an unknown rule, a bad composition, a missing `primary` for rule="primary",
            or an out-of-range `mnar_threshold`.
    """
    c = np.asarray(composition, dtype=float)
    if c.shape != (N_MECH,) or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError(f"composition must be a finite non-negative [3] vector, got {c}")
    if rule not in COLLAPSE_RULES:
        raise ValueError(f"unknown collapse rule {rule!r}; valid: {COLLAPSE_RULES}")
    if rule == "dominant":
        return int(np.argmax(c))
    if rule == "primary":
        if primary not in (MCAR, MAR, MNAR):
            raise ValueError(f"rule='primary' requires primary in (MCAR/MAR/MNAR), got {primary!r}")
        return int(primary)
    # thresholded
    if not 0.0 < mnar_threshold <= 1.0:
        raise ValueError(f"mnar_threshold must be in (0, 1], got {mnar_threshold}")
    if c[MNAR] >= mnar_threshold:
        return MNAR
    return MCAR if c[MCAR] >= c[MAR] else MAR
