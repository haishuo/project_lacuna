"""
lacuna.data.real_mask_sampler

Sample analyst-realistic missingness blocks from a large real survey table (ADR-0007, Stage A).

Raw pooled survey tables (e.g. NHANES merged across cycles) are dominated by *structural* absence
— a variable measured only in some cycles is NaN everywhere else — so their overall missingness
(~80-94%) is not what an analyst actually faces. An analyst works with a coherent block: a handful
of related columns over a subpopulation, carrying genuine item / module nonresponse. This sampler
draws such blocks so their *observable footprint* (`missingness_footprint`) reflects realistic
survey missingness, giving a real-side corpus to calibrate the synthetic generators against.

A block is a random contiguous column window × random row subset (contiguity ≈ topical/module
coherence in merged survey tables → realistic co-missingness structure), trimmed of all-missing
columns and all-missing rows, then KEPT only if it lands in an analyst-plausible regime:
`min_cols ≤ d ≤ max_cols`, `min_rows ≤ n`, and overall missingness in `[miss_lo, miss_hi]` with at
least some observed AND some missing. The missingness band is a documented definitional choice about
what "realistic" means (mirrors the curation of the labelled anchors); structural near-voids and
fully-observed blocks are both excluded.

Determinism (Coding Bible Rule 6): all randomness flows through an injected `numpy.random.Generator`.

Contract:
    sample_mask_blocks(values, rng, *, n_blocks, ...) -> list[(x, r)]
where `values` is a [N, D] float array with NaN = missing; each returned `x` is the block's values
(NaN preserved) and `r` its boolean observed-mask. Returns UP TO `n_blocks` (fewer if the table is
too small/clean to yield them — the caller checks the count). Fails loud (Rule 1) on malformed input.
"""

from typing import List, Tuple

import numpy as np


def sample_mask_blocks(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    n_blocks: int,
    min_cols: int = 4,
    max_cols: int = 30,
    min_rows: int = 100,
    max_rows: int = 5000,
    miss_lo: float = 0.01,
    miss_hi: float = 0.6,
    max_tries_per_block: int = 40,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Draw up to `n_blocks` analyst-realistic (x, r) missingness blocks from `values`.

    See module docstring. `values` may be any dtype coercible to float (NaN marks missing).

    Raises:
        ValueError: on non-2D input, non-positive n_blocks, or incoherent bounds.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D [N, D], got ndim={values.ndim}")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    if not (0 < min_cols <= max_cols) or not (0 < min_rows <= max_rows):
        raise ValueError("require 0 < min_cols <= max_cols and 0 < min_rows <= max_rows")
    if not (0.0 <= miss_lo < miss_hi <= 1.0):
        raise ValueError(f"require 0 <= miss_lo < miss_hi <= 1, got [{miss_lo}, {miss_hi}]")

    N, D = values.shape
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    if D < min_cols or N < min_rows:
        return out  # table too small to yield a valid block

    hi_cols = min(max_cols, D)
    hi_rows = min(max_rows, N)
    for _ in range(n_blocks):
        for _try in range(max_tries_per_block):
            w = int(rng.integers(min_cols, hi_cols + 1))
            c0 = int(rng.integers(0, D - w + 1))
            nr = int(rng.integers(min_rows, hi_rows + 1))
            rows = rng.choice(N, size=nr, replace=False)
            block = values[np.ix_(rows, np.arange(c0, c0 + w))]

            # An analyst would drop entirely-empty columns/rows from the working block.
            block = block[:, ~np.isnan(block).all(axis=0)]
            if block.shape[1] < min_cols:
                continue
            block = block[~np.isnan(block).all(axis=1)]
            if block.shape[0] < min_rows:
                continue

            m = float(np.isnan(block).mean())
            if not (miss_lo <= m <= miss_hi) or m == 0.0 or m == 1.0:
                continue

            out.append((block, ~np.isnan(block)))
            break
    return out
