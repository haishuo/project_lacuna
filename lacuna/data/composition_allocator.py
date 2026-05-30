"""
lacuna.data.composition_allocator

Plan how to realise a target by-cell missingness COMPOSITION on a complete dataset (ADR-0007, Stage B).

This is the pure-planning half of the composition-controlled sampler: given a dataset shape and a
drawn `CompositionTarget`, it produces an `AllocationPlan` — a list of mechanism UNITS (per-column
or block), each owning a disjoint set of columns and a target per-column rate — whose EXPECTED
by-cell composition matches the target. `composition_sampler` then applies the plan and measures the
realised composition. Separating planning from application (Coding Bible Rule 3) keeps this module a
pure, deterministic function that can be checked in isolation: does the plan's expected composition
track the target?

The control principle (why this hits the by-cell target — ADR-0007 commitment 1)
--------------------------------------------------------------------------------
The composition is a RATIO (cells of class c / total missing cells), and the by-cell budget for
class c is `cells_c = f_c · miss_rate · n · d`. We enforce that budget DIRECTLY in two steps:
  1. partition the working columns across classes by the target fractions (largest-remainder, so a
     class with fraction f_c gets ≈ f_c of the working columns); then
  2. draw every unit's rate from one Beta(mean=μ, ν) law — ν tuned so the spread matches real survey
     data (Stage-A Band 2: per-column-rate sd≈0.33, ~40% heavily-missing columns) — and then
     multiplicatively RESCALE each class's rates so its rate-weighted column sum equals exactly
     `f_c · miss_rate · d` (= cells_c / n).
Step 2's rescale makes the expected composition equal the target up to rate-clipping (and pins the
overall miss rate to `miss_rate`), while multiplicative scaling preserves the per-column spread, so
Band-2 variation survives as a *free* by-product rather than a hidden confound — the Stage-5 lesson
(one shared denominator) made constructive. (Without the rescale, a 13–25-column dataset has too few
per-class columns for the rate spread to average out, and the composition wanders — measured, not
assumed: the rescale cut the plan's expected-composition L1 from ≈0.31 to ≈0.02.)

Class structure (the honest seam — ADR-0007 commitment 7)
---------------------------------------------------------
  - MCAR  — per-column Bernoulli only (the RANDOM anchor; the identifiable "structured-vs-random"
            axis rests on MCAR staying unstructured).
  - MAR   — per-column (clean-predictor pool) + skip-logic BLOCKS (a battery skipped on an observed
            gate; missing-by-design co-missingness).
  - MNAR  — per-column (own-value pool) + refusal / attrition / latent BLOCKS (value/dropout/latent
            co-missingness).
Both MAR and MNAR carry block structure, so blockiness does NOT imply MNAR — the MAR-vs-MNAR split
stays the genuinely hard (non-identifiable) part, while MCAR-vs-structured stays identifiable.

Determinism (Coding Bible Rule 6): every draw flows through the injected RNGState. Fails loud
(Rule 1) on a dataset too small to host a composition.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.composition_target import CompositionTarget
from lacuna.data.composition_blocks import MNAR_BLOCK_KINDS, MAR_BLOCK_KINDS

_MIN_BLOCK_WIDTH = 3   # narrower than this is just a per-column unit, not a "block"
# Per-column rate caps. The diverse MAR/MNAR column pools are tuned for confound-control around
# ~0.25 and a few subtypes break above ~0.7 (e.g. a column-specific threshold's percentile goes
# negative), so per-column MAR/MNAR units are capped pool-safe. MCAR (plain Bernoulli) and the block
# mechanisms (module batteries — exactly where heavy missingness realistically lives) tolerate the
# full band, so the heavily-missing budget is routed to them by the water-fill.
_POOL_SAFE_HI = 0.60


@dataclass(frozen=True)
class Unit:
    """One mechanism instance owning a disjoint set of columns.

    Attributes:
        cls: mechanism class id (MCAR/MAR/MNAR) every cell this unit deletes is tagged with.
        kind: "column" (a single per-column mechanism) or a block kind from `composition_blocks`.
        cols: the column indices this unit owns (length 1 for "column"; >= `_MIN_BLOCK_WIDTH` for a block).
        rate: target average per-column missing fraction for this unit (the same rate for every
            column of a block — a battery's items share a response propensity).
        predictor: for a per-column MAR unit, the clean (always-observed) predictor column; None otherwise.
    """
    cls: int
    kind: str
    cols: Tuple[int, ...]
    rate: float
    predictor: Optional[int]


@dataclass(frozen=True)
class AllocationPlan:
    """A full plan for one composed dataset (consumed by `composition_sampler`)."""
    units: Tuple[Unit, ...]
    observed_cols: Tuple[int, ...]
    n: int
    d: int
    target: CompositionTarget
    expected_composition: Tuple[float, float, float]  # by-cell, from the drawn unit rates
    expected_miss_rate: float


def _largest_remainder(fractions: Tuple[float, ...], total: int) -> Tuple[int, ...]:
    """Apportion `total` integer columns across classes by `fractions` (sums to `total` exactly)."""
    raw = [f * total for f in fractions]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    # Hand the leftover columns to the classes with the largest fractional parts.
    order = sorted(range(len(fractions)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return tuple(floors)


_RATE_LO, _RATE_HI = 0.01, 0.95   # per-column rate band accepted by every column-pool generator


def _draw_rate(mu: float, nu: float, rng: RNGState) -> float:
    """Draw one per-column rate from Beta(mu*nu, (1-mu)*nu), clipped to a generator-safe band.

    Mean = mu (sets the overall miss level); nu (small) controls the spread — nu≈1.2 reproduces the
    real survey per-column-rate sd (~0.33) and the ~40%-heavily-missing tail (Stage-A Band 2).
    """
    a = max(mu * nu, 1e-3)
    b = max((1.0 - mu) * nu, 1e-3)
    return float(min(_RATE_HI, max(_RATE_LO, rng.numpy_rng.beta(a, b))))


def _scale_to_budget(rates: List[float], widths: List[int], his: List[float],
                     target_rate_cells: float) -> List[float]:
    """Scale `rates` so their width-weighted sum equals `target_rate_cells`, within per-unit caps.

    Iterative water-fill: multiplicatively scale all rates, clip each to [_RATE_LO, his[i]], and
    repeat — units that saturate at their cap shed their excess budget onto the unsaturated ones on
    the next pass (so a high budget flows to the high-cap block units). A few passes recover the
    budget that a single clipped scaling would lose (the cause of the composition tail at high miss
    rates), while preserving the relative spread (Band 2). If every unit saturates below the target
    (an infeasibly high budget for the available headroom), it converges to the cap — an honest
    undershoot, reflected in the plan's expected_composition rather than silently masked.
    """
    out = [min(h, max(_RATE_LO, r)) for r, h in zip(rates, his)]
    for _ in range(8):
        cur = sum(r * w for r, w in zip(out, widths))
        if cur <= 0.0:
            break
        scale = target_rate_cells / cur
        nxt = [min(h, max(_RATE_LO, r * scale)) for r, h in zip(out, his)]
        if nxt == out:
            break
        out = nxt
        if abs(sum(r * w for r, w in zip(out, widths)) - target_rate_cells) < 1e-6:
            break
    return out


def _emit_units(
    cols: List[int],
    cls: int,
    block_kinds: Tuple[str, ...],
    predictor: Optional[int],
    mu: float,
    nu: float,
    target_rate_cells: float,
    block_rate_share: float,
    max_block_width: int,
    rng: RNGState,
) -> List[Unit]:
    """Emit a class's units over its columns, rescaled to hit the class's exact by-cell budget.

    `block_kinds` empty => per-column only (MCAR, the random anchor). Otherwise ~`block_rate_share`
    of the columns are packed into blocks (each >= `_MIN_BLOCK_WIDTH`, width drawn up to
    `max_block_width`), the remainder become per-column units. Raw rates are drawn from Beta(mu, nu)
    for Band-2 spread, then multiplicatively rescaled so the unit rate-weighted column sum equals
    `target_rate_cells` (= the class's cells_c / n) — pinning the by-cell composition while
    preserving the spread.
    """
    k = len(cols)
    specs: List[Tuple[str, Tuple[int, ...], Optional[int]]] = []
    i = 0
    if block_kinds and k >= _MIN_BLOCK_WIDTH:
        n_block_cols = int(round(block_rate_share * k))
        while (n_block_cols - i) >= _MIN_BLOCK_WIDTH:
            hi = min(max_block_width, n_block_cols - i)
            w = _MIN_BLOCK_WIDTH if hi <= _MIN_BLOCK_WIDTH \
                else int(rng.randint(_MIN_BLOCK_WIDTH, hi + 1, (1,)).item())
            kind = block_kinds[int(rng.randint(0, len(block_kinds), (1,)).item())]
            specs.append((kind, tuple(cols[i:i + w]), None))
            i += w
    for c in cols[i:]:
        specs.append(("column", (c,), predictor))

    raw = [_draw_rate(mu, nu, rng) for _ in specs]
    widths = [len(s[1]) for s in specs]
    # Block units (and per-column MCAR) take the full rate band; per-column MAR/MNAR units are
    # capped pool-safe so the diverse column pools never receive an out-of-range rate.
    his = [_RATE_HI if (kind != "column" or cls == MCAR) else _POOL_SAFE_HI
           for (kind, _, _) in specs]
    scaled = _scale_to_budget(raw, widths, his, target_rate_cells)
    return [
        Unit(cls=cls, kind=kind, cols=cols_, rate=r, predictor=pred)
        for (kind, cols_, pred), r in zip(specs, scaled)
    ]


def plan_allocation(
    n: int,
    d: int,
    target: CompositionTarget,
    rng: RNGState,
    *,
    frac_observed_range: Tuple[float, float] = (0.1, 0.35),
    block_rate_share: float = 0.7,
    rate_spread: float = 1.2,
    max_block_width: int = 8,
) -> AllocationPlan:
    """Plan units that realise `target`'s by-cell composition on an (n, d) dataset.

    Args:
        n, d: dataset shape (rows, columns). Requires d >= 4 (room for >= 1 observed predictor and
            >= 3 working columns to host three classes).
        target: the drawn CompositionTarget (by-cell composition + overall miss rate).
        rng: explicit RNG (all draws flow through it).
        frac_observed_range: uniform draw for the fraction of fully-observed columns (clean MAR
            predictors + the realistic fully-observed-column tail; real `frac_cols_complete`≈0.21).
        block_rate_share: fraction of a structured class's columns packed into blocks (the rest are
            per-column). Higher => more co-missingness / fewer distinct patterns.
        rate_spread: Beta concentration ν for the per-column rate law (smaller => wider spread).
        max_block_width: cap on a single block's column count.

    Returns:
        AllocationPlan with `expected_composition` (by-cell, from the drawn rates) and
        `expected_miss_rate`.

    Raises:
        ValueError: on d < 4 or n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if d < 4:
        raise ValueError(f"composition allocation needs d >= 4 (got d={d}); the simplex cannot be "
                         "hosted on fewer working columns")

    lo, hi = frac_observed_range
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"frac_observed_range must satisfy 0 <= lo <= hi <= 1, got {frac_observed_range}")

    perm = [int(c) for c in rng.shuffle_indices(d)]
    n_obs = min(max(1, round(float(rng.numpy_rng.uniform(lo, hi)) * d)), d - 3)
    observed_cols = tuple(sorted(perm[:n_obs]))
    working = perm[n_obs:]
    w_count = len(working)

    cols_per_class = _largest_remainder(target.as_fractions(), w_count)
    mu = min(0.95, max(0.02, target.miss_rate * d / w_count))

    predictor = observed_cols[0]  # always observed => a clean MAR predictor (clean-MAR regime)
    units: List[Unit] = []
    off = 0
    for cls, n_cols, block_kinds, pred in (
        (MCAR, cols_per_class[MCAR], (), None),
        (MAR, cols_per_class[MAR], MAR_BLOCK_KINDS, predictor),
        (MNAR, cols_per_class[MNAR], MNAR_BLOCK_KINDS, None),
    ):
        cls_cols = working[off:off + n_cols]
        off += n_cols
        if cls_cols:
            # Class by-cell budget as rate-cells (cells_c / n): f_c · miss_rate · d. Hitting this
            # pins both the composition (to target) and the overall miss rate (to miss_rate).
            target_rate_cells = target.as_fractions()[cls] * target.miss_rate * d
            units.extend(_emit_units(cls_cols, cls, block_kinds, pred, mu, rate_spread,
                                     target_rate_cells, block_rate_share, max_block_width, rng))

    # Expected by-cell composition from the drawn rates: cells_c = n * sum(rate over class c's columns).
    cells = [0.0, 0.0, 0.0]
    for u in units:
        cells[u.cls] += u.rate * len(u.cols) * n
    total = sum(cells)
    if total <= 0.0:
        raise ValueError("plan produced zero expected missing cells; check target / dataset shape")
    expected_composition = tuple(c / total for c in cells)
    expected_miss_rate = total / (n * d)

    return AllocationPlan(
        units=tuple(units),
        observed_cols=observed_cols,
        n=n,
        d=d,
        target=target,
        expected_composition=expected_composition,
        expected_miss_rate=expected_miss_rate,
    )
