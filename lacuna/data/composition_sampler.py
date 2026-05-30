"""
lacuna.data.composition_sampler

Realise a target by-cell missingness COMPOSITION on a complete dataset, with exact per-cell tags
(ADR-0007, Stage B — the composition-controlled, realism-tuned generator).

This is the application half of the sampler (planning lives in `composition_allocator`). Given a
complete dataset and a drawn `CompositionTarget`, it:
  1. plans an allocation (which columns/blocks carry which mechanism, at what rate);
  2. applies each unit — per-column mechanisms via the existing MCAR / MAR / MNAR column pools (so
     the per-column fingerprints of Stage-A Band 3 are preserved), block mechanisms via
     `composition_blocks` (so Band-1 co-missingness / few-pattern / monotone structure appears);
  3. records a per-CELL mechanism TAG matrix — the by-cell ground truth — and measures the REALISED
     composition from it.

The generator IS the operational definition of the estimand (ADR-0007 commitment 1): the target it
aims for, the per-cell tags it stamps, and the realised composition it reports all use the SAME
denominator — the missing cell. Because every column is owned by exactly one mechanism unit
(disjoint allocation), each missing cell has an unambiguous tag; this is how the LOCKED joint-cell
convention ("a joint mechanism stamps every cell it co-deletes with its own class") is realised
without any cross-mechanism overlap to adjudicate.

The honest seam is preserved structurally: MCAR is the per-column random anchor (the identifiable
"structured-vs-random" axis), while MAR and MNAR both carry block structure (so the MAR-vs-MNAR
split stays the genuinely non-identifiable part). The tags are exact regardless of identifiability —
they are a generative fact, not an inferential claim (ADR-0007 §For statisticians).

Determinism (Coding Bible Rule 6): all randomness flows through the injected RNGState. Fails loud
(Rule 1) on a contract violation (incomplete input, untagged missing cell — an internal invariant).
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset, MCAR, MAR, MNAR
from lacuna.generators.params import GeneratorParams
from lacuna.generators.families.mcar.bernoulli import MCARBernoulli
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.data.composition_target import CompositionTarget
from lacuna.data.composition_allocator import AllocationPlan, plan_allocation
from lacuna.data.composition_blocks import apply_block
from lacuna.data.mnar_column_pool import sample_mnar_column_generator
from lacuna.data.mar_column_pool import sample_mar_column_generator

# Cell-tag sentinel: this cell is OBSERVED (not missing), so it carries no mechanism class. Distinct
# from the mechanism class ids MCAR=0 / MAR=1 / MNAR=2 that tag the missing cells.
NOT_MISSING = -1


@dataclass(frozen=True)
class CompositionResult:
    """Result of composing a dataset to a target by-cell missingness composition.

    Attributes:
        observed: ObservedDataset with all mechanisms applied (missing cells = 0).
        cell_tags: [n, d] long tensor; each MISSING cell holds its producing mechanism's class
            (MCAR/MAR/MNAR), each observed cell holds NOT_MISSING. The by-cell ground truth.
        target: the CompositionTarget the sampler aimed for.
        realized_composition: (f_MCAR, f_MAR, f_MNAR) of the MISSING CELLS actually produced — the
            same by-cell denominator as the target (ADR-0007 commitment 1). This, not the target, is
            the operational truth the estimand is defined against.
        realized_miss_rate: fraction of all cells that are missing.
        n_missing_cells: total missing cells (the composition denominator).
        per_column_miss_rate: realised per-column missing fraction (reported, never assumed).
        plan: the AllocationPlan used (full traceability).
        source_name: name of the source RawDataset.
    """
    observed: ObservedDataset
    cell_tags: torch.Tensor
    target: CompositionTarget
    realized_composition: Tuple[float, float, float]
    realized_miss_rate: float
    n_missing_cells: int
    per_column_miss_rate: Tuple[float, ...]
    plan: AllocationPlan
    source_name: str


def _per_column_unit_mask(cls: int, col: int, predictor, rate: float, z_full: torch.Tensor,
                          rng: RNGState, strength: float) -> torch.Tensor:
    """Apply a single per-column mechanism to the full predictor view; return the target column mask.

    MAR/MNAR use the diverse column pools (Band-3 fingerprints); MCAR is per-column Bernoulli (the
    random anchor). The pools target a column in the FULL matrix (MAR needs its clean predictor,
    which lives in another column), exactly as `compose_mixed_missingness` does.
    """
    if cls == MCAR:
        gen = MCARBernoulli(0, "comp_mcar", GeneratorParams(miss_rate=rate))
    elif cls == MAR:
        _, gen = sample_mar_column_generator(col, predictor, rng.spawn(),
                                             target_miss_rate=rate, strength=strength)
    else:  # MNAR
        _, gen = sample_mnar_column_generator(col, rng.spawn(),
                                              target_miss_rate=rate, strength=strength)
    return gen.apply_to(z_full, rng.spawn())[:, col]


def compose_composition_missingness(
    raw: RawDataset,
    target: CompositionTarget,
    rng: RNGState,
    *,
    strength: float = 1.5,
    frac_observed_range: Tuple[float, float] = (0.1, 0.35),
    block_rate_share: float = 0.85,
    rate_spread: float = 1.2,
    max_block_width: int = 8,
) -> CompositionResult:
    """Compose `raw` to `target`'s by-cell composition; return the masked data + per-cell tags.

    Args:
        raw: Complete RawDataset (no missing values).
        target: drawn CompositionTarget (by-cell composition + overall miss rate).
        rng: explicit RNG for reproducibility.
        strength: value→missingness coupling strength forwarded to the column pools and blocks.
        frac_observed_range, block_rate_share, rate_spread, max_block_width: allocator knobs (see
            `composition_allocator.plan_allocation`).

    Returns:
        CompositionResult (the realised composition is the ground truth, not the target).

    Raises:
        ValueError: if `raw` has missing values, or an internal tagging invariant is violated.
    """
    X = torch.from_numpy(raw.data.astype("float32"))
    if not torch.isfinite(X).all():
        raise ValueError(f"raw dataset '{raw.name}' has non-finite values; expected complete data")
    n, d = X.shape
    Z = _zscore_columns(X)

    plan = plan_allocation(
        n, d, target, rng.spawn(),
        frac_observed_range=frac_observed_range, block_rate_share=block_rate_share,
        rate_spread=rate_spread, max_block_width=max_block_width,
    )

    R = torch.ones(n, d, dtype=torch.bool)
    tags = torch.full((n, d), NOT_MISSING, dtype=torch.long)

    for unit in plan.units:
        if unit.kind == "column":
            col = unit.cols[0]
            r_col = _per_column_unit_mask(unit.cls, col, unit.predictor, unit.rate, Z,
                                          rng.spawn(), strength)
            R[:, col] = r_col
            tags[~r_col, col] = unit.cls
        else:
            cols = list(unit.cols)
            r_block = apply_block(unit.kind, Z[:, cols], unit.rate, rng.spawn(), strength=strength)
            for k, col in enumerate(cols):
                R[:, col] = r_block[:, k]
                tags[~r_block[:, k], col] = unit.cls

    # Guard: never leave a column fully missing (mirrors apply_missingness). A forced-observed cell
    # is no longer produced by any mechanism, so its tag reverts to NOT_MISSING (keeps tags exact).
    for j in range(d):
        if not bool(R[:, j].any()):
            i = int(rng.randint(0, n, (1,)).item())
            R[i, j] = True
            tags[i, j] = NOT_MISSING

    missing = ~R
    n_missing = int(missing.sum().item())
    # Internal invariant: every missing cell is tagged, every observed cell is not (Coding Bible
    # Rule 1 — fail loud rather than silently miscount the denominator).
    if int((tags != NOT_MISSING).sum().item()) != n_missing:
        raise ValueError("tagging invariant violated: tagged-cell count != missing-cell count")
    if n_missing == 0:
        raise ValueError(f"composition produced zero missing cells on '{raw.name}'")

    counts = [int((tags == c).sum().item()) for c in (MCAR, MAR, MNAR)]
    realized = tuple(c / n_missing for c in counts)
    per_col = (1.0 - R.float().mean(dim=0)).tolist()

    X_observed = X * R.float()
    observed = ObservedDataset(
        x=X_observed, r=R, n=n, d=d,
        feature_names=raw.feature_names,
        dataset_id=f"{raw.name}_composition",
        meta={"source": raw.source, "is_composition": True},
    )

    return CompositionResult(
        observed=observed,
        cell_tags=tags,
        target=target,
        realized_composition=realized,
        realized_miss_rate=n_missing / (n * d),
        n_missing_cells=n_missing,
        per_column_miss_rate=tuple(round(float(m), 4) for m in per_col),
        plan=plan,
        source_name=raw.name,
    )
