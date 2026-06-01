"""
lacuna.data.composition_batch

Build composition-labelled training/eval batches for the Stage-C composition head (ADR-0007).

Each batch item is a complete catalog dataset punched by the Stage-B composition-controlled generator
(`compose_composition_missingness`) at a target composition drawn from the broad simplex prior. The
supervised target is the dataset's **realised by-cell composition** `(f_MCAR, f_MAR, f_MNAR)` — the same
denominator the generator, the per-cell tags, and the eval metric share (ADR-0007 commitment 1). The
realised composition (not the drawn target) is the label, because it is what the generator actually
produced and tagged; the drawn target is carried alongside for analysis.

Returns, per batch:
  - TokenBatch        : tokens/row_mask/col_mask, exactly as the encoder expects.
  - composition [B,3] : the realised by-cell composition (the supervised regression/Dirichlet target).
  - target_drawn [B,3]: the composition that was drawn from the prior (for analysis only).
  - miss_rate [B]     : realised overall missing fraction per item.

Determinism (Coding Bible Rule 6): all randomness flows through an injected RNGState. Fails loud
(Rule 1) on an empty / too-narrow dataset list.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import TokenBatch
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import subsample_raw
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.data.missingness_footprint import missingness_footprint, FOOTPRINT_FEATURES
from lacuna.data.composition_target import sample_composition_target, CompositionTarget
from lacuna.data.composition_sampler import compose_composition_missingness

N_FOOTPRINT_FEATURES = len(FOOTPRINT_FEATURES)

_MIN_D = 4  # the composition sampler needs >= 4 columns to host a 3-class composition


@dataclass(frozen=True)
class CompositionBatch:
    """A composition-labelled batch (TokenBatch + the realised by-cell composition target).

    `footprints` is the [B, 20] observable missingness footprint per item (or None if not requested):
    the explicit cross-column features the encoder's pooled evidence under-represents (Stage-C
    attribution). Computed from each item's mask + observed values — deployable (no oracle).
    """
    batch: TokenBatch
    composition: torch.Tensor       # [B, 3] realised by-cell composition (supervised target)
    target_drawn: torch.Tensor      # [B, 3] composition drawn from the prior (analysis only)
    miss_rate: torch.Tensor         # [B] realised overall missing fraction
    source_names: Tuple[str, ...]
    footprints: torch.Tensor = None  # [B, 20] observable footprint, or None


def build_composition_batch(
    raws: List[RawDataset],
    rng: RNGState,
    *,
    max_rows: int,
    max_cols: int,
    batch_size: int,
    concentration: float = 1.0,
    miss_rate_range: Tuple[float, float] = (0.05, 0.6),
    strength: float = 1.5,
    block_rate_share: float = 0.85,
    with_footprints: bool = False,
    fixed_composition: Optional[Tuple[float, float, float]] = None,
    mnar_subtypes: Optional[Sequence[str]] = None,
    mnar_block_share: Optional[float] = None,
) -> CompositionBatch:
    """Assemble one composition-labelled batch by sampling datasets and drawn compositions.

    Args:
        raws: complete RawDatasets (each with d >= 4 and <= max_cols).
        rng: explicit RNG state.
        max_rows, max_cols: tokenizer padding dimensions (match the encoder config).
        batch_size: number of datasets in the batch.
        concentration: Dirichlet concentration of the composition prior (1.0 = uniform simplex).
        miss_rate_range: overall miss-rate prior range.
        strength, block_rate_share: forwarded to the Stage-B sampler.
        with_footprints: also compute the [B, 20] observable footprint per item (deployable
            features for the composition head; off by default to avoid the cost when unused).
        fixed_composition: optional ``(f_MCAR, f_MAR, f_MNAR)`` (summing to 1) used for EVERY item
            instead of a fresh simplex draw — the miss rate is still drawn from `miss_rate_range`,
            and the rng is consumed identically to the drawn path, so two calls with the same rng
            seed differing only in `mnar_subtypes` are matched item-for-item (the Stage-F design).
            ``None`` (default) = draw the composition from the prior (bit-identical to before).
        mnar_subtypes, mnar_block_share: forwarded to the Stage-B sampler (default ``None`` = full
            MNAR pool, MNAR block share = `block_rate_share`). Stage-F sets these to restrict the
            per-column MNAR family (loud vs quiet) and force MNAR per-column-only.

    Returns:
        CompositionBatch (footprints is None unless with_footprints=True).

    Raises:
        ValueError: on empty `raws`, batch_size < 1, a dataset with d < 4 or d > max_cols, or a
            `fixed_composition` that is not three non-negative fractions summing to 1.
    """
    if not raws:
        raise ValueError("raws must be non-empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for r in raws:
        if r.d < _MIN_D or r.d > max_cols:
            raise ValueError(f"dataset '{r.name}' has d={r.d}; require {_MIN_D} <= d <= {max_cols}")
    if fixed_composition is not None:
        if len(fixed_composition) != 3:
            raise ValueError(f"fixed_composition must have 3 entries, got {fixed_composition}")
        if abs(sum(fixed_composition) - 1.0) > 1e-6:
            raise ValueError(f"fixed_composition must sum to 1, got {fixed_composition}")

    observed_datasets = []
    realized: List[Tuple[float, float, float]] = []
    drawn: List[Tuple[float, float, float]] = []
    miss: List[float] = []
    names: List[str] = []
    footprints: List[List[float]] = []

    for _ in range(batch_size):
        item_rng = rng.spawn()
        raw = raws[item_rng.randint(0, len(raws), (1,)).item()]
        raw_sub = subsample_raw(raw, max_rows=max_rows, rng=item_rng.spawn())
        target = sample_composition_target(item_rng.spawn(), concentration=concentration,
                                           miss_rate_range=miss_rate_range)
        if fixed_composition is not None:
            # Override the composition with the fixed target but keep the drawn miss rate (so the
            # rng is consumed identically to the drawn path → matched corpora across mnar_subtypes).
            target = CompositionTarget(f_mcar=fixed_composition[0], f_mar=fixed_composition[1],
                                       f_mnar=fixed_composition[2], miss_rate=target.miss_rate)
        res = compose_composition_missingness(
            raw_sub, target, item_rng.spawn(),
            strength=strength, block_rate_share=block_rate_share,
            mnar_subtypes=mnar_subtypes, mnar_block_share=mnar_block_share)
        observed_datasets.append(res.observed)
        realized.append(res.realized_composition)
        drawn.append(target.as_fractions())
        miss.append(res.realized_miss_rate)
        names.append(res.source_name)
        if with_footprints:
            fp = missingness_footprint(res.observed.x, res.observed.r)
            footprints.append([fp[k] for k in FOOTPRINT_FEATURES])

    batch = tokenize_and_batch(observed_datasets, max_rows=max_rows, max_cols=max_cols)
    return CompositionBatch(
        batch=batch,
        composition=torch.tensor(realized, dtype=torch.float32),
        target_drawn=torch.tensor(drawn, dtype=torch.float32),
        miss_rate=torch.tensor(miss, dtype=torch.float32),
        source_names=tuple(names),
        footprints=torch.tensor(footprints, dtype=torch.float32) if with_footprints else None,
    )
