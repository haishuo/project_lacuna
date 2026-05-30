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
from typing import List, Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import TokenBatch
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import subsample_raw
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.data.composition_target import sample_composition_target
from lacuna.data.composition_sampler import compose_composition_missingness

_MIN_D = 4  # the composition sampler needs >= 4 columns to host a 3-class composition


@dataclass(frozen=True)
class CompositionBatch:
    """A composition-labelled batch (TokenBatch + the realised by-cell composition target)."""
    batch: TokenBatch
    composition: torch.Tensor       # [B, 3] realised by-cell composition (supervised target)
    target_drawn: torch.Tensor      # [B, 3] composition drawn from the prior (analysis only)
    miss_rate: torch.Tensor         # [B] realised overall missing fraction
    source_names: Tuple[str, ...]


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

    Returns:
        CompositionBatch.

    Raises:
        ValueError: on empty `raws`, batch_size < 1, or a dataset with d < 4 or d > max_cols.
    """
    if not raws:
        raise ValueError("raws must be non-empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    for r in raws:
        if r.d < _MIN_D or r.d > max_cols:
            raise ValueError(f"dataset '{r.name}' has d={r.d}; require {_MIN_D} <= d <= {max_cols}")

    observed_datasets = []
    realized: List[Tuple[float, float, float]] = []
    drawn: List[Tuple[float, float, float]] = []
    miss: List[float] = []
    names: List[str] = []

    for _ in range(batch_size):
        item_rng = rng.spawn()
        raw = raws[item_rng.randint(0, len(raws), (1,)).item()]
        raw_sub = subsample_raw(raw, max_rows=max_rows, rng=item_rng.spawn())
        target = sample_composition_target(item_rng.spawn(), concentration=concentration,
                                           miss_rate_range=miss_rate_range)
        res = compose_composition_missingness(
            raw_sub, target, item_rng.spawn(),
            strength=strength, block_rate_share=block_rate_share)
        observed_datasets.append(res.observed)
        realized.append(res.realized_composition)
        drawn.append(target.as_fractions())
        miss.append(res.realized_miss_rate)
        names.append(res.source_name)

    batch = tokenize_and_batch(observed_datasets, max_rows=max_rows, max_cols=max_cols)
    return CompositionBatch(
        batch=batch,
        composition=torch.tensor(realized, dtype=torch.float32),
        target_drawn=torch.tensor(drawn, dtype=torch.float32),
        miss_rate=torch.tensor(miss, dtype=torch.float32),
        source_names=tuple(names),
    )
