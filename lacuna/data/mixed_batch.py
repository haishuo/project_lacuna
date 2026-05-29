"""
lacuna.data.mixed_batch

Build per-column-labelled training/eval batches for the column-level experiment (Stage 1,
ADR-0006). Each batch item is a mixed-mechanism dataset (different mechanism per column,
clean-MAR regime) produced by `compose_mixed_missingness`; the per-column ground-truth
mechanism labels travel alongside the TokenBatch as separate tensors (the frozen TokenBatch
type is left unchanged).

Returns, per batch:
  - TokenBatch        : tokens/row_mask/col_mask, exactly as the dataset-level path expects.
  - labels [B, C]     : per-column mechanism class (MCAR/MAR/MNAR) at supervised positions;
                        0 (placeholder) elsewhere.
  - supervision_mask  : [B, C] bool — True only for columns with a known mechanism
                        (non-OBSERVED, non-padding). Loss/metrics mask on this.
  - compositions      : list of the per-item column_classes tuples (ground truth for analysis).

Determinism (Coding Bible Rule 6): all randomness flows through an explicit RNGState.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import TokenBatch, MCAR, MAR, MNAR
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import subsample_raw
from lacuna.data.mixed_missingness import compose_mixed_missingness, OBSERVED
from lacuna.data.tokenization import tokenize_and_batch

_MECHANISMS = (MCAR, MAR, MNAR)


@dataclass(frozen=True)
class MixedBatch:
    """A per-column-labelled batch (TokenBatch + per-column targets).

    `complete_values` carries the TRUE (un-zeroed) values for every cell, padded to
    [B, max_rows, max_cols]. It is only knowable because the data is semi-synthetic; Stage 2b
    uses it as the reconstruction target so per-column reconstruction error reflects genuine
    accuracy rather than prediction magnitude (vs the zeroed `TokenBatch.original_values`).
    """
    batch: TokenBatch
    labels: torch.Tensor            # [B, max_cols] long
    supervision_mask: torch.Tensor  # [B, max_cols] bool
    compositions: Tuple[Tuple[int, ...], ...]  # per-item column_classes
    complete_values: torch.Tensor   # [B, max_rows, max_cols] true values (un-zeroed)
    mnar_subtypes: Tuple[Dict[int, str], ...]  # per-item {col -> MNAR subtype} (realised mechanism)
    mar_subtypes: Tuple[Dict[int, str], ...]   # per-item {col -> MAR subtype} (realised mechanism)


def sample_column_classes(d: int, rng: RNGState, p_observed: float = 0.25) -> Tuple[int, ...]:
    """Sample a per-column mechanism assignment for a d-column dataset.

    Each column is OBSERVED (fully observed) with probability `p_observed`, else a uniform
    draw from {MCAR, MAR, MNAR}. Two guards keep the result valid for the composer:
      - at least one OBSERVED-or-MCAR column exists (a clean MAR predictor; ADR-0006);
      - at least one column carries a mechanism (otherwise nothing is supervised).
    For d < 2 the logistic MAR/MNAR generators are unusable, so only {OBSERVED, MCAR} are drawn.
    """
    if not 0.0 <= p_observed <= 1.0:
        raise ValueError(f"p_observed must be in [0, 1], got {p_observed}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")

    pool_choices = (OBSERVED, MCAR) if d < 2 else (OBSERVED, MCAR, MAR, MNAR)
    classes = []
    for _ in range(d):
        if rng.rand(1).item() < p_observed:
            classes.append(OBSERVED)
        else:
            # uniform over mechanisms available at this d
            mechs = (MCAR,) if d < 2 else _MECHANISMS
            classes.append(mechs[rng.randint(0, len(mechs), (1,)).item()])

    # Guard 1: ensure a clean predictor (OBSERVED or MCAR) for any MAR columns.
    if not any(c in (OBSERVED, MCAR) for c in classes):
        classes[rng.randint(0, d, (1,)).item()] = MCAR
    # Guard 2: ensure at least one supervised (mechanism-bearing) column.
    if not any(c in _MECHANISMS for c in classes):
        classes[rng.randint(0, d, (1,)).item()] = MCAR
    return tuple(classes)


def build_mixed_batch(
    raws: List[RawDataset],
    rng: RNGState,
    *,
    max_rows: int,
    max_cols: int,
    batch_size: int,
    p_observed: float = 0.25,
    target_miss_rate: float = 0.25,
    mar_strength: float = 1.5,
    mnar_strength: float = 1.5,
    mnar_diverse: bool = False,
    mar_diverse: bool = False,
) -> MixedBatch:
    """Assemble one per-column-labelled batch by sampling datasets and mixed compositions.

    `mnar_diverse` / `mar_diverse` (both default False) are forwarded to
    `compose_mixed_missingness`: when True, MNAR / MAR columns draw diverse subtypes instead of
    the single-family default. The per-column LABEL is unchanged (MNAR / MAR); only the realised
    subtype varies. The realised subtypes are returned per item (`mnar_subtypes`, `mar_subtypes`)
    so evaluation can break detectability down by subtype (Stage 5, ADR-0006).
    """
    if not raws:
        raise ValueError("raws must be non-empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    observed_datasets = []
    compositions: List[Tuple[int, ...]] = []
    complete_data = []  # per-item true (complete) values, numpy [n_i, d_i]
    mnar_subtypes: List[Dict[int, str]] = []
    mar_subtypes: List[Dict[int, str]] = []

    for _ in range(batch_size):
        item_rng = rng.spawn()
        raw = raws[item_rng.randint(0, len(raws), (1,)).item()]
        raw_sub = subsample_raw(raw, max_rows=max_rows, rng=item_rng.spawn())
        classes = sample_column_classes(raw_sub.d, item_rng.spawn(), p_observed=p_observed)
        res = compose_mixed_missingness(
            raw_sub, classes, item_rng.spawn(),
            target_miss_rate=target_miss_rate,
            mar_strength=mar_strength, mnar_strength=mnar_strength,
            mnar_diverse=mnar_diverse, mar_diverse=mar_diverse,
        )
        observed_datasets.append(res.observed)
        compositions.append(res.column_classes)
        complete_data.append(raw_sub.data)  # row order matches the observed dataset / tokenizer
        mnar_subtypes.append(res.mnar_subtypes)
        mar_subtypes.append(res.mar_subtypes)

    batch = tokenize_and_batch(observed_datasets, max_rows=max_rows, max_cols=max_cols)

    labels = torch.zeros(batch_size, max_cols, dtype=torch.long)
    sup_mask = torch.zeros(batch_size, max_cols, dtype=torch.bool)
    complete_values = torch.zeros(batch_size, max_rows, max_cols, dtype=torch.float32)
    for i, classes in enumerate(compositions):
        for j, c in enumerate(classes):  # j < d_i <= max_cols (validated upstream)
            if c in _MECHANISMS:
                labels[i, j] = c
                sup_mask[i, j] = True
        data = complete_data[i]
        n, d = data.shape  # n <= max_rows (subsampled upstream), d <= max_cols
        complete_values[i, :n, :d] = torch.from_numpy(data.astype("float32"))

    return MixedBatch(
        batch=batch,
        labels=labels,
        supervision_mask=sup_mask,
        compositions=tuple(compositions),
        complete_values=complete_values,
        mnar_subtypes=tuple(mnar_subtypes),
        mar_subtypes=tuple(mar_subtypes),
    )
