"""
lacuna.survey.column_batching

Column-major batching for the Level-1 φ-spine (Stage-1 spec §2; MASTER §5).

ONE job: turn a list of `batching.DeltaExample`s (role-B complete-projection X + imposed δ-holes)
into a COLUMN-MAJOR batch for the per-column distribution encoder φ — the observed values of the
SUPPLIED target column as a row-permutation-invariant set, standardized within observed. This is the
deliberate inversion of the row-major per-cell tokenization the BERT encoder used (which is NOT the
δ spine; MASTER §10). The δ-bin label + continuous δ are carried out-of-band (never in the values).

Determinism (Rule 6): no RNG here — row subsampling and hole-punching already happened upstream in
the generators. Fail loud (Rule 1) on an empty list or an all-missing target column.
"""

from dataclasses import dataclass
from typing import List

import torch

from .batching import DeltaExample

_MIN_OBS = 1  # at least one observed target value is required to form a distribution


@dataclass(frozen=True)
class ColumnBatch:
    """Column-major batch: observed target values (a set per example) + out-of-band δ supervision."""

    target_values: torch.Tensor  # [B, max_rows, 1] standardized-within-observed; padding = 0
    value_mask: torch.Tensor     # [B, max_rows] bool — True = a real observed value
    target_idx: torch.Tensor     # [B] long — the supplied target column (provenance)
    delta_bin: torch.Tensor      # [B] long — 7-bin label
    delta: torch.Tensor          # [B] float — continuous answer-sheet δ (for any coarse scheme)
    sheets: List                 # per-example AnswerSheet (provenance; not fed to the model)


def collate_columns(examples: List[DeltaExample], *, max_rows: int) -> ColumnBatch:
    """Build a ColumnBatch from DeltaExamples (observed target values, standardized within observed).

    Args:
        examples: role-B DeltaExamples (observed dataset + answer sheet).
        max_rows: pad/truncate each example's observed-value set to this length.

    Raises:
        ValueError: empty list, or an example whose target column has no observed values.
    """
    if len(examples) == 0:
        raise ValueError("cannot collate an empty example list")
    b = len(examples)
    V = torch.zeros(b, max_rows, 1, dtype=torch.float32)
    MK = torch.zeros(b, max_rows, dtype=torch.bool)
    for i, ex in enumerate(examples):
        t = ex.answer_sheet.target_col_idx
        obs = ex.observed.r[:, t].bool()
        v = ex.observed.x[obs, t].to(torch.float32)
        if int(v.numel()) < _MIN_OBS:
            raise ValueError(
                f"example {i} ('{ex.observed.dataset_id}') target col {t} has no observed values"
            )
        sd = v.std(unbiased=False)
        v = (v - v.mean()) / sd if float(sd) > 0 else v - v.mean()  # scale-invariant within observed
        k = min(int(v.numel()), max_rows)
        V[i, :k, 0] = v[:k]
        MK[i, :k] = True
    delta_bin = torch.tensor([e.answer_sheet.delta_bin for e in examples], dtype=torch.long)
    delta = torch.tensor([e.answer_sheet.delta for e in examples], dtype=torch.float32)
    target_idx = torch.tensor([e.answer_sheet.target_col_idx for e in examples], dtype=torch.long)
    return ColumnBatch(
        target_values=V, value_mask=MK, target_idx=target_idx,
        delta_bin=delta_bin, delta=delta, sheets=[e.answer_sheet for e in examples],
    )
