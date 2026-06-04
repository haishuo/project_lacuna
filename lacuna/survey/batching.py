"""
lacuna.survey.batching

Generator-to-batch plumbing: P2.1 answer-sheeted examples -> (TokenBatch, out-of-band labels).

ONE job: turn real survey datasets into tokenized training batches for the δ-prior, carrying the
δ-bin label and the continuous δ ALONGSIDE the batch — never inside `generator_ids`/`class_ids`/
`variant_ids` (those belong to the dead 3-class world; P2.2 constraint #3). Rows are subsampled
BEFORE the mechanism runs (reusing `subsample_raw`), matching the v1.0 hot path and keeping the
matched-rate solve on the rows the model actually sees.

Determinism (Rule 6): every example is drawn from an injected RNGState; the same seed reproduces
identical batches and labels. Fail loud (Rule 1) on an empty example list or a dataset too wide
for `max_cols`.
"""

from dataclasses import dataclass
from typing import List

import torch

from lacuna.core.rng import RNGState
from lacuna.core.types import ObservedDataset, TokenBatch
from lacuna.data.ingestion import RawDataset
from lacuna.data.semisynthetic import subsample_raw
from lacuna.data.tokenization import tokenize_and_batch

from .answer_sheet import AnswerSheet
from .delta_generator import generate_self_censor_example


@dataclass(frozen=True)
class DeltaExample:
    """One tokenizable example: the observed (censored) dataset + its answer sheet."""

    observed: ObservedDataset
    answer_sheet: AnswerSheet


@dataclass(frozen=True)
class DeltaBatch:
    """A model-ready batch: tokens + the out-of-band δ supervision and provenance."""

    tokens: TokenBatch
    delta_bin: torch.Tensor  # [B] long — the supervised ordered-bin label
    delta: torch.Tensor  # [B] float — the continuous answer-sheet δ (for E[δ] error)
    sheets: List[AnswerSheet]  # per-example audit records (provenance; not fed to the model)


def make_example(
    raw: RawDataset,
    *,
    beta1: float,
    delta: float,
    target_rate: float,
    rng: RNGState,
    max_rows: int,
) -> DeltaExample:
    """Subsample rows, apply own-value self-censoring, wrap as a tokenizable ObservedDataset."""
    raw_sub = subsample_raw(raw, max_rows=max_rows, rng=rng.spawn())
    res = generate_self_censor_example(
        raw_sub, beta1=beta1, delta=delta, target_rate=target_rate, rng=rng.spawn()
    )
    n, d = res.mask.shape
    observed = ObservedDataset(
        x=res.x_observed,
        r=res.mask,
        n=n,
        d=d,
        feature_names=raw_sub.feature_names,
        dataset_id=raw.name,
        meta={"source": raw.name, "is_semisynthetic": True},
    )
    return DeltaExample(observed=observed, answer_sheet=res.answer_sheet)


def collate(examples: List[DeltaExample], *, max_rows: int, max_cols: int) -> DeltaBatch:
    """Tokenize a list of examples into a DeltaBatch (δ-bin label carried out-of-band).

    `generator_ids`/`class_ids`/`variant_ids` are left None — the δ objective never uses the
    3-class fields (P2.2 constraint #3).
    """
    if len(examples) == 0:
        raise ValueError("cannot collate an empty example list")
    datasets = [e.observed for e in examples]
    for ds in datasets:
        if ds.d > max_cols:
            raise ValueError(f"dataset '{ds.dataset_id}' has d={ds.d} > max_cols={max_cols}")
    batch = tokenize_and_batch(datasets, max_rows=max_rows, max_cols=max_cols)
    delta_bin = torch.tensor([e.answer_sheet.delta_bin for e in examples], dtype=torch.long)
    delta = torch.tensor([e.answer_sheet.delta for e in examples], dtype=torch.float32)
    sheets = [e.answer_sheet for e in examples]
    return DeltaBatch(tokens=batch, delta_bin=delta_bin, delta=delta, sheets=sheets)
