"""
lacuna.data.subtype_targets

Glue between a per-column-labelled `MixedBatch` and the subtype ontology (ADR-0008 commitment 6).

`build_mixed_batch` returns, per item, the per-column MECHANISM label (MCAR/MAR/MNAR) plus the
realised generator SUBTYPE for each MNAR / MAR column (`mnar_subtypes` / `mar_subtypes`). This module
turns those into the two per-column target tensors the subtype layer trains and scores against:

  - likelihood_labels [B, C]: the DATA-readout target over L (`subtype_ontology.LIKELIHOOD_LABELS`) --
    threshold / detection get their loud label; every other supervised column folds to the reject
    class (the honest "the data should abstain here" target).
  - ontology_truth   [B, C]: the GROUND-TRUTH O index (`subtype_ontology.SUBTYPES`) for scoring the
    fused prior x likelihood posterior and the dataset subtype-composition.

Both are 0 at unsupervised positions; `supervision_mask` (echoed from the batch) selects the
mechanism-bearing columns. Determinism (Rule 6): pure, no RNG. Fails loud (Rule 1) if a supervised
MNAR/MAR column is missing its recorded subtype (a broken data path, not silently defaulted).
"""

from typing import Tuple

import torch

from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.data.mixed_batch import MixedBatch
from lacuna.priors.subtype_ontology import (
    realised_subtype_to_likelihood_label, realised_subtype_to_ontology,
)

_MCAR_SENTINEL = "mcar"  # MCAR columns carry no generator subtype; the maps ignore the string for MCAR.


def subtype_targets(mb: MixedBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-column (likelihood_labels [B,C] long, ontology_truth [B,C] long, supervision_mask [B,C] bool).

    Raises:
        ValueError: if a supervised MNAR/MAR column has no recorded subtype (broken batch).
    """
    labels = mb.labels
    sup = mb.supervision_mask
    B, C = labels.shape
    like = torch.zeros(B, C, dtype=torch.long)
    onto = torch.zeros(B, C, dtype=torch.long)
    for i in range(B):
        mnar_sub = mb.mnar_subtypes[i]
        mar_sub = mb.mar_subtypes[i]
        for j in range(C):
            if not bool(sup[i, j]):
                continue
            mech = int(labels[i, j])
            if mech == MNAR:
                if j not in mnar_sub:
                    raise ValueError(f"supervised MNAR column (item {i}, col {j}) has no recorded subtype")
                subtype = mnar_sub[j]
            elif mech == MAR:
                if j not in mar_sub:
                    raise ValueError(f"supervised MAR column (item {i}, col {j}) has no recorded subtype")
                subtype = mar_sub[j]
            else:  # MCAR
                subtype = _MCAR_SENTINEL
            like[i, j] = realised_subtype_to_likelihood_label(mech, subtype)
            onto[i, j] = realised_subtype_to_ontology(mech, subtype)
    return like, onto, sup
