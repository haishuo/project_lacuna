"""
lacuna.survey.role_b_projection

Project a role-A (natural-missingness) survey source into a role-B complete-case base
(cross-domain plan §2; MASTER §7). We impose the δ-holes ourselves, so we keep ground truth.

ONE job: the pure transformation — recode refusal/admin sentinels → NaN, complete-case project (keep
columns with observed-rate ≥ τ_col, then rows complete over them), flag continuous δ-targets
(cardinality ≥ min_card), and emit a provenance record. The original role-A table is preserved
ELSEWHERE (the caller archives it); this module never discards it.

BINDING honesty (MASTER §7): the result is tagged `projected_from_naturally_missing` and kept distinct
from native-complete `survey_*` — complete-case projection is a mechanism-laden SELECTION (the base is
the complete-case subpopulation, not the population), and the provenance records exactly that.

Pure NumPy; deterministic; no RNG, no file I/O (the caller reads/writes). Fail loud (Rule 1).
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# Conservative default refusal/admin sentinel codes (surveys encode refuse/don't-know as numerics).
# Per-source overrides are expected (ESS 77/88/99…, NHANES 7/9/77/99/777/999…). Recorded in provenance.
DEFAULT_SENTINELS = (7, 8, 9, 77, 88, 99, 777, 888, 999, 7777, 9999)
PROJECTION_FLAG = "projected_from_naturally_missing"


@dataclass(frozen=True)
class ProjectionResult:
    data: np.ndarray              # [rows_retained, cols_kept] complete-case role-B matrix
    feature_names: Tuple[str, ...]
    targetable_idx: Tuple[int, ...]  # columns eligible as continuous δ-targets (card >= min_card)
    provenance: dict


def recode_sentinels(x: np.ndarray, sentinels: Sequence[float]) -> np.ndarray:
    """Return a copy of `x` with any sentinel value set to NaN (refusal/admin codes are not values)."""
    out = x.astype(np.float64, copy=True)
    if len(sentinels) == 0:
        return out
    mask = np.isin(out, np.asarray(sentinels, dtype=np.float64))
    out[mask] = np.nan
    return out


def _observed_rate(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.isnan(x).mean(axis=0)


def complete_case_project(
    x: np.ndarray, names: Sequence[str], *, tau_col: float,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Keep columns observed in ≥ τ_col of rows, then keep rows complete over those columns.

    Returns (sub_matrix, kept_names, kept_row_indices). Raises on empty/degenerate result.
    """
    if not (0.0 < tau_col <= 1.0):
        raise ValueError(f"tau_col must be in (0, 1], got {tau_col}")
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(f"x must be [n, {len(names)}], got {x.shape}")
    keep_cols = [j for j in range(x.shape[1]) if _observed_rate(x[:, [j]])[0] >= tau_col]
    if not keep_cols:
        raise ValueError(f"no column has observed-rate >= {tau_col}; cannot project")
    sub = x[:, keep_cols]
    complete = ~np.isnan(sub).any(axis=1)
    rows = np.where(complete)[0]
    if len(rows) < 2:
        raise ValueError(f"complete-case projection left {len(rows)} rows (<2); cannot project")
    return sub[rows], [names[j] for j in keep_cols], rows.tolist()


def continuous_targets(x: np.ndarray, *, min_card: int = 30) -> List[int]:
    """Indices of columns with cardinality ≥ min_card (continuous δ-targets; survey idioms need them)."""
    out = []
    for j in range(x.shape[1]):
        col = x[:, j]
        col = col[~np.isnan(col)]
        if np.unique(col).size >= min_card:
            out.append(j)
    return out


def project_source(
    x: np.ndarray, names: Sequence[str], *, source: str, name: str, domain: str, source_block: str,
    sentinels: Sequence[float] = DEFAULT_SENTINELS, tau_col: float = 0.95, min_card: int = 30,
) -> ProjectionResult:
    """Full role-A → role-B projection with provenance (cross-domain plan §2)."""
    n0, d0 = x.shape
    recoded = recode_sentinels(np.asarray(x, dtype=np.float64), sentinels)
    sub, kept_names, kept_rows = complete_case_project(recoded, names, tau_col=tau_col)
    tgt = continuous_targets(sub, min_card=min_card)
    if not tgt:
        raise ValueError(f"{name}: no continuous target (card>={min_card}) survived projection")
    provenance = {
        "source": source, "name": name, "domain": domain, "source_block": source_block,
        "flag": PROJECTION_FLAG, "projection": "complete_case_from_naturally_missing",
        "sentinels_recoded": list(sentinels), "tau_col": tau_col, "min_card": min_card,
        "n_rows_in": int(n0), "n_cols_in": int(d0),
        "rows_retained": int(sub.shape[0]), "cols_kept": int(sub.shape[1]),
        "n_continuous_targets": len(tgt), "targetable_cols": [kept_names[j] for j in tgt],
        "survey_weights": "present_but_unused",
    }
    return ProjectionResult(data=sub.astype(np.float32), feature_names=tuple(kept_names),
                            targetable_idx=tuple(tgt), provenance=provenance)


def to_raw_dataset(result: ProjectionResult):
    """Wrap a role-B projection as a RawDataset (flagged source) for the generators."""
    from lacuna.data.ingestion import RawDataset
    return RawDataset(data=result.data, feature_names=result.feature_names,
                      source=PROJECTION_FLAG, name=result.provenance["name"])
