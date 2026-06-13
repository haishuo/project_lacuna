"""
lacuna.analysis.realism_gate.corpora

Load real cell-level missingness from the raw survey corpora into a plasmode-ready
``MaskCorpus`` for the realism gate.

One job: raw corpus file(s) -> (a) ``M_real`` = the real binary missing mask on a
chosen, column-aligned item set, with per-row block labels; and (b)
``X_complete`` = a complete-case real-X substrate on the SAME columns that
generators inject holes into via ``apply_to``. The gate then asks whether a
generator's masks on ``X_complete`` look like ``M_real``.

Sentinels are resolved per column with the width-based ESS resolver
(``lacuna.survey.ess_codes``) — the same resolver works for NHANES bounded items
(7/9 = refuse/DK on short scales); for a *binary* missing mask we only need
"is this cell a missing sentinel", not the sentinel type. GSS (typed .r/.d/.n/.i
codes, 597 MB .dta) is a separate, heavier loader — deferred (see module note).

Column selection (deterministic): among resolvable bounded items, keep those whose
overall missing rate lies in [rate_min, rate_max] (carry genuine refusal/DK signal
but are not mostly-missing), then take the ``n_cols`` whose rates are closest to
``target_rate`` — a representative moderate-missingness footprint that still leaves
a healthy complete-case yield for the X substrate. The very-high-refusal columns
(income) whose substrate needs imputation are a documented follow-up, not silently
included.

Fail loud (Coding Bible §1) on missing files, too few eligible columns, or too few
complete-case rows. Deterministic given ``seed``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from lacuna.survey.ess_codes import resolve_ess_column


@dataclass(frozen=True)
class MaskCorpus:
    """A plasmode-ready real-missingness corpus for the realism gate.

    Attributes:
        name: corpus label ("ESS", "NHANES").
        columns: the d selected, column-aligned item names.
        M_real: [n_real, d] uint8 real missing mask (1 = missing/sentinel).
        real_block_ids: [n_real] int block label per real row (e.g. country).
        X_complete: [n_cc, d] float64 complete-case real values (no NaN) — the
            substrate generators inject holes into.
        cc_block_ids: [n_cc] int block label per complete-case row.
        block_names: ordered block labels; index == block id.
        block_aware: True if >= 2 blocks exist (leave-block-out CV is meaningful).
    """

    name: str
    columns: Tuple[str, ...]
    M_real: np.ndarray
    real_block_ids: np.ndarray
    X_complete: np.ndarray
    cc_block_ids: np.ndarray
    block_names: Tuple[str, ...]
    block_aware: bool

    @property
    def d(self) -> int:
        return len(self.columns)

    @property
    def n_real(self) -> int:
        return self.M_real.shape[0]

    @property
    def n_complete(self) -> int:
        return self.X_complete.shape[0]


# =============================================================================
# Shared mechanics
# =============================================================================


def _missing_matrix(num: pd.DataFrame) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Resolve every numeric column to a (missing, value) pair via the ESS resolver.

    Returns:
        (cols, missing, values) where
          cols    : list of k resolvable column names,
          missing : [n, k] uint8, 1 where the cell is a sentinel or NaN,
          values  : [n, k] float64 raw answer (NaN where missing).
        Columns the resolver rejects (not clean bounded items) are dropped.

    Raises:
        ValueError: if no column resolves.
    """
    cols: List[str] = []
    miss_cols: List[np.ndarray] = []
    val_cols: List[np.ndarray] = []
    for c in num.columns:
        x = num[c].to_numpy(np.float64)
        codes = resolve_ess_column(x)
        if codes is None:
            continue
        sentinel = np.isin(x, list(codes.all_sentinels))
        missing = sentinel | np.isnan(x)
        vals = np.where(missing, np.nan, x)
        cols.append(str(c))
        miss_cols.append(missing.astype(np.uint8))
        val_cols.append(vals)
    if not cols:
        raise ValueError("No column resolved to a clean bounded item; cannot build corpus")
    return cols, np.stack(miss_cols, axis=1), np.stack(val_cols, axis=1)


def _select_columns(
    missing: np.ndarray,
    cols: Sequence[str],
    *,
    n_cols: int,
    rate_min: float,
    rate_max: float,
    target_rate: float,
) -> np.ndarray:
    """Pick column indices: rate in [rate_min, rate_max], closest to target_rate.

    Deterministic; ties broken by ascending column name. Returns int index array
    of length n_cols.

    Raises:
        ValueError: if fewer than n_cols columns are eligible.
    """
    if not 0.0 <= rate_min < rate_max <= 1.0:
        raise ValueError(f"need 0 <= rate_min < rate_max <= 1, got [{rate_min}, {rate_max}]")
    if not rate_min <= target_rate <= rate_max:
        raise ValueError(f"target_rate {target_rate} must lie in [{rate_min}, {rate_max}]")
    rates = missing.mean(axis=0)
    eligible = np.where((rates >= rate_min) & (rates <= rate_max))[0]
    if len(eligible) < n_cols:
        raise ValueError(
            f"only {len(eligible)} columns have missing rate in [{rate_min}, {rate_max}]; "
            f"need n_cols={n_cols}. Widen the band or lower n_cols."
        )
    # closest to the target footprint rate; deterministic tie-break by name.
    order = sorted(eligible, key=lambda j: (abs(rates[j] - target_rate), cols[j]))
    return np.array(order[:n_cols], dtype=np.int64)


def _project(
    name: str,
    cols: Sequence[str],
    missing: np.ndarray,
    values: np.ndarray,
    block_ids: np.ndarray,
    block_names: Sequence[str],
    *,
    n_cols: int,
    rate_min: float,
    rate_max: float,
    target_rate: float,
    min_complete_rows: int,
    max_real_rows: int,
    seed: int,
) -> MaskCorpus:
    """Select columns, build M_real, and carve the complete-case X substrate."""
    sel = _select_columns(
        missing, cols, n_cols=n_cols, rate_min=rate_min, rate_max=rate_max,
        target_rate=target_rate,
    )
    sel_cols = tuple(cols[j] for j in sel)
    M = missing[:, sel].astype(np.uint8)
    V = values[:, sel]

    complete_row = (M.sum(axis=1) == 0)
    n_complete = int(complete_row.sum())
    if n_complete < min_complete_rows:
        raise ValueError(
            f"{name}: only {n_complete} complete-case rows over the {n_cols} selected "
            f"columns; need >= {min_complete_rows}. Lower n_cols or relax the rate band."
        )
    X_complete = V[complete_row].astype(np.float64)
    cc_blocks = block_ids[complete_row].astype(np.int64)

    rng = np.random.default_rng(seed)
    if max_real_rows > 0 and M.shape[0] > max_real_rows:
        keep = rng.choice(M.shape[0], size=max_real_rows, replace=False)
        keep.sort()
        M = M[keep]
        real_blocks = block_ids[keep].astype(np.int64)
    else:
        real_blocks = block_ids.astype(np.int64)

    block_aware = len(set(block_names)) >= 2 and len(np.unique(real_blocks)) >= 2
    return MaskCorpus(
        name=name,
        columns=sel_cols,
        M_real=M,
        real_block_ids=real_blocks,
        X_complete=X_complete,
        cc_block_ids=cc_blocks,
        block_names=tuple(block_names),
        block_aware=block_aware,
    )


def _encode_blocks(labels: np.ndarray) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Map string/categorical block labels to dense int ids 0..B-1."""
    uniq = np.array(sorted({str(v) for v in labels}))
    name_to_id = {name: i for i, name in enumerate(uniq)}
    ids = np.array([name_to_id[str(v)] for v in labels], dtype=np.int64)
    return ids, tuple(uniq.tolist())


# =============================================================================
# ESS
# =============================================================================


def load_ess_corpus(
    path: Path = Path("/mnt/data/lacuna/rejected/ESS11e04_1.csv"),
    *,
    n_cols: int = 12,
    rate_min: float = 0.01,
    rate_max: float = 0.40,
    target_rate: float = 0.08,
    min_complete_rows: int = 500,
    max_real_rows: int = 20000,
    seed: int = 2026,
) -> MaskCorpus:
    """Load ESS round-11 real missingness, blocked by country (cntry).

    Raises:
        FileNotFoundError: if the ESS CSV is absent.
        ValueError: on too few eligible columns / complete-case rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ESS corpus not found: {path}")
    ess = pd.read_csv(path, low_memory=False)
    if "cntry" not in ess.columns:
        raise ValueError("ESS file missing 'cntry' column for block labels")
    block_ids, block_names = _encode_blocks(ess["cntry"].to_numpy())
    num = ess.select_dtypes(include=[np.number])
    cols, missing, values = _missing_matrix(num)
    return _project(
        "ESS", cols, missing, values, block_ids, block_names,
        n_cols=n_cols, rate_min=rate_min, rate_max=rate_max, target_rate=target_rate,
        min_complete_rows=min_complete_rows, max_real_rows=max_real_rows, seed=seed,
    )


# =============================================================================
# NHANES
# =============================================================================

_DEFAULT_NHANES = (
    Path("/mnt/data/lacuna/incoming/DPQ_J.xpt"),
    Path("/mnt/data/lacuna/incoming/DUQ_J.xpt"),
    Path("/mnt/data/lacuna/incoming/INQ_J.xpt"),
    Path("/mnt/data/lacuna/incoming/WHQ_J.xpt"),
)


def load_nhanes_corpus(
    paths: Sequence[Path] = _DEFAULT_NHANES,
    *,
    n_cols: int = 12,
    rate_min: float = 0.01,
    rate_max: float = 0.40,
    target_rate: float = 0.08,
    min_complete_rows: int = 500,
    max_real_rows: int = 20000,
    seed: int = 2026,
) -> MaskCorpus:
    """Load NHANES 2017-2018 (cycle J) questionnaire missingness, merged on SEQN.

    Single cycle ⇒ one row-block (``block_aware=False``); the gate falls back to a
    random K-fold and reports that leave-block-out was not possible.

    Raises:
        FileNotFoundError: if any .xpt is absent.
        ValueError: on too few eligible columns / complete-case rows.
    """
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"NHANES file not found: {p}")
        df = pd.read_sas(p)
        if "SEQN" not in df.columns:
            raise ValueError(f"NHANES file {p} missing SEQN key")
        frames.append(df.set_index("SEQN"))
    merged = pd.concat(frames, axis=1, join="outer")
    num = merged.select_dtypes(include=[np.number])
    cols, missing, values = _missing_matrix(num)
    block_ids = np.zeros(missing.shape[0], dtype=np.int64)  # one cycle = one block
    return _project(
        "NHANES", cols, missing, values, block_ids, ("J",),
        n_cols=n_cols, rate_min=rate_min, rate_max=rate_max, target_rate=target_rate,
        min_complete_rows=min_complete_rows, max_real_rows=max_real_rows, seed=seed,
    )
