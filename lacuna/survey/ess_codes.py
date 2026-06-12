"""ESS sentinel-code resolution for a single column (the age-77 trap, done right).

ESS missing codes are FIELD-WIDTH based: a 1-digit item uses 6/7/8/9
(not-applicable/refusal/don't-know/no-answer), a 2-digit item uses 66/77/88/99, etc.
Consequently single-digit 7/8/9 are VALID ANSWERS on any item whose scale reaches 6+
(e.g. 0-10 scales), and counting them as sentinels mislabels real answers as refusals.

Contract:
  resolve_ess_column(values) -> ColumnCodes | None
    values: 1-D float array of a column's non-NaN values (raw ESS integrated-file coding).
    Returns None if the column is not a clean bounded item under the rules below
    (caller must treat None as "exclude column", never guess).

Rules (conservative; ambiguous columns are REJECTED, never guessed):
  - If wide sentinel codes are present (66/77/88/99 or wider): the item is a >=2-digit field;
    sentinels = the wide families ONLY; single-digit 6/7/8/9 are valid answers.
  - Else if single-digit 6/7/8/9 are present AND all remaining values are <= 5: the item is a
    1-digit field; sentinels = {6,7,8,9}.
  - Else: reject (no resolvable sentinel structure).
  - In all cases the valid block must be non-negative integers with max <= 30 (bounded item;
    excludes counts/codes/continuous where sentinels collide with values).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

_WIDE_NAP = (66, 666, 6666)
_WIDE_REF = (77, 777, 7777)
_WIDE_DK = (88, 888, 8888)
_WIDE_NA = (99, 999, 9999)
_WIDE_ALL = _WIDE_NAP + _WIDE_REF + _WIDE_DK + _WIDE_NA


@dataclass(frozen=True)
class ColumnCodes:
    refusal: frozenset
    dont_know: frozenset
    no_answer: frozenset
    not_applicable: frozenset
    valid_max: float

    @property
    def all_sentinels(self) -> frozenset:
        return self.refusal | self.dont_know | self.no_answer | self.not_applicable


def resolve_ess_column(values: np.ndarray) -> Optional[ColumnCodes]:
    """Resolve sentinel codes for one ESS column, or None if not cleanly resolvable."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    has_wide = bool(np.isin(v, _WIDE_ALL).any())
    if has_wide:
        ref, dk, na, nap = frozenset(_WIDE_REF), frozenset(_WIDE_DK), frozenset(_WIDE_NA), frozenset(_WIDE_NAP)
        valid = v[~np.isin(v, _WIDE_ALL)]
    else:
        single = np.isin(v, (6, 7, 8, 9))
        if not single.any():
            return None
        valid = v[~single]
        if len(valid) and valid.max() > 5:
            return None  # 7/8/9 collide with the value range and no wide codes to disambiguate
        ref, dk, na, nap = frozenset({7}), frozenset({8}), frozenset({9}), frozenset({6})
    if len(valid) == 0:
        return None
    if not np.all(valid == valid.astype(int)):
        return None
    if valid.min() < 0 or valid.max() > 30:
        return None
    return ColumnCodes(refusal=ref, dont_know=dk, no_answer=na, not_applicable=nap,
                       valid_max=float(valid.max()))
