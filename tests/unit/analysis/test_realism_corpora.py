"""Tests for lacuna.analysis.realism_gate.corpora (pure helpers + guarded IO)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lacuna.analysis.realism_gate import corpora as co
from lacuna.analysis.realism_gate.corpora import MaskCorpus, load_ess_corpus


def test_encode_blocks():
    ids, names = co._encode_blocks(np.array(["DE", "FR", "DE", "AT"]))
    assert names == ("AT", "DE", "FR")
    assert list(ids) == [1, 2, 1, 0]


def test_missing_matrix_resolves_bounded_item():
    # 0-5 scale with single-digit sentinels 7/8/9, plus a plain-NaN cell.
    df = pd.DataFrame({
        "q1": [1, 2, 3, 7, 8, np.nan, 4, 5, 2, 1],   # 7,8 sentinels; NaN missing
        "q2": [0, 5, 2, 3, 9, 1, 0, 1, 2, 3],          # 9 sentinel on a 0-5 scale
    })
    cols, missing, values = co._missing_matrix(df)
    assert "q1" in cols and "q2" in cols
    j = cols.index("q1")
    # rows 3,4 (7,8) and row 5 (NaN) are missing
    assert missing[3, j] == 1 and missing[4, j] == 1 and missing[5, j] == 1
    assert np.isnan(values[3, j])
    k = cols.index("q2")
    assert missing[4, k] == 1   # the 9


def test_missing_matrix_raises_when_nothing_resolves():
    df = pd.DataFrame({"x": np.linspace(100, 200, 10)})  # not a bounded item
    with pytest.raises(ValueError):
        co._missing_matrix(df)


def test_select_columns_target_rate():
    # 4 columns with rates 0.02, 0.10, 0.20, 0.50; band [0.01,0.40], target 0.10.
    n = 1000
    missing = np.zeros((n, 4), dtype=np.uint8)
    for j, rate in enumerate([0.02, 0.10, 0.20, 0.50]):
        missing[: int(rate * n), j] = 1
    sel = co._select_columns(missing, ["a", "b", "c", "d"], n_cols=2,
                             rate_min=0.01, rate_max=0.40, target_rate=0.10)
    # closest to 0.10: col1 (|0.00|) then col0 (|0.08|); col2 is |0.10|, col3 out of band
    assert set(sel.tolist()) == {0, 1}


def test_select_columns_too_few_eligible_raises():
    missing = np.zeros((100, 2), dtype=np.uint8)
    missing[:50, 0] = 1   # rate 0.5 (out of band)
    with pytest.raises(ValueError):
        co._select_columns(missing, ["a", "b"], n_cols=2,
                           rate_min=0.01, rate_max=0.40, target_rate=0.10)


def test_select_columns_bad_band_raises():
    missing = np.zeros((100, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        co._select_columns(missing, ["a", "b"], n_cols=1,
                           rate_min=0.5, rate_max=0.4, target_rate=0.45)


def test_maskcorpus_properties():
    M = np.zeros((5, 3), dtype=np.uint8)
    X = np.zeros((4, 3))
    c = MaskCorpus("T", ("a", "b", "c"), M, np.zeros(5, int), X, np.zeros(4, int), ("0",), False)
    assert c.d == 3 and c.n_real == 5 and c.n_complete == 4


_ESS = Path("/mnt/data/lacuna/rejected/ESS11e04_1.csv")


@pytest.mark.skipif(not _ESS.exists(), reason="ESS corpus not on disk")
def test_load_ess_corpus_smoke():
    c = load_ess_corpus(n_cols=8, max_real_rows=5000)
    assert c.d == 8
    assert c.n_real <= 5000
    assert c.block_aware is True
    assert not np.isnan(c.X_complete).any()       # substrate is complete
    assert set(np.unique(c.M_real)).issubset({0, 1})
