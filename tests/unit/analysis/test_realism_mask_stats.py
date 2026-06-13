"""Tests for lacuna.analysis.realism_gate.mask_stats."""

import numpy as np
import pytest

from lacuna.analysis.realism_gate import mask_stats as ms


# --- check_mask: normal / edge / failure -----------------------------------

def test_check_mask_accepts_binary():
    M = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    out = ms.check_mask(M)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2)


def test_check_mask_rejects_non_2d():
    with pytest.raises(ValueError):
        ms.check_mask(np.array([0, 1, 0]))


def test_check_mask_rejects_empty():
    with pytest.raises(ValueError):
        ms.check_mask(np.zeros((0, 3)))


def test_check_mask_rejects_non_binary():
    with pytest.raises(ValueError):
        ms.check_mask(np.array([[0, 2], [1, 0]]))


def test_check_mask_rejects_nan():
    with pytest.raises(ValueError):
        ms.check_mask(np.array([[0.0, np.nan], [1.0, 0.0]]))


# --- longest_run / row_features --------------------------------------------

def test_longest_run_per_row():
    M = np.array([
        [1, 1, 0, 1],   # longest run 2
        [0, 0, 0, 0],   # 0
        [1, 1, 1, 1],   # 4
    ], dtype=np.uint8)
    assert list(ms.longest_run_per_row(M)) == [2, 0, 4]


def test_row_features_shape_and_summaries():
    M = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    F = ms.row_features(M)
    assert F.shape == (2, 3 + 2)
    # row 0: count 2, longest run 1; row 1: count 0, longest 0
    assert F[0, -2] == 2 and F[0, -1] == 1
    assert F[1, -2] == 0 and F[1, -1] == 0


def test_feature_names_align():
    names = ms.feature_names(3)
    assert names == ("col_0", "col_1", "col_2", "row_missing_count", "longest_run")


# --- footprint statistics ---------------------------------------------------

def test_column_rates():
    M = np.array([[1, 0], [1, 0], [0, 0], [0, 0]], dtype=np.uint8)
    assert np.allclose(ms.column_rates(M), [0.5, 0.0])


def test_comissingness_vector_length_and_perfect_corr():
    # two perfectly co-missing columns + one independent -> d=3 -> 3 pairs
    M = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    v = ms.comissingness_vector(M)
    assert v.shape == (3,)
    # cols 0,1 perfectly correlated
    assert v[0] == pytest.approx(1.0)


def test_comissingness_requires_two_cols():
    with pytest.raises(ValueError):
        ms.comissingness_vector(np.array([[1], [0]], dtype=np.uint8))


def test_run_length_hist_normalised():
    M = np.array([[1, 1, 0, 1]], dtype=np.uint8)  # runs: length2, length1
    h = ms.run_length_hist(M, max_len=4)
    assert h.sum() == pytest.approx(1.0)
    assert h[0] == pytest.approx(0.5) and h[1] == pytest.approx(0.5)


def test_run_length_hist_empty_returns_zeros():
    M = np.zeros((3, 4), dtype=np.uint8)
    h = ms.run_length_hist(M, max_len=4)
    assert h.sum() == 0.0


def test_run_length_hist_bad_maxlen():
    with pytest.raises(ValueError):
        ms.run_length_hist(np.ones((2, 2), dtype=np.uint8), max_len=0)


def test_row_count_hist_normalised():
    M = np.array([[1, 1], [0, 0], [1, 0]], dtype=np.uint8)
    h = ms.row_count_hist(M, n_bins=3)
    assert h.sum() == pytest.approx(1.0)


def test_overall_rate():
    assert ms.overall_rate(np.array([[1, 0], [0, 0]], dtype=np.uint8)) == pytest.approx(0.25)
