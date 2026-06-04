"""Tests for lacuna.survey.delta_bins."""

import pytest

from lacuna.survey.delta_bins import NUM_BINS, assign_delta_bin, bin_edges


# ---------- normal cases ----------

def test_zero_is_bin_zero():
    assert assign_delta_bin(0.0) == 0


def test_interior_values_map_to_expected_bins():
    assert assign_delta_bin(0.1) == 1
    assert assign_delta_bin(0.4) == 2
    assert assign_delta_bin(0.8) == 3
    assert assign_delta_bin(1.2) == 4
    assert assign_delta_bin(1.8) == 5
    assert assign_delta_bin(3.0) == 6


def test_large_delta_is_tail_bin():
    assert assign_delta_bin(100.0) == NUM_BINS - 1


# ---------- edge cases (exact boundaries: half-open (lower, upper]) ----------

def test_upper_edges_are_inclusive():
    # δ exactly on an upper edge belongs to the LOWER bin (half-open on the right).
    assert assign_delta_bin(0.25) == 1
    assert assign_delta_bin(0.5) == 2
    assert assign_delta_bin(1.0) == 3
    assert assign_delta_bin(1.5) == 4
    assert assign_delta_bin(2.0) == 5


def test_just_above_edge_moves_to_next_bin():
    assert assign_delta_bin(0.2500001) == 2
    assert assign_delta_bin(2.0000001) == 6


def test_tiny_positive_is_not_the_zero_bin():
    assert assign_delta_bin(1e-9) == 1


# ---------- failure cases ----------

def test_negative_delta_rejected():
    with pytest.raises(ValueError):
        assign_delta_bin(-0.1)


def test_nan_delta_rejected():
    with pytest.raises(ValueError):
        assign_delta_bin(float("nan"))


# ---------- metadata ----------

def test_bin_edges_metadata_is_consistent():
    meta = bin_edges()
    assert meta["num_bins"] == NUM_BINS
    assert meta["zero_bin_index"] == 0
    assert meta["tail_bin_index"] == NUM_BINS - 1
    assert meta["finite_upper_edges"] == [0.25, 0.5, 1.0, 1.5, 2.0]
    assert len(meta["labels"]) == NUM_BINS
