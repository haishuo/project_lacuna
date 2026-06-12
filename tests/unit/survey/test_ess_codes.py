"""Tests for lacuna.survey.ess_codes — the width-based ESS sentinel resolution."""

import numpy as np
import pytest

from lacuna.survey.ess_codes import resolve_ess_column


class TestNormalCases:
    def test_two_digit_field_treats_single_digits_as_valid(self):
        # 0-10 happiness scale with 77/88 sentinels: 7/8/9 are REAL ANSWERS
        v = np.array([0, 3, 7, 8, 9, 10, 77, 88, 99] * 10, float)
        c = resolve_ess_column(v)
        assert c is not None
        assert 7 not in c.refusal and 8 not in c.dont_know and 9 not in c.no_answer
        assert 77 in c.refusal and 88 in c.dont_know and 99 in c.no_answer
        assert c.valid_max == 10

    def test_one_digit_field_uses_single_digit_sentinels(self):
        # 1-5 scale, no wide codes: 7=refusal, 8=DK per ESS 1-digit convention
        v = np.array([1, 2, 3, 4, 5, 7, 8, 9, 6] * 5, float)
        c = resolve_ess_column(v)
        assert c is not None
        assert c.refusal == frozenset({7}) and c.dont_know == frozenset({8})
        assert c.valid_max == 5

    def test_wide_presence_wins_over_single_digit(self):
        # 1-7 scale (pray): 7 = "never", wide 77/88 present => 7 must be VALID
        v = np.array([1, 2, 3, 4, 5, 6, 7, 77, 88] * 5, float)
        c = resolve_ess_column(v)
        assert c is not None
        assert 7 not in c.refusal
        assert c.valid_max == 7


class TestEdgeCases:
    def test_nan_only_returns_none(self):
        assert resolve_ess_column(np.array([np.nan, np.nan])) is None

    def test_empty_returns_none(self):
        assert resolve_ess_column(np.array([], dtype=float)) is None

    def test_all_sentinels_no_valid_returns_none(self):
        assert resolve_ess_column(np.array([77, 88, 99], float)) is None

    def test_three_digit_family(self):
        v = np.array(list(range(0, 25)) + [777, 888], float)
        c = resolve_ess_column(np.asarray(v))
        assert c is not None and 777 in c.refusal and 888 in c.dont_know


class TestRejections:
    def test_single_digit_field_per_convention(self):
        # values {1..9}, no wide codes: per the ESS width convention this IS a 1-digit field
        # (valid 1-5, sentinels 6-9) — resolvable, not ambiguous
        v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], float)
        c = resolve_ess_column(v)
        assert c is not None and c.refusal == frozenset({7}) and c.valid_max == 5

    def test_unbounded_column_rejected(self):
        # count variable (e.g. emplno) where 77 is a legitimate value: vmax > 30
        v = np.array([0, 5, 40, 77, 100, 44900 % 1000], float)
        assert resolve_ess_column(v) is None

    def test_non_integer_rejected(self):
        v = np.array([1.5, 2.3, 77], float)
        assert resolve_ess_column(v) is None

    def test_negative_values_rejected(self):
        v = np.array([-2, 1, 2, 77], float)
        assert resolve_ess_column(v) is None

    def test_no_sentinels_rejected(self):
        v = np.array([1, 2, 3, 4, 5], float)
        assert resolve_ess_column(v) is None
