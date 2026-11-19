"""Comprehensive tests for transform_tabular — translated from WL test suite.

Every test uses explicit expected results and pure functions (lambda),
matching the Wolfram Language TransformTabular tests one-to-one.
"""

import functools
import math
import operator
import statistics
import warnings
from itertools import accumulate

import numpy as np
import pandas as pd
import pytest

from transform_tabular import (
    ColumnwiseThread,
    ColumnwiseValue,
    transform_columns,
    transform_tabular,
)


# ------------------------------------------------------------------ helpers

def _rank_order(col):
    """Equivalent of WL Ordering[Ordering[#]]."""
    indexed = sorted(range(len(col)), key=lambda i: col[i])
    ranks = [0] * len(col)
    for r, i in enumerate(indexed, 1):
        ranks[i] = r
    return ranks


def _rescale(col):
    """Rescale to [0, 1]."""
    mn, mx = min(col), max(col)
    return [(x - mn) / (mx - mn) for x in col]


def _standardize(col):
    """Standardize (z-score)."""
    m = sum(col) / len(col)
    s = statistics.stdev(col)
    return [(x - m) / s for x in col]


def _moving_avg_prepend(col):
    """MovingAverage[Prepend[#, First[#]], 2]."""
    p = [col[0]] + list(col)
    return [(p[i] + p[i + 1]) / 2 for i in range(len(col))]


def assert_df(result, expected_records):
    """Compare DataFrame against expected list-of-dicts."""
    expected = pd.DataFrame(expected_records)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        atol=1e-10,
    )


# ================================================================== Setup

@pytest.fixture
def tab1():
    return pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})

@pytest.fixture
def tab2():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

@pytest.fixture
def tab_single():
    return pd.DataFrame({"a": [42], "b": [99]})

@pytest.fixture
def tab_one_col():
    return pd.DataFrame({"val": [10, 20, 30]})

@pytest.fixture
def tab_str():
    return pd.DataFrame({"name": ["alice", "bob", "carol"], "score": [85, 92, 78]})

@pytest.fixture
def tab_neg():
    return pd.DataFrame({"a": [-3, 0, 7], "b": [5, -1, 2]})

@pytest.fixture
def tab_float():
    return pd.DataFrame({"p": [1.5, 4.5], "q": [2.5, 5.5], "r": [3.5, 6.5]})

@pytest.fixture
def tab_large():
    return pd.DataFrame({
        "a": [10, 11, 12, 13],
        "b": [20, 21, 22, 23],
        "c": [30, 31, 32, 33],
        "d": [40, 41, 42, 43],
        "e": [50, 51, 52, 53],
    })


# ============================================ 1. Basic Examples

class TestBasicExamples:

    def test_increment_all(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x + 1),
            [{"a": 2, "b": 11, "c": 101},
             {"a": 3, "b": 21, "c": 201},
             {"a": 4, "b": 31, "c": 301}],
        )

    def test_double_all(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: 2 * x),
            [{"a": 2, "b": 20, "c": 200},
             {"a": 4, "b": 40, "c": 400},
             {"a": 6, "b": 60, "c": 600}],
        )

    def test_square_all(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x ** 2),
            [{"a": 1, "b": 100, "c": 10000},
             {"a": 4, "b": 400, "c": 40000},
             {"a": 9, "b": 900, "c": 90000}],
        )

    def test_abs_all(self, tab_neg):
        assert_df(
            transform_tabular(tab_neg, lambda x: abs(x)),
            [{"a": 3, "b": 5},
             {"a": 0, "b": 1},
             {"a": 7, "b": 2}],
        )

    def test_identity_all(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x),
            [{"a": 1, "b": 10, "c": 100},
             {"a": 2, "b": 20, "c": 200},
             {"a": 3, "b": 30, "c": 300}],
        )

    def test_negate_all(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: -x),
            [{"a": -1, "b": -10, "c": -100},
             {"a": -2, "b": -20, "c": -200},
             {"a": -3, "b": -30, "c": -300}],
        )

    def test_constant_zero_all(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: 0),
            [{"a": 0, "b": 0, "c": 0},
             {"a": 0, "b": 0, "c": 0},
             {"a": 0, "b": 0, "c": 0}],
        )

    def test_add_large_constant(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x + 1000),
            [{"a": 1001, "b": 1010, "c": 1100},
             {"a": 1002, "b": 1020, "c": 1200},
             {"a": 1003, "b": 1030, "c": 1300}],
        )

    def test_square_floats(self, tab2):
        assert_df(
            transform_tabular(tab2, lambda x: x ** 2),
            [{"x": 1.0, "y": 16.0},
             {"x": 4.0, "y": 25.0},
             {"x": 9.0, "y": 36.0}],
        )

    def test_modular_arithmetic(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x % 3, columns=["a"]),
            [{"a": 1, "b": 10, "c": 100},
             {"a": 2, "b": 20, "c": 200},
             {"a": 0, "b": 30, "c": 300}],
        )

    def test_to_upper_case_column(self, tab_str):
        assert_df(
            transform_tabular(tab_str, lambda x: x.upper(), columns=["name"]),
            [{"name": "ALICE", "score": 85},
             {"name": "BOB", "score": 92},
             {"name": "CAROL", "score": 78}],
        )

    def test_string_length_column(self, tab_str):
        assert_df(
            transform_tabular(tab_str, lambda x: len(x), columns=["name"]),
            [{"name": 5, "score": 85},
             {"name": 3, "score": 92},
             {"name": 5, "score": 78}],
        )


# ============================================ 2. Column Specification

class TestColumnSpecification:

    def test_list_of_names(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x + 1, columns=["a", "c"]),
            [{"a": 2, "b": 10, "c": 101},
             {"a": 3, "b": 20, "c": 201},
             {"a": 4, "b": 30, "c": 301}],
        )

    def test_single_name(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x * 10, columns="b"),
            [{"a": 1, "b": 100, "c": 100},
             {"a": 2, "b": 200, "c": 200},
             {"a": 3, "b": 300, "c": 300}],
        )

    def test_integer_indices(self, tab1):
        # WL {1,3} (1-indexed) → Python [0,2] (0-indexed)
        assert_df(
            transform_tabular(tab1, lambda x: x + 1, columns=[0, 2]),
            [{"a": 2, "b": 10, "c": 101},
             {"a": 3, "b": 20, "c": 201},
             {"a": 4, "b": 30, "c": 301}],
        )

    def test_single_integer(self, tab1):
        # WL 2 → Python 1
        assert_df(
            transform_tabular(tab1, lambda x: x + 1, columns=1),
            [{"a": 1, "b": 11, "c": 100},
             {"a": 2, "b": 21, "c": 200},
             {"a": 3, "b": 31, "c": 300}],
        )

    def test_span_first_two(self, tab1):
        # WL 1;;2 → Python slice(0, 2)
        assert_df(
            transform_tabular(tab1, lambda x: x * 0, columns=slice(0, 2)),
            [{"a": 0, "b": 0, "c": 100},
             {"a": 0, "b": 0, "c": 200},
             {"a": 0, "b": 0, "c": 300}],
        )

    def test_span_last_two(self, tab1):
        # WL 2;;3 → Python slice(1, 3)
        assert_df(
            transform_tabular(tab1, lambda x: x + 1, columns=slice(1, 3)),
            [{"a": 1, "b": 11, "c": 101},
             {"a": 2, "b": 21, "c": 201},
             {"a": 3, "b": 31, "c": 301}],
        )

    def test_all_columns(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x + 1, columns=None),
            [{"a": 2, "b": 11, "c": 101},
             {"a": 3, "b": 21, "c": 201},
             {"a": 4, "b": 31, "c": 301}],
        )

    def test_negative_index(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x + 1, columns=-1),
            [{"a": 1, "b": 10, "c": 101},
             {"a": 2, "b": 20, "c": 201},
             {"a": 3, "b": 30, "c": 301}],
        )

    def test_span_with_step(self, tab_large):
        # WL 1;;-1;;2 → Python slice(None, None, 2), selects a, c, e
        assert_df(
            transform_tabular(tab_large, lambda x: x + 1, columns=slice(None, None, 2)),
            [{"a": 11, "b": 20, "c": 31, "d": 40, "e": 51},
             {"a": 12, "b": 21, "c": 32, "d": 41, "e": 52},
             {"a": 13, "b": 22, "c": 33, "d": 42, "e": 53},
             {"a": 14, "b": 23, "c": 34, "d": 43, "e": 54}],
        )

    def test_single_column_tabular(self, tab_one_col):
        assert_df(
            transform_tabular(tab_one_col, lambda x: x * 2),
            [{"val": 20}, {"val": 40}, {"val": 60}],
        )

    def test_first_column_only(self, tab_large):
        assert_df(
            transform_tabular(tab_large, lambda x: x + 100, columns=["a"]),
            [{"a": 110, "b": 20, "c": 30, "d": 40, "e": 50},
             {"a": 111, "b": 21, "c": 31, "d": 41, "e": 51},
             {"a": 112, "b": 22, "c": 32, "d": 42, "e": 52},
             {"a": 113, "b": 23, "c": 33, "d": 43, "e": 53}],
        )

    def test_non_adjacent_columns(self, tab_large):
        assert_df(
            transform_tabular(tab_large, lambda x: x + 100, columns=["a", "e"]),
            [{"a": 110, "b": 20, "c": 30, "d": 40, "e": 150},
             {"a": 111, "b": 21, "c": 31, "d": 41, "e": 151},
             {"a": 112, "b": 22, "c": 32, "d": 42, "e": 152},
             {"a": 113, "b": 23, "c": 33, "d": 43, "e": 153}],
        )


# ============================================ 3. Operator Forms

class TestOperatorForms:

    def test_fun_only(self, tab1):
        assert_df(
            transform_tabular(lambda x: x + 1)(tab1),
            [{"a": 2, "b": 11, "c": 101},
             {"a": 3, "b": 21, "c": 201},
             {"a": 4, "b": 31, "c": 301}],
        )

    def test_fun_and_col_list(self, tab1):
        assert_df(
            transform_tabular(lambda x: x + 1, ["a", "c"])(tab1),
            [{"a": 2, "b": 10, "c": 101},
             {"a": 3, "b": 20, "c": 201},
             {"a": 4, "b": 30, "c": 301}],
        )

    def test_fun_and_span(self, tab1):
        assert_df(
            transform_tabular(lambda x: x * 2, slice(0, 2))(tab1),
            [{"a": 2, "b": 20, "c": 100},
             {"a": 4, "b": 40, "c": 200},
             {"a": 6, "b": 60, "c": 300}],
        )

    def test_fun_and_integer(self, tab1):
        assert_df(
            transform_tabular(lambda x: x * 2, 0)(tab1),
            [{"a": 2, "b": 10, "c": 100},
             {"a": 4, "b": 20, "c": 200},
             {"a": 6, "b": 30, "c": 300}],
        )

    def test_fun_and_all(self, tab1):
        assert_df(
            transform_tabular(lambda x: x * 2, None)(tab1),
            [{"a": 2, "b": 20, "c": 200},
             {"a": 4, "b": 40, "c": 400},
             {"a": 6, "b": 60, "c": 600}],
        )

    def test_abs_function(self, tab_neg):
        assert_df(
            transform_tabular(lambda x: abs(x))(tab_neg),
            [{"a": 3, "b": 5},
             {"a": 0, "b": 1},
             {"a": 7, "b": 2}],
        )

    def test_chaining(self, tab1):
        # increment all, then double all
        assert_df(
            transform_tabular(lambda x: x * 2)(
                transform_tabular(lambda x: x + 1)(tab1)
            ),
            [{"a": 4, "b": 22, "c": 202},
             {"a": 6, "b": 42, "c": 402},
             {"a": 8, "b": 62, "c": 602}],
        )


# ============================================ 4. ColumnwiseValue

class TestColumnwiseValue:

    def test_subtract_mean(self, tab1):
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab1, lambda x: x - cv),
            [{"a": -1.0, "b": -10.0, "c": -100.0},
             {"a":  0.0, "b":   0.0, "c":    0.0},
             {"a":  1.0, "b":  10.0, "c":  100.0}],
        )

    def test_divide_by_max(self, tab2):
        cv = ColumnwiseValue(lambda col: max(col))
        assert_df(
            transform_tabular(tab2, lambda x: x / cv),
            [{"x": 1 / 3, "y": 4 / 6},
             {"x": 2 / 3, "y": 5 / 6},
             {"x": 1.0,   "y": 1.0}],
        )

    def test_subtract_min(self, tab1):
        cv = ColumnwiseValue(lambda col: min(col))
        assert_df(
            transform_tabular(tab1, lambda x: x - cv),
            [{"a": 0, "b":  0, "c":   0},
             {"a": 1, "b": 10, "c": 100},
             {"a": 2, "b": 20, "c": 200}],
        )

    def test_min_max_normalization(self, tab1):
        cv_min = ColumnwiseValue(lambda col: min(col))
        cv_max = ColumnwiseValue(lambda col: max(col))
        assert_df(
            transform_tabular(tab1, lambda x: (x - cv_min) / (cv_max - cv_min)),
            [{"a": 0.0, "b": 0.0, "c": 0.0},
             {"a": 0.5, "b": 0.5, "c": 0.5},
             {"a": 1.0, "b": 1.0, "c": 1.0}],
        )

    def test_z_score(self, tab2):
        cv_mean = ColumnwiseValue(lambda col: sum(col) / len(col))
        cv_std = ColumnwiseValue(lambda col: statistics.stdev(col))
        assert_df(
            transform_tabular(tab2, lambda x: (x - cv_mean) / cv_std),
            [{"x": -1.0, "y": -1.0},
             {"x":  0.0, "y":  0.0},
             {"x":  1.0, "y":  1.0}],
        )

    def test_selected_columns(self, tab1):
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab1, lambda x: x - cv, columns=["a", "b"]),
            [{"a": -1.0, "b": -10.0, "c": 100},
             {"a":  0.0, "b":   0.0, "c": 200},
             {"a":  1.0, "b":  10.0, "c": 300}],
        )

    def test_divide_by_total(self, tab1):
        cv = ColumnwiseValue(lambda col: sum(col))
        assert_df(
            transform_tabular(tab1, lambda x: x / cv, columns=["a"]),
            [{"a": 1 / 6, "b": 10, "c": 100},
             {"a": 2 / 6, "b": 20, "c": 200},
             {"a": 3 / 6, "b": 30, "c": 300}],
        )

    def test_divide_by_length(self, tab1):
        cv = ColumnwiseValue(lambda col: len(col))
        assert_df(
            transform_tabular(tab1, lambda x: x / cv, columns=["a"]),
            [{"a": 1 / 3, "b": 10, "c": 100},
             {"a": 2 / 3, "b": 20, "c": 200},
             {"a": 3 / 3, "b": 30, "c": 300}],
        )

    def test_subtract_median(self, tab1):
        cv = ColumnwiseValue(lambda col: statistics.median(col))
        assert_df(
            transform_tabular(tab1, lambda x: x - cv),
            [{"a": -1, "b": -10, "c": -100},
             {"a":  0, "b":   0, "c":    0},
             {"a":  1, "b":  10, "c":  100}],
        )

    def test_single_row(self, tab_single):
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab_single, lambda x: x - cv),
            [{"a": 0.0, "b": 0.0}],
        )

    def test_operator_form(self, tab1):
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(lambda x: x - cv)(tab1),
            [{"a": -1.0, "b": -10.0, "c": -100.0},
             {"a":  0.0, "b":   0.0, "c":    0.0},
             {"a":  1.0, "b":  10.0, "c":  100.0}],
        )

    def test_boolean_above_mean(self, tab1):
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab1, lambda x: x > cv, columns=["a"]),
            [{"a": False, "b": 10, "c": 100},
             {"a": False, "b": 20, "c": 200},
             {"a": True,  "b": 30, "c": 300}],
        )

    def test_subtract_first(self, tab1):
        cv = ColumnwiseValue(lambda col: col[0])
        assert_df(
            transform_tabular(tab1, lambda x: x - cv, columns=["a"]),
            [{"a": 0, "b": 10, "c": 100},
             {"a": 1, "b": 20, "c": 200},
             {"a": 2, "b": 30, "c": 300}],
        )

    def test_subtract_last(self, tab1):
        cv = ColumnwiseValue(lambda col: col[-1])
        assert_df(
            transform_tabular(tab1, lambda x: x - cv, columns=["a"]),
            [{"a": -2, "b": 10, "c": 100},
             {"a": -1, "b": 20, "c": 200},
             {"a":  0, "b": 30, "c": 300}],
        )

    def test_divide_by_variance(self, tab2):
        cv = ColumnwiseValue(lambda col: statistics.variance(col))
        assert_df(
            transform_tabular(tab2, lambda x: x / cv, columns=["x"]),
            [{"x": 1.0, "y": 4.0},
             {"x": 2.0, "y": 5.0},
             {"x": 3.0, "y": 6.0}],
        )

    def test_multiple_cv_instances(self, tab2):
        # (x + mean) * stddev on "x"; mean=2, stddev=1 → x+2
        cv_m = ColumnwiseValue(lambda col: sum(col) / len(col))
        cv_s = ColumnwiseValue(lambda col: statistics.stdev(col))
        assert_df(
            transform_tabular(tab2, lambda x: (x + cv_m) * cv_s, columns=["x"]),
            [{"x": 3.0, "y": 4.0},
             {"x": 4.0, "y": 5.0},
             {"x": 5.0, "y": 6.0}],
        )

    def test_three_cv_instances(self, tab1):
        # min-max normalization on column "b" only
        cv_min = ColumnwiseValue(lambda col: min(col))
        cv_max = ColumnwiseValue(lambda col: max(col))
        assert_df(
            transform_tabular(
                tab1, lambda x: (x - cv_min) / (cv_max - cv_min), columns=["b"]
            ),
            [{"a": 1, "b": 0.0, "c": 100},
             {"a": 2, "b": 0.5, "c": 200},
             {"a": 3, "b": 1.0, "c": 300}],
        )


# ============================================ 5. ColumnwiseThread

class TestColumnwiseThread:

    def test_sort_already_sorted(self, tab2):
        ct = ColumnwiseThread(lambda col: sorted(col))
        assert_df(
            transform_tabular(tab2, lambda x: ct, columns=["x"]),
            [{"x": 1.0, "y": 4.0},
             {"x": 2.0, "y": 5.0},
             {"x": 3.0, "y": 6.0}],
        )

    def test_reverse_sort(self, tab1):
        ct = ColumnwiseThread(lambda col: sorted(col, reverse=True))
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 3, "b": 10, "c": 100},
             {"a": 2, "b": 20, "c": 200},
             {"a": 1, "b": 30, "c": 300}],
        )

    def test_sort_unsorted(self, tab_neg):
        ct = ColumnwiseThread(lambda col: sorted(col))
        assert_df(
            transform_tabular(tab_neg, lambda x: ct, columns=["a"]),
            [{"a": -3, "b": 5},
             {"a":  0, "b": -1},
             {"a":  7, "b": 2}],
        )

    def test_sort_all_columns(self, tab_neg):
        ct = ColumnwiseThread(lambda col: sorted(col))
        assert_df(
            transform_tabular(tab_neg, lambda x: ct),
            [{"a": -3, "b": -1},
             {"a":  0, "b":  2},
             {"a":  7, "b":  5}],
        )

    def test_rank_order(self, tab2):
        ct = ColumnwiseThread(lambda col: _rank_order(col))
        assert_df(
            transform_tabular(tab2, lambda x: ct, columns=["y"]),
            [{"x": 1.0, "y": 1},
             {"x": 2.0, "y": 2},
             {"x": 3.0, "y": 3}],
        )

    def test_cumulative_sum(self, tab1):
        ct = ColumnwiseThread(lambda col: list(accumulate(col)))
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 1, "b": 10, "c": 100},
             {"a": 3, "b": 20, "c": 200},
             {"a": 6, "b": 30, "c": 300}],
        )

    def test_differences(self, tab1):
        ct = ColumnwiseThread(
            lambda col: [0] + [col[i + 1] - col[i] for i in range(len(col) - 1)]
        )
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 0, "b": 10, "c": 100},
             {"a": 1, "b": 20, "c": 200},
             {"a": 1, "b": 30, "c": 300}],
        )

    def test_normalize_by_total(self, tab1):
        ct = ColumnwiseThread(lambda col: [x / sum(col) for x in col])
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 1 / 6, "b": 10, "c": 100},
             {"a": 2 / 6, "b": 20, "c": 200},
             {"a": 3 / 6, "b": 30, "c": 300}],
        )

    def test_standardize(self, tab2):
        ct = ColumnwiseThread(lambda col: _standardize(col))
        assert_df(
            transform_tabular(tab2, lambda x: ct, columns=["x"]),
            [{"x": -1.0, "y": 4.0},
             {"x":  0.0, "y": 5.0},
             {"x":  1.0, "y": 6.0}],
        )

    def test_rescale(self, tab1):
        ct = ColumnwiseThread(lambda col: _rescale(col))
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 0.0, "b": 10, "c": 100},
             {"a": 0.5, "b": 20, "c": 200},
             {"a": 1.0, "b": 30, "c": 300}],
        )

    def test_reverse(self, tab1):
        ct = ColumnwiseThread(lambda col: list(reversed(col)))
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 3, "b": 10, "c": 100},
             {"a": 2, "b": 20, "c": 200},
             {"a": 1, "b": 30, "c": 300}],
        )

    def test_rotate_left(self, tab1):
        ct = ColumnwiseThread(lambda col: col[1:] + [col[0]])
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 2, "b": 10, "c": 100},
             {"a": 3, "b": 20, "c": 200},
             {"a": 1, "b": 30, "c": 300}],
        )

    def test_accumulate_multiple_cols(self, tab1):
        ct = ColumnwiseThread(lambda col: list(accumulate(col)))
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a", "b"]),
            [{"a": 1, "b": 10,  "c": 100},
             {"a": 3, "b": 30,  "c": 200},
             {"a": 6, "b": 60,  "c": 300}],
        )

    def test_operator_form(self, tab1):
        ct = ColumnwiseThread(lambda col: sorted(col))
        assert_df(
            transform_tabular(lambda x: ct, ["a"])(tab1),
            [{"a": 1, "b": 10, "c": 100},
             {"a": 2, "b": 20, "c": 200},
             {"a": 3, "b": 30, "c": 300}],
        )

    def test_single_column_tabular(self, tab_one_col):
        ct = ColumnwiseThread(lambda col: list(accumulate(col)))
        assert_df(
            transform_tabular(tab_one_col, lambda x: ct),
            [{"val": 10}, {"val": 30}, {"val": 60}],
        )

    def test_convert_to_float(self, tab1):
        ct = ColumnwiseThread(lambda col: [float(x) for x in col])
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 1.0, "b": 10, "c": 100},
             {"a": 2.0, "b": 20, "c": 200},
             {"a": 3.0, "b": 30, "c": 300}],
        )

    def test_moving_average(self, tab1):
        ct = ColumnwiseThread(lambda col: _moving_avg_prepend(col))
        assert_df(
            transform_tabular(tab1, lambda x: ct, columns=["a"]),
            [{"a": 1.0, "b": 10, "c": 100},
             {"a": 1.5, "b": 20, "c": 200},
             {"a": 2.5, "b": 30, "c": 300}],
        )


# ============================================ 6. Combined CV + CT

class TestCombinedCVCT:

    def test_sort_plus_mean(self, tab1):
        # sorted "a" = [1,2,3], mean = 2 → [3, 4, 5]
        ct = ColumnwiseThread(lambda col: sorted(col))
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab1, lambda x: ct + cv, columns=["a"]),
            [{"a": 3.0, "b": 10, "c": 100},
             {"a": 4.0, "b": 20, "c": 200},
             {"a": 5.0, "b": 30, "c": 300}],
        )

    def test_sort_minus_mean(self, tab1):
        ct = ColumnwiseThread(lambda col: sorted(col))
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab1, lambda x: ct - cv, columns=["a"]),
            [{"a": -1.0, "b": 10, "c": 100},
             {"a":  0.0, "b": 20, "c": 200},
             {"a":  1.0, "b": 30, "c": 300}],
        )

    def test_sort_divide_by_max(self, tab1):
        ct = ColumnwiseThread(lambda col: sorted(col))
        cv = ColumnwiseValue(lambda col: max(col))
        assert_df(
            transform_tabular(tab1, lambda x: ct / cv, columns=["a"]),
            [{"a": 1 / 3, "b": 10, "c": 100},
             {"a": 2 / 3, "b": 20, "c": 200},
             {"a": 1.0,   "b": 30, "c": 300}],
        )

    def test_all_columns(self, tab2):
        ct = ColumnwiseThread(lambda col: sorted(col))
        cv = ColumnwiseValue(lambda col: min(col))
        assert_df(
            transform_tabular(tab2, lambda x: ct - cv),
            [{"x": 0.0, "y": 0.0},
             {"x": 1.0, "y": 1.0},
             {"x": 2.0, "y": 2.0}],
        )

    def test_min_max_with_sort(self, tab1):
        ct = ColumnwiseThread(lambda col: sorted(col))
        cv_min = ColumnwiseValue(lambda col: min(col))
        cv_max = ColumnwiseValue(lambda col: max(col))
        assert_df(
            transform_tabular(
                tab1, lambda x: (ct - cv_min) / (cv_max - cv_min), columns=["a"]
            ),
            [{"a": 0.0, "b": 10, "c": 100},
             {"a": 0.5, "b": 20, "c": 200},
             {"a": 1.0, "b": 30, "c": 300}],
        )

    def test_operator_form(self, tab1):
        ct = ColumnwiseThread(lambda col: sorted(col))
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(lambda x: ct + cv, ["a"])(tab1),
            [{"a": 3.0, "b": 10, "c": 100},
             {"a": 4.0, "b": 20, "c": 200},
             {"a": 5.0, "b": 30, "c": 300}],
        )

    def test_accumulate_minus_mean(self, tab1):
        # cumsum "a" = [1,3,6], mean = 2 → [-1, 1, 4]
        ct = ColumnwiseThread(lambda col: list(accumulate(col)))
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab1, lambda x: ct - cv, columns=["a"]),
            [{"a": -1.0, "b": 10, "c": 100},
             {"a":  1.0, "b": 20, "c": 200},
             {"a":  4.0, "b": 30, "c": 300}],
        )

    def test_reverse_divide_by_total(self, tab1):
        # reverse "a" = [3,2,1], total = 6 → [1/2, 1/3, 1/6]
        ct = ColumnwiseThread(lambda col: list(reversed(col)))
        cv = ColumnwiseValue(lambda col: sum(col))
        assert_df(
            transform_tabular(tab1, lambda x: ct / cv, columns=["a"]),
            [{"a": 3 / 6, "b": 10, "c": 100},
             {"a": 2 / 6, "b": 20, "c": 200},
             {"a": 1 / 6, "b": 30, "c": 300}],
        )


# ============================================ 7. Error and Edge Cases

class TestErrorAndEdgeCases:

    def test_mixed_composition(self, tab1):
        """ColumnwiseThread wrapping ColumnwiseValue should raise."""
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        ct = ColumnwiseThread(lambda col: cv)
        with pytest.raises(ValueError, match="Mixed compositions"):
            transform_tabular(tab1, lambda x: ct, columns=["a"])

    def test_single_row(self, tab_single):
        assert_df(
            transform_tabular(tab_single, lambda x: x * 3),
            [{"a": 126, "b": 297}],
        )

    def test_single_column(self, tab_one_col):
        assert_df(
            transform_tabular(tab_one_col, lambda x: x + 5),
            [{"val": 15}, {"val": 25}, {"val": 35}],
        )

    def test_composed_function(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: abs(x - 2), columns=["a"]),
            [{"a": 1, "b": 10, "c": 100},
             {"a": 0, "b": 20, "c": 200},
             {"a": 1, "b": 30, "c": 300}],
        )

    def test_conditional_function(self, tab1):
        assert_df(
            transform_tabular(tab1, lambda x: x ** 2 if x > 1 else x, columns=["a"]),
            [{"a": 1, "b": 10, "c": 100},
             {"a": 4, "b": 20, "c": 200},
             {"a": 9, "b": 30, "c": 300}],
        )

    def test_inverse_operations(self, tab1):
        assert_df(
            transform_tabular(
                transform_tabular(tab1, lambda x: x * 7),
                lambda x: x / 7,
            ),
            [{"a": 1.0, "b": 10.0, "c": 100.0},
             {"a": 2.0, "b": 20.0, "c": 200.0},
             {"a": 3.0, "b": 30.0, "c": 300.0}],
        )

    def test_add_subtract_cancels(self, tab1):
        assert_df(
            transform_tabular(
                transform_tabular(tab1, lambda x: x + 99),
                lambda x: x - 99,
            ),
            [{"a": 1, "b": 10, "c": 100},
             {"a": 2, "b": 20, "c": 200},
             {"a": 3, "b": 30, "c": 300}],
        )

    def test_square_sqrt_cancels(self, tab2):
        assert_df(
            transform_tabular(
                transform_tabular(tab2, lambda x: x ** 2),
                lambda x: math.sqrt(x),
            ),
            [{"x": 1.0, "y": 4.0},
             {"x": 2.0, "y": 5.0},
             {"x": 3.0, "y": 6.0}],
        )

    def test_cv_single_row(self, tab_single):
        # mean of single value is itself: 42 + 42 = 84
        cv = ColumnwiseValue(lambda col: sum(col) / len(col))
        assert_df(
            transform_tabular(tab_single, lambda x: x + cv),
            [{"a": 84.0, "b": 198.0}],
        )

    def test_ct_single_row(self, tab_single):
        ct = ColumnwiseThread(lambda col: sorted(col))
        assert_df(
            transform_tabular(tab_single, lambda x: ct),
            [{"a": 42, "b": 99}],
        )


# ============================================================================
# Additions for version 0.9.0 (TransformTabular 1.0.1)
#
# The expected results of sections 8 to 10 were obtained from the original
# function in Wolfram Language 15.0.1.
# ============================================================================

MISSING = None  # stands for the Round[Missing[]] cells of the Wolfram example


@pytest.fixture
def tab():
    return pd.DataFrame([[2, 8, 5], [10, 0, 4], [7, 0, 7]], columns=["a", "b", "c"])


@pytest.fixture
def counts():
    """The table of the Scope examples: faulty cells left by some data processing."""
    data = {
        "Municipality": [
            "Adro", "Berlingo", "Castegnato", "Cazzago San Martino", "Corte Franca",
            "Erbusco", "Gussago", "Lograto", "Monticelli Brusati", "Ospitaletto",
            "Paderno Franciacorta", "Passirano", "Provaglio D'Iseo", "Rodengo-Saiano",
            "Roncadelle", "Rovato", "Torbole Casaglia", "Travagliato"],
        "2023": [210, 273, 1032, MISSING, 78, 340, 282, 312, 256, 92, 479, 330, MISSING,
                 602, 556, 392, 286, 243],
        "2022": [MISSING, 344, 1089, 273, 216, 386, 295, 449, 332, 367, 533, 489, 490, 641,
                 644, MISSING, 424, 335],
        "2021": [194, 322, MISSING, 248, 198, 374, MISSING, 531, 400, MISSING, 581, 304,
                 454, 540, 529, 385, 513, 441],
        "2020": [160, 342, 820, 454, 239, 447, MISSING, 486, 492, 234, 392, 317, 423, 509,
                 344, 317, 563, 457],
        "2019": [92, 246, 493, 271, 301, 405, 202, 417, 262, 1, MISSING, 197, 271, 302,
                 MISSING, 277, 456, 351],
        "2018": [152, 132, 444, 342, 335, MISSING, 216, 480, 191, 35, 339, 178, 377, 242,
                 405, 315, 391, 380],
        "2017": [171, 120, 170, 153, 269, 693, 96, 450, 181, 39, 253, 49, 349, 174, 554,
                 248, 108, 315],
        "2016": [125, 173, 73, 189, MISSING, 616, 140, 371, 179, 56, 146, 110, 245, 101,
                 413, 310, 159, MISSING],
    }
    return pd.DataFrame(data, dtype=object)


@pytest.fixture
def clean(counts):
    # tabClean = TransformTabular[tabData, # /. Round[Missing[]] :> 0 &]
    return transform_tabular(counts, lambda x: 0 if x is MISSING else x)


# ============================================ 8. Function Specification

class TestFunctionSpecification:
    """The function can be any callable: in Wolfram terms, also a symbol or a composition."""

    def test_symbol(self, tab):
        # TransformTabular[tab, Sqrt]
        assert_df(
            transform_tabular(tab, math.sqrt),
            [{"a": math.sqrt(2), "b": 2 * math.sqrt(2), "c": math.sqrt(5)},
             {"a": math.sqrt(10), "b": 0.0, "c": 2.0},
             {"a": math.sqrt(7), "b": 0.0, "c": math.sqrt(7)}],
        )

    def test_builtin(self, tab_neg):
        assert_df(
            transform_tabular(tab_neg, abs),
            [{"a": 3, "b": 5}, {"a": 0, "b": 1}, {"a": 7, "b": 2}],
        )

    def test_composition_on_a_span(self, tab):
        # TransformTabular[tab, N @* Sqrt, 2 ;;]
        def compose(*functions):
            return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)

        assert_df(
            transform_tabular(tab, compose(float, math.sqrt), slice(1, None)),
            [{"a": 2, "b": 2.8284271247461903, "c": 2.23606797749979},
             {"a": 10, "b": 0.0, "c": 2.0},
             {"a": 7, "b": 0.0, "c": 2.6457513110645907}],
        )

    def test_right_composition_on_a_column(self, tab):
        # TransformTabular[tab, Sqrt /* N, {"a"}]
        assert_df(
            transform_tabular(tab, lambda x: float(math.sqrt(x)), ["a"]),
            [{"a": 1.4142135623730951, "b": 8, "c": 5},
             {"a": 3.1622776601683795, "b": 0, "c": 4},
             {"a": 2.6457513110645907, "b": 0, "c": 7}],
        )

    def test_operator_form_with_a_symbol(self, tab):
        # TransformTabular[Minus][tab]
        assert_df(
            transform_tabular(operator.neg)(tab),
            [{"a": -2, "b": -8, "c": -5}, {"a": -10, "b": 0, "c": -4}, {"a": -7, "b": 0, "c": -7}],
        )

    def test_operator_form_with_columns(self, tab):
        # TransformTabular[# + 1 &, {"a", "c"}][tab]
        expected = [{"a": 3, "b": 8, "c": 6}, {"a": 11, "b": 0, "c": 5}, {"a": 8, "b": 0, "c": 8}]
        assert_df(transform_tabular(lambda x: x + 1, ["a", "c"])(tab), expected)
        assert_df(transform_tabular(lambda x: x + 1, columns=["a", "c"])(tab), expected)

    def test_method_partial_and_callable_object(self, tab_str):
        assert transform_tabular(tab_str, str.upper, "name")["name"].tolist() == ["ALICE", "BOB", "CAROL"]
        assert transform_tabular(tab_str, functools.partial(operator.mul, 2), "score")[
            "score"].tolist() == [170, 184, 156]

        class Scale:
            def __call__(self, x):
                return x / 100

        assert transform_tabular(tab_str, Scale(), "score")["score"].tolist() == [0.85, 0.92, 0.78]

    def test_named_function(self, tab):
        # a function with a named argument, which the original does not support
        def successor(value):
            return value + 1

        assert_df(
            transform_tabular(tab, successor, "b"),
            [{"a": 2, "b": 9, "c": 5}, {"a": 10, "b": 1, "c": 4}, {"a": 7, "b": 1, "c": 7}],
        )


# ============================================ 9. Documented Examples

class TestDocumentedExamples:

    def test_clean_the_faulty_cells(self, counts, clean):
        # TransformTabular[tabData, # /. Round[Missing[]] :> 0 &]
        assert clean["Municipality"].tolist() == counts["Municipality"].tolist()
        assert clean["2023"].tolist() == [
            210, 273, 1032, 0, 78, 340, 282, 312, 256, 92, 479, 330, 0, 602, 556, 392, 286, 243]
        assert clean["2022"].tolist() == [
            0, 344, 1089, 273, 216, 386, 295, 449, 332, 367, 533, 489, 490, 641, 644, 0, 424, 335]
        assert clean["2017"].tolist() == counts["2017"].tolist()
        assert all(clean[year].dtype == "int64" for year in clean.columns[1:])

    def test_numeric_function_on_a_span_of_columns(self, clean):
        # TransformTabular[tabClean, N @* Sqrt, 2 ;;]
        result = transform_tabular(clean, math.sqrt, slice(1, None))
        assert result["Municipality"].tolist() == clean["Municipality"].tolist()
        assert result["2023"].tolist() == pytest.approx([
            14.491376746189438, 16.522711641858304, 32.12475680841802, 0.0,
            8.831760866327848, 18.439088914585774, 16.792855623746664, 17.663521732655695,
            16.0, 9.591663046625438, 21.88606862823929, 18.16590212458495, 0.0,
            24.535688292770594, 23.57965224510319, 19.79898987322333, 16.911534525287763,
            15.588457268119896])
        assert result["2016"].tolist() == pytest.approx([
            11.180339887498949, 13.152946437965905, 8.54400374531753, 13.74772708486752, 0.0,
            24.819347291981714, 11.832159566199232, 19.261360284258224, 13.379088160259652,
            7.483314773547883, 12.083045973594572, 10.488088481701515, 15.652475842498529,
            10.04987562112089, 20.322401432901575, 17.60681686165901, 12.609520212918492, 0.0])

    def test_columnwise_value(self, clean):
        # TransformTabular[tabClean, N[# - ColumnwiseValue@Median[#]] &, 2 ;;]
        result = transform_tabular(
            clean, lambda x: x - ColumnwiseValue(statistics.median), slice(1, None)
        )
        assert result["2023"].tolist() == [
            -74.0, -11.0, 748.0, -284.0, -206.0, 56.0, -2.0, 28.0, -28.0, -192.0, 195.0,
            46.0, -284.0, 318.0, 272.0, 108.0, 2.0, -41.0]
        assert result["2019"].tolist() == [
            -179.0, -25.0, 222.0, 0.0, 30.0, 134.0, -69.0, 146.0, -9.0, -270.0, -271.0,
            -74.0, 0.0, 31.0, -271.0, 6.0, 185.0, 80.0]
        assert result["2016"].tolist() == [
            -27.5, 20.5, -79.5, 36.5, -152.5, 463.5, -12.5, 218.5, 26.5, -96.5, -6.5, -42.5,
            92.5, -51.5, 260.5, 157.5, 6.5, -152.5]

    def test_columnwise_thread(self, clean):
        # TransformTabular[tabClean, ColumnwiseThread[Accumulate@#] &, 2 ;;]
        result = transform_tabular(
            clean, lambda x: ColumnwiseThread(lambda col: list(accumulate(col))), slice(1, None)
        )
        assert result["Municipality"].tolist() == clean["Municipality"].tolist()
        assert result["2023"].tolist() == [
            210, 483, 1515, 1515, 1593, 1933, 2215, 2527, 2783, 2875, 3354, 3684, 3684, 4286,
            4842, 5234, 5520, 5763]
        assert result["2016"].tolist() == [
            125, 298, 371, 560, 560, 1176, 1316, 1687, 1866, 1922, 2068, 2178, 2423, 2524,
            2937, 3247, 3406, 3406]

    def test_columns_by_names_and_by_positions(self):
        # TransformTabular[TransformTabular[tabInt, f, {"a", "c", "e", "g", "i"}], g, {2, 4, 6, 8, 10}]
        tab_int = pd.DataFrame(
            [[10 * row + column for column in range(10)] for row in range(4)],
            columns=list("abcdefghij"),
        )
        result = transform_tabular(
            transform_tabular(tab_int, lambda x: -x, ["a", "c", "e", "g", "i"]),
            lambda x: x + 1000, [1, 3, 5, 7, 9],
        )
        assert result.values.tolist() == [
            [-(10 * row + column) if column % 2 == 0 else 10 * row + column + 1000
             for column in range(10)]
            for row in range(4)
        ]


# ============================================ 10. Properties and Relations

class TestPropertiesAndRelations:

    def test_same_as_transform_columns_on_every_column(self, tab):
        # TransformTabular[tab, # - ColumnwiseValue[Min[#]] &] ==
        #   TransformColumns[tab, {"a" -> Function[#a - ColumnwiseValue[Min[#a]]], ...}]
        def below_minimum(label):
            minimum = ColumnwiseValue(lambda t: t[label].min())
            return lambda r: r[label] - minimum

        result = transform_tabular(tab, lambda x: x - ColumnwiseValue(min))
        assert_df(result, [{"a": 0, "b": 8, "c": 1}, {"a": 8, "b": 0, "c": 0}, {"a": 5, "b": 0, "c": 3}])
        pd.testing.assert_frame_equal(
            result, transform_columns(tab, {label: below_minimum(label) for label in tab.columns})
        )

    def test_same_as_mapping_the_function_over_the_table(self, tab):
        # TransformTabular[tab, #^2 &] == Tabular[Query[All, All, #^2 &][Normal@tab]]
        elementwise = getattr(pd.DataFrame, "map", None) or pd.DataFrame.applymap
        pd.testing.assert_frame_equal(
            transform_tabular(tab, lambda x: x ** 2), elementwise(tab, lambda x: x ** 2)
        )

    def test_combined_thread_and_value(self, tab):
        # TransformTabular[tab, ColumnwiseThread[Accumulate@#] - ColumnwiseValue[Mean[#]] &]
        result = transform_tabular(
            tab,
            lambda x: ColumnwiseThread(lambda col: list(accumulate(col)))
            - ColumnwiseValue(statistics.mean),
        )
        assert_df(result, [
            {"a": -4.333333333333333, "b": 5.333333333333334, "c": -0.33333333333333304},
            {"a": 5.666666666666667, "b": 5.333333333333334, "c": 3.666666666666667},
            {"a": 12.666666666666668, "b": 5.333333333333334, "c": 10.666666666666668},
        ])

    def test_empty_list_of_columns(self, tab):
        # TransformTabular[tab, # + 1 &, {}]
        pd.testing.assert_frame_equal(transform_tabular(tab, lambda x: x + 1, []), tab)


# ============================================ 11. Column Specification (additions)

class TestColumnLabelsAndPositions:

    def test_tuple_range_and_array(self, tab):
        expected = [{"a": 3, "b": 8, "c": 6}, {"a": 11, "b": 0, "c": 5}, {"a": 8, "b": 0, "c": 8}]
        for columns in (("a", "c"), range(0, 3, 2), np.array([0, 2]), pd.Index(["a", "c"]),
                        [np.int64(0), "c"], ["a", "c", "a", 0]):
            assert_df(transform_tabular(tab, lambda x: x + 1, columns), expected)

    def test_integer_labels(self):
        tab = pd.DataFrame([[1, 2, 3]], columns=[2024, 2025, 2026])
        assert transform_tabular(tab, lambda x: -x, 2025).values.tolist() == [[1, -2, 3]]
        assert transform_tabular(tab, lambda x: -x, [2024, 2026]).values.tolist() == [[-1, 2, -3]]
        # an integer that is not a label is a position
        assert transform_tabular(tab, lambda x: -x, 0).values.tolist() == [[-1, 2, 3]]
        assert transform_tabular(tab, lambda x: -x, [-1]).values.tolist() == [[1, 2, -3]]

    def test_labels_come_before_positions(self):
        tab = pd.DataFrame([[1, 2, 3]], columns=[1, 2, 0])
        assert transform_tabular(tab, lambda x: -x, 0).values.tolist() == [[1, 2, -3]]
        assert transform_tabular(tab, lambda x: -x, slice(0, 1)).values.tolist() == [[-1, 2, 3]]

    def test_columns_of_a_matrix(self):
        tab = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        assert transform_tabular(tab, lambda x: 10 * x, [0, 2]).values.tolist() == [
            [10, 2, 30], [40, 5, 60]]

    def test_tuple_label(self):
        tab = pd.DataFrame([[1, 2]], columns=[("x", "y"), "z"])
        assert transform_tabular(tab, lambda x: -x, [("x", "y")]).values.tolist() == [[-1, 2]]


# ============================================ 12. Markers given as functions

class TestMarkersAsFunctions:

    def test_thread(self, tab):
        # TransformTabular[tab, ColumnwiseThread[Accumulate@#] &, {"a"}]
        running = ColumnwiseThread(lambda col: list(accumulate(col)))
        expected = [{"a": 2, "b": 8, "c": 5}, {"a": 12, "b": 0, "c": 4}, {"a": 19, "b": 0, "c": 7}]
        assert_df(transform_tabular(tab, running, ["a"]), expected)
        assert_df(transform_tabular(running, ["a"])(tab), expected)

    def test_value(self, tab):
        assert_df(
            transform_tabular(tab, ColumnwiseValue(max)),
            [{"a": 10, "b": 8, "c": 7}] * 3,
        )


# ============================================ 13. Tables

class TestTables:

    def test_input_is_not_modified(self, tab):
        before = tab.copy()
        transform_tabular(tab, lambda x: x * 0)
        pd.testing.assert_frame_equal(tab, before)

    def test_key_column(self):
        # a Tabular with "KeyColumns" -> {"k"}: the key column corresponds to the index
        tab = pd.DataFrame({"a": [1, 3], "b": [2, 4]}, index=pd.Index(["x", "y"], name="k"))
        result = transform_tabular(tab, lambda x: x + 10, ["a", "b"])
        assert result.values.tolist() == [[11, 12], [13, 14]]
        assert result.index.equals(tab.index) and result.index.name == "k"

    def test_index_with_repeated_labels_and_attributes(self):
        tab = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "x", "y"])
        tab.attrs = {"source": "test"}
        result = transform_tabular(tab, lambda x: x - ColumnwiseValue(max))
        assert result["a"].tolist() == [-2, -1, 0]
        assert list(result.index) == ["x", "x", "y"]
        assert result.attrs == {"source": "test"}

    def test_untouched_columns_keep_their_type(self):
        tab = pd.DataFrame({
            "a": pd.array([1, 2, 3], dtype="int8"),
            "kind": pd.Categorical(["u", "v", "u"]),
        })
        result = transform_tabular(tab, lambda x: x + 0.5, "a")
        assert result["a"].tolist() == [1.5, 2.5, 3.5]
        assert result["kind"].dtype == tab["kind"].dtype

    def test_cells_of_any_kind(self, tab):
        result = transform_tabular(tab, lambda x: [x, x], "a")
        assert result["a"].tolist() == [[2, 2], [10, 10], [7, 7]]
        result = transform_tabular(tab, lambda x: {"value": x}, "a")
        assert result["a"].tolist() == [{"value": 2}, {"value": 10}, {"value": 7}]

    def test_missing_values(self):
        tab = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = transform_tabular(tab, lambda x: -1.0 if pd.isna(x) else x)
        assert result["a"].tolist() == [1.0, -1.0, 3.0]

    def test_result_can_be_modified(self, tab):
        result = transform_tabular(tab, lambda x: x + 1, "a")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result["d"] = 0
            result.iloc[0, 0] = -1
        assert tab.iloc[0, 0] == 2

    def test_table_without_rows(self):
        tab = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        result = transform_tabular(tab, lambda x: x / 2)
        pd.testing.assert_frame_equal(result, tab)


# ============================================ 14. Errors

class TestErrors:

    def test_unknown_column(self, tab):
        # the original does not terminate on a column that does not exist
        with pytest.raises(KeyError, match="zzz"):
            transform_tabular(tab, lambda x: x + 1, ["zzz"])
        with pytest.raises(KeyError, match="zzz"):
            transform_tabular(tab, lambda x: x + 1, "zzz")
        with pytest.raises(KeyError, match="zzz"):
            transform_tabular(tab, lambda x: x + 1, ["a", "zzz"])

    def test_column_position_out_of_range(self, tab):
        with pytest.raises(IndexError, match="out of range"):
            transform_tabular(tab, lambda x: x + 1, 7)
        with pytest.raises(IndexError, match="out of range"):
            transform_tabular(tab, lambda x: x + 1, [0, -4])

    @pytest.mark.parametrize("columns", [1.5, True, object()])
    def test_invalid_columns(self, tab, columns):
        with pytest.raises(TypeError, match="Invalid column specification"):
            transform_tabular(tab, lambda x: x + 1, columns)

    def test_invalid_arguments(self, tab):
        with pytest.raises(TypeError, match="DataFrame or callable"):
            transform_tabular([[1, 2]], lambda x: x)
        with pytest.raises(ValueError, match="required"):
            transform_tabular(tab)
        with pytest.raises(TypeError, match="callable"):
            transform_tabular(tab, 5)

    def test_repeated_column_labels(self):
        tab = pd.DataFrame([[1, 2]], columns=["a", "a"])
        with pytest.raises(ValueError, match="repeated column labels"):
            transform_tabular(tab, lambda x: x)
