"""Comprehensive tests for transform_tabular — translated from WL test suite.

Every test uses explicit expected results and pure functions (lambda),
matching the Wolfram Language TransformTabular tests one-to-one.
"""

import math
import statistics
from itertools import accumulate

import pandas as pd
import pytest

from transform_tabular import ColumnwiseValue, ColumnwiseThread, transform_tabular


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
