"""Tests for transform_rows, translated from the Wolfram Language function TransformRows.

The first class ports one-to-one the ten VerificationTests of the TransformRows 0.6.1
definition notebook; the following ones port its documented examples.  The expected
results of all the classes up to "Semantics" were obtained from the original function
in Wolfram Language 15.0.1.

Rows are counted from 0 here and from 1 in Wolfram Language: WL ``{1, 3}`` is ``[0, 2]``.
"""

import math
import statistics
import warnings
from itertools import accumulate

import numpy as np
import pandas as pd
import pytest

from transform_tabular import ColumnwiseThread, ColumnwiseValue, transform_rows


def assert_table(result, columns, rows):
    """*result* has exactly the given columns, in order, and the given rows."""
    assert list(result.columns) == list(columns)
    expected = pd.DataFrame(rows, columns=list(columns))
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected, check_dtype=False
    )


@pytest.fixture
def tab():
    return pd.DataFrame([[2, 8, 5], [10, 0, 4], [7, 0, 7]], columns=["a", "b", "c"])


@pytest.fixture
def tab2():
    return pd.DataFrame([[9, 1, 6], [10, 8, 5], [7, 0, 5]], columns=["a", "b", "c"])


@pytest.fixture
def companies():
    return pd.DataFrame(
        [["A", 8012, 5178], ["B", 20111, 40222], ["C", 992, 7032]],
        columns=["Company", "Expense", "Revenue"],
    )


@pytest.fixture
def municipalities():
    return pd.DataFrame({
        "id": ["103016047", "103016049", "103016053", "103016057", "103016059",
               "103016060", "103016061", "103016062", "103016066", "103016137"],
        "municipality": [
            "Comune di Calvenzano", "Comune di Canonica d'Adda", "Comune di Caravaggio",
            "Comune di Carvico", "Comune di Casirate d'Adda", "Comune di Casnigo",
            "Comune di Cassiglio", "Comune di Castelli Calepio", "Comune di Cavernago",
            "Comune di Monasterolo Del Castello"],
        "surface": [6.0, 3.0, 32.0, 4.0, 10.0, 13.0, 14.0, 9.0, 7.0, 8.0],
        "population": [4279.0, 4235.0, 16109.0, 4646.0, 4069.0, 3075.0, 111.0,
                       10398.0, 2720.0, 1141.0],
        "amount": [7710.57, 104397.91, 335500.39, 24906.36, 10660.15, 48422.68,
                   None, None, 9910.17, 46412.63],
    })


# ====================================== 1. VerificationTests of the notebook

class TestNotebookTests:

    def test_01_single_row_given_as_an_integer(self, tab):
        # TransformRows[tab, 3, "c" -> Function[2 #a]]
        assert_table(
            transform_rows(tab, 2, {"c": lambda r: 2 * r["a"]}),
            ["a", "b", "c"], [[2, 8, 5], [10, 0, 4], [7, 0, 14]],
        )

    def test_02_list_of_rows(self, tab):
        # TransformRows[tab, {1, 2}, "a" -> Function[#a + #b]]
        assert_table(
            transform_rows(tab, [0, 1], {"a": lambda r: r["a"] + r["b"]}),
            ["a", "b", "c"], [[10, 8, 5], [10, 0, 4], [7, 0, 7]],
        )

    def test_03_unsorted_row_list_keeps_the_row_order(self, tab):
        # TransformRows[tab, {3, 1}, "b" -> Function[#a + #c]]
        assert_table(
            transform_rows(tab, [2, 0], {"b": lambda r: r["a"] + r["c"]}),
            ["a", "b", "c"], [[2, 7, 5], [10, 0, 4], [7, 14, 7]],
        )

    def test_04_function_applied_to_all_cells_of_the_rows(self, tab):
        # TransformRows[tab, {1, 3}, #^3 &]
        assert_table(
            transform_rows(tab, [0, 2], lambda x: x ** 3),
            ["a", "b", "c"], [[8, 512, 125], [10, 0, 4], [343, 0, 343]],
        )

    def test_05_rows_selected_by_a_criterion(self, tab2):
        # TransformRows[tab, EvenQ[#c] &, "a" -> Function[2 #a + 2 #b]]
        assert_table(
            transform_rows(
                tab2, lambda r: r["c"] % 2 == 0, {"a": lambda r: 2 * r["a"] + 2 * r["b"]}
            ),
            ["a", "b", "c"], [[20, 1, 6], [10, 8, 5], [7, 0, 5]],
        )

    def test_06_list_of_rules(self, tab2):
        # TransformRows[tab, {2, 3}, {"a" -> Function[2 #a + 2 #b], "c" -> Function[#b - #c]}]
        assert_table(
            transform_rows(tab2, [1, 2], {
                "a": lambda r: 2 * r["a"] + 2 * r["b"],
                "c": lambda r: r["b"] - r["c"],
            }),
            ["a", "b", "c"], [[9, 1, 6], [36, 8, 3], [14, 0, -5]],
        )

    def test_07_columnwise_value_is_computed_over_the_whole_column(self, tab2):
        # TransformRows[tab, {2, 3}, "a" -> Function[2 #a - ColumnwiseValue[Max[#b]]]]
        largest_b = ColumnwiseValue(lambda t: t["b"].max())
        assert_table(
            transform_rows(tab2, [1, 2], {"a": lambda r: 2 * r["a"] - largest_b}),
            ["a", "b", "c"], [[9, 1, 6], [12, 8, 5], [6, 0, 5]],
        )

    def test_08_no_row_matches_and_the_table_is_unchanged(self):
        tab = pd.DataFrame([[8, 0, 0], [10, 10, 9], [10, 2, 1]], columns=["a", "b", "c"])
        result = transform_rows(
            tab, lambda r: r["c"] > 10, {"a": lambda r: 2 * r["a"] + 2 * r["b"]}
        )
        pd.testing.assert_frame_equal(result, tab)

    def test_09_options_of_the_table_are_preserved(self):
        # The Options of a Tabular (Alignment, ItemStyle) correspond to the metadata
        # that a DataFrame carries along: its attrs and its index.
        tab = pd.DataFrame(
            [[1, 2, 3.0], [4, 5, 6.0], [7, 8, 9.0]],
            columns=["a", "b", "c"], index=["x", "y", "z"],
        )
        tab.attrs = {"Alignment": ["Left", "Center", "Right"],
                     "ItemStyle": ["Bold", "Italic", "Red"]}
        result = transform_rows(tab, [1, 2], {"c": lambda r: 2 * r["c"]})
        assert result.attrs == tab.attrs
        assert list(result.index) == ["x", "y", "z"]
        assert_table(result, ["a", "b", "c"], [[1, 2, 3.0], [4, 5, 12.0], [7, 8, 18.0]])

    def test_10_fill_missing_values(self):
        # TransformRows[tab, Function[MissingQ[#amount]], "amount" -> Function[
        #   With[{name = #name}, First[KeySelect[values, StringEndsQ[name, #] &], #amount]]]]
        values = {"x": 1.5, "y": 2.5}
        tab = pd.DataFrame({"name": ["ax", "by", "cx"], "amount": [1.0, None, None]})

        def from_values(r):
            found = [value for key, value in values.items() if r["name"].endswith(key)]
            return found[0] if found else r["amount"]

        assert_table(
            transform_rows(tab, lambda r: pd.isna(r["amount"]), {"amount": from_values}),
            ["name", "amount"], [["ax", 1.0], ["by", 2.5], ["cx", 1.5]],
        )


# ====================================== 2. Documented examples

class TestBasicExamples:

    def test_single_cell(self, tab):
        # TransformRows[tab, {3}, "c" -> Function[2 #c]]
        assert_table(
            transform_rows(tab, [2], {"c": lambda r: 2 * r["c"]}),
            ["a", "b", "c"], [[2, 8, 5], [10, 0, 4], [7, 0, 14]],
        )

    def test_first_and_second_rows_of_the_first_column(self, tab):
        # TransformRows[tab, {1, 2}, "a" -> Function[#a + #b]]
        assert_table(
            transform_rows(tab, [0, 1], {"a": lambda r: r["a"] + r["b"]}),
            ["a", "b", "c"], [[10, 8, 5], [10, 0, 4], [7, 0, 7]],
        )


class TestScope:

    def test_all_cells_of_the_first_and_third_rows(self, tab2):
        # TransformRows[tab, {1, 3}, #^3 &]
        assert_table(
            transform_rows(tab2, [0, 2], lambda x: x ** 3),
            ["a", "b", "c"], [[729, 1, 216], [10, 8, 5], [343, 0, 125]],
        )

    def test_rows_selected_by_a_condition(self, tab2):
        # TransformRows[tab, EvenQ[#c] &, "a" -> Function[2 #a + 2 #b]]
        assert_table(
            transform_rows(
                tab2, lambda r: r["c"] % 2 == 0, {"a": lambda r: 2 * r["a"] + 2 * r["b"]}
            ),
            ["a", "b", "c"], [[20, 1, 6], [10, 8, 5], [7, 0, 5]],
        )

    def test_different_columns(self, tab2):
        # TransformRows[tab, {2, 3}, {"a" -> Function[2 #a + 2 #b], "b" -> Function[2 #a - 2 #b]}]
        # "b" is computed from the original "a", not from the transformed one
        assert_table(
            transform_rows(tab2, [1, 2], {
                "a": lambda r: 2 * r["a"] + 2 * r["b"],
                "b": lambda r: 2 * r["a"] - 2 * r["b"],
            }),
            ["a", "b", "c"], [[9, 1, 6], [36, 4, 5], [14, 14, 5]],
        )

    def test_columnwise_value_and_thread(self, tab2):
        # TransformRows[tab, {2, 3}, {"a" -> Function[2 #a - ColumnwiseValue[Median[#b]]],
        #                             "b" -> Function[#b + ColumnwiseThread[Accumulate[#b]]]}]
        median_b = ColumnwiseValue(lambda t: t["b"].median())
        running_b = ColumnwiseThread(lambda t: t["b"].cumsum())
        assert_table(
            transform_rows(tab2, [1, 2], {
                "a": lambda r: 2 * r["a"] - median_b,
                "b": lambda r: r["b"] + running_b,
            }),
            ["a", "b", "c"], [[9.0, 1, 6], [19.0, 17, 5], [13.0, 9, 5]],
        )


class TestKeepOrder:

    def test_transformed_rows_first(self, companies):
        # TransformRows[tab, #Company =!= "A" &, "Expense" -> Function[#Expense + 432],
        #   "KeepOrder" -> False]
        result = transform_rows(
            companies, lambda r: r["Company"] != "A",
            {"Expense": lambda r: r["Expense"] + 432}, keep_order=False,
        )
        assert_table(
            result, ["Company", "Expense", "Revenue"],
            [["B", 20543, 40222], ["C", 1424, 7032], ["A", 8012, 5178]],
        )
        # every row keeps its index label
        assert list(result.index) == [1, 2, 0]

    def test_transformed_rows_in_the_given_order(self, tab):
        # TransformRows[tab, {3, 1}, "a" -> Function[2 #a], "KeepOrder" -> False]
        result = transform_rows(tab, [2, 0], {"a": lambda r: 2 * r["a"]}, keep_order=False)
        assert_table(result, ["a", "b", "c"], [[14, 0, 7], [4, 8, 5], [10, 0, 4]])
        assert list(result.index) == [2, 0, 1]

    def test_keep_order_is_the_default(self, companies):
        result = transform_rows(
            companies, lambda r: r["Company"] != "A",
            {"Expense": lambda r: r["Expense"] + 432},
        )
        assert_table(
            result, ["Company", "Expense", "Revenue"],
            [["A", 8012, 5178], ["B", 20543, 40222], ["C", 1424, 7032]],
        )


class TestApplications:

    def test_replace_missing_values_from_another_source(self, municipalities):
        # TransformRows[tabData, Function[StringContainsQ[#municipality, {"Castelli", "Cassiglio"}]],
        #   {"amount" -> Function[With[{mun = #municipality},
        #      First[KeySelect[missingValues, StringWithinQ[mun, #] &], #amount]]]}]
        missing_values = {"Castelli": 11920.21, "Cassiglio": 13080.55}

        def named(r):
            return any(name in r["municipality"] for name in missing_values)

        def amount(r):
            found = [v for name, v in missing_values.items() if name in r["municipality"]]
            return found[0] if found else r["amount"]

        result = transform_rows(municipalities, named, {"amount": amount})
        assert result["amount"].tolist() == [
            7710.57, 104397.91, 335500.39, 24906.36, 10660.15, 48422.68,
            13080.55, 11920.21, 9910.17, 46412.63,
        ]
        untouched = ["id", "municipality", "surface", "population"]
        pd.testing.assert_frame_equal(result[untouched], municipalities[untouched])


class TestPropertiesAndRelations:

    def test_no_matching_row(self):
        tab0 = pd.DataFrame([[8, 0, 0], [10, 10, 9], [10, 2, 1]], columns=["a", "b", "c"])
        tab1 = transform_rows(
            tab0, lambda r: r["c"] > 10, {"a": lambda r: 2 * r["a"] + 2 * r["b"]}
        )
        assert tab1.equals(tab0)
        assert tab1 is not tab0

    def test_square_roots_of_the_last_rows(self):
        # TransformRows[tabS, {2, 3}, Sqrt[#] &]
        tab = pd.DataFrame([[1, 2, 3.0], [4, 5, 6.0], [7, 8, 9.0]], columns=["a", "b", "c"])
        assert_table(
            transform_rows(tab, [1, 2], lambda x: math.sqrt(x)),
            ["a", "b", "c"],
            [[1, 2, 3.0],
             [2.0, math.sqrt(5), math.sqrt(6)],
             [math.sqrt(7), 2 * math.sqrt(2), 3.0]],
        )


class TestPossibleIssues:
    """The large-table example of the notebook, scaled down from 10^7 to 2*10^4 rows."""

    @pytest.fixture
    def large(self):
        generator = np.random.default_rng(20261116)
        table = pd.DataFrame(
            generator.integers(0, 10 ** 6, size=(20000, 10)), columns=list("abcdefghij")
        )
        rows = np.unique(generator.integers(0, 20000, size=2000))
        return table, rows

    def test_many_rows(self, large):
        table, rows = large
        result = transform_rows(table, rows, {"a": lambda r: 2 * r["a"]})
        expected = table.copy()
        expected.loc[rows, "a"] *= 2
        pd.testing.assert_frame_equal(result, expected)

    def test_many_rows_without_keeping_the_order(self, large):
        table, rows = large
        result = transform_rows(table, rows, {"a": lambda r: 2 * r["a"]}, keep_order=False)
        expected = table.copy()
        expected.loc[rows, "a"] *= 2
        others = np.setdiff1d(np.arange(len(table)), rows)
        pd.testing.assert_frame_equal(result, expected.iloc[np.concatenate([rows, others])])


# ====================================== 3. Semantics (verified against the original)

class TestSemantics:

    def test_single_function_sees_whole_columns(self, tab2):
        # TransformRows[tab, {2, 3}, # - ColumnwiseValue[Max[#]] &]
        # the maximum of "c" is in the first row, which is not transformed
        assert_table(
            transform_rows(tab2, [1, 2], lambda x: x - ColumnwiseValue(max)),
            ["a", "b", "c"], [[9, 1, 6], [0, 0, -1], [-3, -8, -1]],
        )

    def test_columnwise_value_of_a_plain_function(self, tab2):
        # the median of the whole column "b" is 1
        median_b = ColumnwiseValue(lambda t: statistics.median(t["b"]))
        result = transform_rows(tab2, [1], {"a": lambda r: r["a"] - median_b})
        assert result["a"].tolist() == [9, 9, 7]

    def test_columnwise_thread_follows_the_row_positions(self, tab2):
        running = ColumnwiseThread(lambda col: list(accumulate(col)))
        assert_table(
            transform_rows(tab2, [2, 0], lambda x: running),
            ["a", "b", "c"], [[9, 1, 6], [10, 8, 5], [26, 9, 16]],
        )

    def test_new_column_is_missing_in_the_other_rows(self, tab):
        # TransformRows[tab, {1}, "d" -> Function[#a + #b]]
        result = transform_rows(tab, [0], {"d": lambda r: r["a"] + r["b"]})
        assert list(result.columns) == ["a", "b", "c", "d"]
        assert result["d"].iloc[0] == 10
        assert result["d"].iloc[1:].isna().all()
        pd.testing.assert_frame_equal(result[["a", "b", "c"]], tab)

    def test_removed_column_loses_the_cells_of_the_rows(self, tab):
        # TransformRows[tab, {1}, "b" -> Nothing]
        result = transform_rows(tab, [0], {"b": None})
        assert list(result.columns) == ["a", "b", "c"]
        assert pd.isna(result["b"].iloc[0])
        assert result["b"].iloc[1:].tolist() == [0, 0]

    def test_removed_column_with_all_rows_selected(self, tab):
        # TransformRows[tab, {1, 2, 3}, "b" -> Nothing]
        result = transform_rows(tab, [0, 1, 2], {"b": None})
        assert list(result.columns) == ["a", "b", "c"]
        assert result["b"].isna().all()

    def test_all_rows(self, tab):
        # TransformRows[tab, {1, 2, 3}, "a" -> Function[2 #a]]
        assert_table(
            transform_rows(tab, [0, 1, 2], {"a": lambda r: 2 * r["a"]}),
            ["a", "b", "c"], [[4, 8, 5], [20, 0, 4], [14, 0, 7]],
        )

    def test_empty_list_of_rows(self, tab):
        # TransformRows[tab, {}, "a" -> Function[2 #a]]
        pd.testing.assert_frame_equal(transform_rows(tab, [], {"a": lambda r: 2 * r["a"]}), tab)

    def test_function_is_not_needed_outside_the_rows(self, tab):
        # TransformRows[tab, #b != 0 &, "a" -> Function[#a/#b]]
        # the division by zero in the other rows does not matter
        assert_table(
            transform_rows(tab, lambda r: r["b"] != 0, {"a": lambda r: r["a"] / r["b"]}),
            ["a", "b", "c"], [[0.25, 8, 5], [10, 0, 4], [7, 0, 7]],
        )


# ====================================== 4. Row specifications

class TestRowSpecifications:

    def test_negative_position(self, tab):
        assert_table(
            transform_rows(tab, -1, {"a": lambda r: 2 * r["a"]}),
            ["a", "b", "c"], [[2, 8, 5], [10, 0, 4], [14, 0, 7]],
        )

    def test_repeated_positions_count_once(self, tab):
        assert_table(
            transform_rows(tab, [0, 0, -3], {"a": lambda r: 2 * r["a"]}),
            ["a", "b", "c"], [[4, 8, 5], [10, 0, 4], [7, 0, 7]],
        )

    def test_slice(self, tab):
        assert_table(
            transform_rows(tab, slice(1, None), {"a": lambda r: -r["a"]}),
            ["a", "b", "c"], [[2, 8, 5], [-10, 0, 4], [-7, 0, 7]],
        )

    def test_range_and_array(self, tab):
        expected = [[2, 8, 5], [-10, 0, 4], [-7, 0, 7]]
        negate = {"a": lambda r: -r["a"]}
        assert_table(transform_rows(tab, range(1, 3), negate), ["a", "b", "c"], expected)
        assert_table(transform_rows(tab, np.array([1, 2]), negate), ["a", "b", "c"], expected)

    def test_boolean_series(self, tab):
        assert_table(
            transform_rows(tab, tab["b"] == 0, {"a": lambda r: -r["a"]}),
            ["a", "b", "c"], [[2, 8, 5], [-10, 0, 4], [-7, 0, 7]],
        )

    def test_boolean_list(self, tab):
        assert_table(
            transform_rows(tab, [False, True, True], {"a": lambda r: -r["a"]}),
            ["a", "b", "c"], [[2, 8, 5], [-10, 0, 4], [-7, 0, 7]],
        )

    def test_nullable_boolean_series(self, tab):
        selection = pd.array([None, True, True], dtype="boolean")
        assert_table(
            transform_rows(tab, pd.Series(selection), {"a": lambda r: -r["a"]}),
            ["a", "b", "c"], [[2, 8, 5], [-10, 0, 4], [-7, 0, 7]],
        )

    def test_criterion_reads_the_row_by_attribute(self, tab):
        assert_table(
            transform_rows(tab, lambda r: r.b == 0, {"a": lambda r: r.a + r.c}),
            ["a", "b", "c"], [[2, 8, 5], [14, 0, 4], [14, 0, 7]],
        )

    def test_criterion_with_a_columnwise_value(self, tab):
        mean_a = ColumnwiseValue(lambda t: t["a"].mean())
        assert_table(
            transform_rows(tab, lambda r: r["a"] > mean_a, {"c": lambda r: 0}),
            ["a", "b", "c"], [[2, 8, 5], [10, 0, 0], [7, 0, 0]],
        )

    def test_positions_do_not_depend_on_the_index(self):
        tab = pd.DataFrame({"a": [1, 2, 3]}, index=[2, 0, 1])
        result = transform_rows(tab, 0, {"a": lambda r: 10 * r["a"]})
        assert result["a"].tolist() == [10, 2, 3]
        assert list(result.index) == [2, 0, 1]


# ====================================== 5. Tables

class TestTables:

    def test_input_is_not_modified(self, tab):
        before = tab.copy()
        transform_rows(tab, [0], {"a": lambda r: 100, "d": lambda r: 1, "b": None})
        pd.testing.assert_frame_equal(tab, before)

    def test_index_with_repeated_labels(self):
        tab = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "x", "y"])
        result = transform_rows(tab, [1], {"a": lambda r: r["a"] + r["b"], "c": lambda r: 0})
        assert list(result.index) == ["x", "x", "y"]
        assert result["a"].tolist() == [1, 7, 3]
        assert result["c"].iloc[1] == 0 and result["c"].isna().sum() == 2

    def test_untouched_columns_keep_their_type(self):
        tab = pd.DataFrame({
            "a": pd.array([1, 2, 3], dtype="int8"),
            "when": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "kind": pd.Categorical(["u", "v", "u"]),
        })
        result = transform_rows(tab, [0], {"b": lambda r: 1.5})
        assert result[["a", "when", "kind"]].dtypes.tolist() == tab.dtypes.tolist()

    def test_column_type_follows_the_new_values(self, tab):
        result = transform_rows(tab, [0], {"a": lambda r: r["a"] / 4, "c": lambda r: "five"})
        assert result["a"].tolist() == [0.5, 10.0, 7.0]
        assert result["a"].dtype == "float64"
        assert result["b"].dtype == "int64"
        assert result["c"].tolist() == ["five", 4, 7]

    def test_dates(self):
        tab = pd.DataFrame({"when": pd.to_datetime(["2026-01-01", "2026-06-30"])})
        result = transform_rows(tab, [1], {"when": lambda r: r["when"].replace(month=7, day=1)})
        assert result["when"].tolist() == list(pd.to_datetime(["2026-01-01", "2026-07-01"]))
        assert result["when"].dtype == tab["when"].dtype

    def test_placement_of_a_new_column(self, tab):
        result = transform_rows(tab, [0], ["a", {"d": lambda r: r["a"] + r["b"]}])
        assert list(result.columns) == ["a", "d", "b", "c"]

    def test_result_can_be_modified(self, tab):
        result = transform_rows(tab, [0], {"d": lambda r: 1}, keep_order=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result["e"] = 0
            result.iloc[0, 0] = -1
        assert tab.iloc[0, 0] == 2

    def test_empty_table(self):
        tab = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        result = transform_rows(tab, lambda r: True, {"a": lambda r: 2 * r["a"]})
        pd.testing.assert_frame_equal(result, tab)


# ====================================== 6. Errors

class TestErrors:

    def test_position_out_of_range(self, tab):
        with pytest.raises(IndexError, match="out of range"):
            transform_rows(tab, [3], {"a": lambda r: 0})
        with pytest.raises(IndexError, match="out of range"):
            transform_rows(tab, -4, {"a": lambda r: 0})

    def test_boolean_selection_of_the_wrong_length(self, tab):
        with pytest.raises(ValueError, match="one entry per row"):
            transform_rows(tab, [True, False], {"a": lambda r: 0})

    @pytest.mark.parametrize("rows", ["a", 1.5, [0.5, 1.5], ["a"], {0: 1}, None])
    def test_invalid_rows(self, tab, rows):
        with pytest.raises(TypeError):
            transform_rows(tab, rows, {"a": lambda r: 0})

    @pytest.mark.parametrize("transformation", [5, "a", None])
    def test_invalid_transformation(self, tab, transformation):
        with pytest.raises(TypeError):
            transform_rows(tab, [0], transformation)

    def test_not_a_table(self):
        with pytest.raises(TypeError, match="DataFrame"):
            transform_rows([[1, 2]], [0], {"a": lambda r: 0})

    def test_keep_order_is_a_keyword(self, tab):
        with pytest.raises(TypeError):
            transform_rows(tab, [0], {"a": lambda r: 0}, False)

    def test_error_in_the_transformation(self, tab):
        with pytest.raises(ZeroDivisionError):
            transform_rows(tab, [1], {"a": lambda r: r["a"] / r["b"]})
        # the failed call leaves nothing behind
        assert_table(
            transform_rows(tab, [0], {"a": lambda r: r["a"] / r["b"]}),
            ["a", "b", "c"], [[0.25, 8, 5], [10, 0, 4], [7, 0, 7]],
        )
