"""Tests for transform_columns, the translation of the built-in Wolfram function TransformColumns.

The classes up to "Semantics" port the examples of the TransformColumns reference page
and further cases probing its behaviour; their expected results were obtained from the
original function in Wolfram Language 15.0.1.

A rule ``"col" -> Function[...]`` of Wolfram Language is an item ``"col": function`` of a
dictionary here; ``Nothing`` is ``None``.
"""

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

from transform_tabular import ColumnwiseThread, ColumnwiseValue, transform_columns


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
def one_row():
    return pd.DataFrame([{"a": 1, "b": 2, "c": 3, "d": 4}])


def total(r):
    return r["a"] + r["b"]


def product(r):
    return r["a"] * r["c"]


def one(r):
    return 1


# ====================================== 1. Basic Examples

class TestBasicExamples:

    @pytest.fixture
    def basic(self):
        return pd.DataFrame([[2, 2.78], [1, 3.14], [3, 1.68]], columns=["a", "b"])

    def test_difference_into_a_new_column(self, basic):
        # TransformColumns[tab, "c" -> Function[#b - #a]]
        assert_table(
            transform_columns(basic, {"c": lambda r: r["b"] - r["a"]}),
            ["a", "b", "c"], [[2, 2.78, 0.78], [1, 3.14, 2.14], [3, 1.68, -1.32]],
        )

    def test_double_one_column_and_halve_another(self, basic):
        # TransformColumns[tab, {"a" -> (2 #a &), "b" -> (#b/2 &)}]
        assert_table(
            transform_columns(basic, {"a": lambda r: 2 * r["a"], "b": lambda r: r["b"] / 2}),
            ["a", "b"], [[4, 1.39], [2, 1.57], [6, 0.84]],
        )

    def test_existing_columns_into_a_new_column(self):
        # TransformColumns[tab, "date" -> Function[DateObject[{#a, #b, #c}, "Day"]]]
        tab = pd.DataFrame([[2004, 4, 1], [2005, 4, 3], [2006, 4, 5]], columns=["a", "b", "c"])
        result = transform_columns(
            tab, {"date": lambda r: datetime.date(r["a"], r["b"], r["c"])}
        )
        assert list(result.columns) == ["a", "b", "c", "date"]
        assert result["date"].tolist() == [
            datetime.date(2004, 4, 1), datetime.date(2005, 4, 3), datetime.date(2006, 4, 5)
        ]


# ====================================== 2. Scope: transformations

class TestTransformations:

    def test_transform_an_existing_column(self):
        # TransformColumns[Tabular[{{1, 2}, {3, 4}}, {"a", "b"}], "a" -> Function[#a^2]]
        tab = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        assert_table(
            transform_columns(tab, {"a": lambda r: r["a"] ** 2}),
            ["a", "b"], [[1, 2], [9, 4]],
        )

    def test_split_an_existing_column(self):
        # TransformColumns[tab, {"year" -> (First[#b] &), "month" -> (Last[#b] &), "b" -> Nothing}]
        tab = pd.DataFrame({"a": [1, 3, 5], "b": [[2004, 1], [2005, 3], [2006, 2]]})
        assert_table(
            transform_columns(tab, {
                "year": lambda r: r["b"][0],
                "month": lambda r: r["b"][-1],
                "b": None,
            }),
            ["a", "year", "month"], [[1, 2004, 1], [3, 2005, 3], [5, 2006, 2]],
        )

    def test_function_acts_on_whole_rows(self):
        # TransformColumns[tab, "new" -> f] gives f[<|"c1" -> x11, "c2" -> x12|>], ...
        tab = pd.DataFrame([["x11", "x12"], ["x21", "x22"], ["x31", "x32"]], columns=["c1", "c2"])
        result = transform_columns(tab, {"new": lambda r: dict(r)})
        assert result["new"].tolist() == [
            {"c1": "x11", "c2": "x12"}, {"c1": "x21", "c2": "x22"}, {"c1": "x31", "c2": "x32"}
        ]

    def test_operator_form(self):
        # op = TransformColumns["b" -> Function[#b + 10]]; op[tab]
        tab = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["a", "b"])
        op = transform_columns({"b": lambda r: r["b"] + 10})
        assert_table(op(tab), ["a", "b"], [[1, 12], [3, 14], [5, 16]])

    def test_remove_a_column(self, one_row):
        # TransformColumns[tab, {"e" -> (50 &), "f" -> (60 &), "a" -> Nothing}]
        result = transform_columns(one_row, {"e": lambda r: 50, "f": lambda r: 60, "a": None})
        assert_table(result, ["b", "c", "d", "e", "f"], [[2, 3, 4, 50, 60]])

    def test_remove_three_columns(self, one_row):
        # TransformColumns[tab, {"e" -> (50 &), "f" -> (60 &), {"a", "c", "d"} -> Nothing}]
        result = transform_columns(one_row, {
            "e": lambda r: 50, "f": lambda r: 60, "a": None, "c": None, "d": None,
        })
        assert_table(result, ["b", "e", "f"], [[2, 50, 60]])

    def test_columns_without_names(self):
        # A matrix has no column keys in Wolfram Language; a DataFrame made from a
        # matrix has the labels 0, 1, ..., used like any other label.
        tab = pd.DataFrame([[1, 10], [2, 20], [3, 30]])
        result = transform_columns(tab, {2: lambda r: r[0] + r[1]})
        assert_table(result, [0, 1, 2], [[1, 10, 11], [2, 20, 22], [3, 30, 33]])


# ====================================== 3. Scope: input data

class TestInputData:

    @pytest.fixture
    def data(self):
        return pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    def test_modify_an_existing_column(self, data):
        # TransformColumns[data, "a" -> Function[#a^2]]
        assert_table(
            transform_columns(data, {"a": lambda r: r["a"] ** 2}), ["a", "b"], [[1, 2], [9, 4]]
        )

    def test_new_column_from_existing_columns(self, data):
        # TransformColumns[data, "c" -> Function[#a + #b]]
        assert_table(
            transform_columns(data, {"c": lambda r: r["a"] + r["b"]}),
            ["a", "b", "c"], [[1, 2, 3], [3, 4, 7]],
        )


# ====================================== 4. Scope: columnwise operations

class TestColumnwiseOperations:

    def test_subtract_the_mean_of_a_column(self):
        # TransformColumns[tab, "c-μ" -> (#c - ColumnwiseValue[Mean[#c]] &)]
        tab = pd.DataFrame({"c": [3.71, -2.7, -5.29, 9.33, 6.26]})
        mean_c = ColumnwiseValue(lambda t: t["c"].mean())
        result = transform_columns(tab, {"c-μ": lambda r: r["c"] - mean_c})
        assert_table(
            result, ["c", "c-μ"],
            [[3.71, 1.448], [-2.7, -4.962], [-5.29, -7.552], [9.33, 7.068], [6.26, 3.998]],
        )

    @pytest.fixture
    def values(self):
        return pd.DataFrame({"c": [4.24, 1.14, -7.38, 6.74, 6.22, 7.5]})

    def test_elements_above_the_median(self, values):
        # TransformColumns[tab, "above median" -> (#c > ColumnwiseValue[Median[#c]] &)]
        median_c = ColumnwiseValue(lambda t: t["c"].median())
        result = transform_columns(values, {"above median": lambda r: r["c"] > median_c})
        assert result["above median"].tolist() == [False, False, False, True, True, True]

    def test_constant_column_with_the_median(self, values):
        # TransformColumns[tab, {"median" -> (ColumnwiseValue[Median[#c]] &),
        #                        "above median" -> (#c > ColumnwiseValue[Median[#c]] &)}]
        median_c = ColumnwiseValue(lambda t: t["c"].median())
        result = transform_columns(values, {
            "median": lambda r: median_c,
            "above median": lambda r: r["c"] > median_c,
        })
        assert_table(result, ["c", "median", "above median"], [
            [4.24, 5.23, False], [1.14, 5.23, False], [-7.38, 5.23, False],
            [6.74, 5.23, True], [6.22, 5.23, True], [7.5, 5.23, True],
        ])

    def test_composition_of_operator_forms(self, values):
        # (TransformColumns["median" -> (ColumnwiseValue[Median[#c]] &)] /*
        #    TransformColumns["above_median" -> (#c > #median &)])[tab]
        median_c = ColumnwiseValue(lambda t: t["c"].median())
        result = (
            values
            .pipe(transform_columns({"median": lambda r: median_c}))
            .pipe(transform_columns({"above_median": lambda r: r["c"] > r["median"]}))
        )
        assert_table(result, ["c", "median", "above_median"], [
            [4.24, 5.23, False], [1.14, 5.23, False], [-7.38, 5.23, False],
            [6.74, 5.23, True], [6.22, 5.23, True], [7.5, 5.23, True],
        ])

    def test_accumulate_a_column(self):
        # TransformColumns[tab, "acc" -> Function[ColumnwiseThread[Accumulate[#c]]]]
        tab = pd.DataFrame({"c": [1, 2, 3, 4]})
        running = ColumnwiseThread(lambda t: t["c"].cumsum())
        assert_table(
            transform_columns(tab, {"acc": lambda r: running}),
            ["c", "acc"], [[1, 1], [2, 3], [3, 6], [4, 10]],
        )

    def test_subtract_the_reversed_column(self):
        # TransformColumns[tab, "d" -> Function[#c - ColumnwiseThread[Reverse[#c]]]]
        tab = pd.DataFrame({"c": [1, 2, 3, 4]})
        backwards = ColumnwiseThread(lambda t: t["c"][::-1])
        assert_table(
            transform_columns(tab, {"d": lambda r: r["c"] - backwards}),
            ["c", "d"], [[1, -3], [2, -1], [3, 1], [4, 3]],
        )

    def test_number_of_evaluations(self):
        # The "Echo" examples of the reference page: a function of the row is evaluated
        # at every row, a columnwise expression once.
        tab = pd.DataFrame({"old": [1, 2, 3]})
        calls = {"row": 0, "value": 0, "thread": 0}

        def row_function(r):
            calls["row"] += 1
            return r["old"] + sum_of_column + shifted

        def whole_sum(t):
            calls["value"] += 1
            return t["old"].sum()

        def whole_shift(t):
            calls["thread"] += 1
            return t["old"] + 10

        sum_of_column = ColumnwiseValue(whole_sum)
        shifted = ColumnwiseThread(whole_shift)
        result = transform_columns(tab, {"new": row_function})
        assert result["new"].tolist() == [1 + 6 + 11, 2 + 6 + 12, 3 + 6 + 13]
        assert calls == {"row": 3, "value": 1, "thread": 1}


# ====================================== 5. Scope: placement of columns

class TestPlacementOfColumns:

    def test_anchor_given_by_a_column_name(self, one_row):
        # TransformColumns[tab, {"e" -> (50 &), "f" -> (60 &), "a"}]
        result = transform_columns(one_row, [{"e": lambda r: 50, "f": lambda r: 60}, "a"])
        assert_table(result, ["e", "f", "a", "b", "c", "d"], [[50, 60, 1, 2, 3, 4]])

    def test_new_columns_are_appended_by_default(self, one_row):
        # TransformColumns[tab, {"e" -> (50 &), "f" -> (60 &)}]
        result = transform_columns(one_row, {"e": lambda r: 50, "f": lambda r: 60})
        assert_table(result, ["a", "b", "c", "d", "e", "f"], [[1, 2, 3, 4, 50, 60]])

    def test_before_one_column_and_after_another(self, one_row):
        # TransformColumns[tab, {"e" -> (50 &), "a", "b", "f" -> (60 &)}]
        result = transform_columns(one_row, [{"e": lambda r: 50}, "a", "b", {"f": lambda r: 60}])
        assert_table(result, ["e", "a", "b", "f", "c", "d"], [[50, 1, 2, 60, 3, 4]])

    def test_placement_and_removal(self, one_row):
        # TransformColumns[tab, {"e" -> (50 &), "a", "b", "f" -> (60 &), "a" -> Nothing}]
        result = transform_columns(
            one_row, [{"e": lambda r: 50}, "a", "b", {"f": lambda r: 60, "a": None}]
        )
        assert_table(result, ["e", "b", "f", "c", "d"], [[50, 2, 60, 3, 4]])

    def test_after_a_column(self, tab):
        # TransformColumns[tab, {"a", "d" -> Function[#a + #b]}]
        assert_table(
            transform_columns(tab, ["a", {"d": total}]),
            ["a", "d", "b", "c"], [[2, 10, 8, 5], [10, 10, 0, 4], [7, 7, 0, 7]],
        )

    @pytest.mark.parametrize("specification, columns", [
        # new columns named before the first anchor come first
        ([{"d": total}, "b"], "d a b c"),
        ([{"d": total}, "b", {"e": product}], "d a b e c"),
        ([{"d": total, "e": product}, "c"], "d e a b c"),
        # new columns named after an anchor follow it, in the given order
        (["b", {"d": total, "e": product}], "a b d e c"),
        (["a", {"d": total}, "c", {"e": product}], "a d b c e"),
        # naming an anchor again restarts from it
        (["b", {"d": total}, "b", {"e": product}], "a b e d c"),
        # an existing column is replaced where it is, and is not an anchor
        (["c", {"a": total}], "a b c"),
        (["c", {"a": total, "d": product}], "a b c d"),
        (["a", {"b": total, "d": product}], "a d b c"),
        (["b", {"c": total, "d": product}], "a b d c"),
        ([{"c": total}, "a", {"d": product}], "a d b c"),
        # an anchor alone changes nothing
        (["a"], "a b c"),
        # removals
        (["b", {"d": total, "a": None}], "b d c"),
        (["b", {"d": total}, {"b": None}], "a d c"),
    ])
    def test_order_of_three_columns(self, tab, specification, columns):
        assert " ".join(transform_columns(tab, specification).columns) == columns

    @pytest.mark.parametrize("specification, columns", [
        (["c", {"x": one}, "a", {"y": one}], "a y b c x d"),
        (["b", {"x": one}, "a", {"y": one}], "a y b x c d"),
        (["c", {"x": one}, "b", {"y": one}, "a", {"z": one}], "a z b y c x d"),
        ([{"x": one}, "c", {"y": one}, "a", {"z": one}], "x a z b c y d"),
        (["b", {"x": one}, "b", {"y": one}], "a b y x c d"),
        (["b", {"x": one, "y": one}, "b", {"z": one}], "a b z x y c d"),
        (["a", "b", {"x": one}], "a b x c d"),
        (["b", "a", {"x": one}], "a x b c d"),
        ([{"x": one}, "d"], "x a b c d"),
        (["c", {"x": one}, "d"], "a b c x d"),
        (["b", {"x": one}, "d"], "a b x c d"),
    ])
    def test_order_of_four_columns(self, one_row, specification, columns):
        assert " ".join(transform_columns(one_row, specification).columns) == columns


# ====================================== 6. Semantics (verified against the original)

class TestSemantics:

    def test_transformations_see_the_original_columns(self, tab):
        # TransformColumns[tab, {"a" -> Function[2 #a], "b" -> Function[#a + 1]}]
        assert_table(
            transform_columns(tab, {"a": lambda r: 2 * r["a"], "b": lambda r: r["a"] + 1}),
            ["a", "b", "c"], [[4, 3, 5], [20, 11, 4], [14, 8, 7]],
        )

    def test_new_column_cannot_be_used_by_another(self, tab):
        # TransformColumns[tab, {"d" -> Function[#a + #b], "e" -> Function[2 #d]}] fails:
        # "The row function expects a key #d that is not in the column keys"
        with pytest.raises(KeyError, match="d"):
            transform_columns(tab, {"d": total, "e": lambda r: 2 * r["d"]})

    def test_removed_column_can_still_be_used(self, tab):
        # TransformColumns[tab, {"b" -> Nothing, "d" -> Function[#b + 1]}]
        assert_table(
            transform_columns(tab, {"b": None, "d": lambda r: r["b"] + 1}),
            ["a", "c", "d"], [[2, 5, 9], [10, 4, 1], [7, 7, 1]],
        )

    def test_existing_column_is_replaced_in_place(self, tab):
        # ColumnKeys[TransformColumns[tab, {"d" -> Function[#a + #b], "a" -> Function[-#a]}]]
        assert_table(
            transform_columns(tab, {"d": total, "a": lambda r: -r["a"]}),
            ["a", "b", "c", "d"], [[-2, 8, 5, 10], [-10, 0, 4, 10], [-7, 0, 7, 7]],
        )

    def test_last_of_repeated_transformations_wins(self, tab):
        # TransformColumns[tab, {"a" -> Function[#a + #b], "a" -> Function[#a #c]}]
        assert_table(
            transform_columns(tab, [{"a": total}, {"a": product}]),
            ["a", "b", "c"], [[10, 8, 5], [40, 0, 4], [49, 0, 7]],
        )
        # TransformColumns[tab, {"d" -> Function[#a + #b], "d" -> Function[#a #c]}]
        assert_table(
            transform_columns(tab, [{"d": total}, {"d": product}]),
            ["a", "b", "c", "d"], [[2, 8, 5, 10], [10, 0, 4, 40], [7, 0, 7, 49]],
        )

    def test_removal_and_addition_follow_the_given_order(self, tab):
        # {"a" -> Nothing, "a" -> f}: the column comes back as a new one, at the end
        assert list(transform_columns(tab, [{"a": None}, {"a": total}]).columns) == ["b", "c", "a"]
        # {"a" -> f, "a" -> Nothing}
        assert list(transform_columns(tab, [{"a": total}, {"a": None}]).columns) == ["b", "c"]
        # {"d" -> f, "d" -> Nothing}
        assert list(transform_columns(tab, [{"d": total}, {"d": None}]).columns) == ["a", "b", "c"]

    def test_remove_all_columns(self, tab):
        # TransformColumns[tab, {"a" -> Nothing, "b" -> Nothing, "c" -> Nothing}]
        result = transform_columns(tab, {"a": None, "b": None, "c": None})
        assert result.shape == (3, 0)
        assert list(result.index) == [0, 1, 2]

    def test_list_valued_cells(self, tab):
        # TransformColumns[tab, "d" -> Function[{#a, #b}]]
        result = transform_columns(tab, {"d": lambda r: [r["a"], r["b"]]})
        assert result["d"].tolist() == [[2, 8], [10, 0], [7, 0]]

    def test_constant_column(self, tab):
        # TransformColumns[tab, "d" -> (7 &)]
        assert transform_columns(tab, {"d": lambda r: 7})["d"].tolist() == [7, 7, 7]

    def test_row_and_columnwise_terms_together(self, tab):
        # TransformColumns[tab, "d" -> Function[
        #   #a + ColumnwiseValue[Total[#a]] + ColumnwiseThread[Reverse[#b]]]]
        sum_a = ColumnwiseValue(lambda t: t["a"].sum())
        b_backwards = ColumnwiseThread(lambda t: t["b"][::-1])
        result = transform_columns(tab, {"d": lambda r: r["a"] + sum_a + b_backwards})
        assert result["d"].tolist() == [21, 29, 34]

    def test_columnwise_value_can_be_a_list(self, tab):
        # TransformColumns[tab, "d" -> Function[ColumnwiseValue[Accumulate[#a]]]]
        running = ColumnwiseValue(lambda t: t["a"].cumsum().tolist())
        result = transform_columns(tab, {"d": lambda r: running})
        assert result["d"].tolist() == [[2, 12, 19]] * 3

    def test_columnwise_thread_sorting_a_column(self, tab):
        # TransformColumns[tab, "a" -> Function[ColumnwiseThread[Sort[#a]]]]
        in_order = ColumnwiseThread(lambda t: t["a"].sort_values())
        assert_table(
            transform_columns(tab, {"a": lambda r: in_order}),
            ["a", "b", "c"], [[2, 8, 5], [7, 0, 4], [10, 0, 7]],
        )

    def test_columnwise_value_of_the_median(self, tab):
        # TransformColumns[tab, "a" -> Function[2 #a - ColumnwiseValue[Median[#b]]]]
        median_b = ColumnwiseValue(lambda t: t["b"].median())
        assert_table(
            transform_columns(tab, {"a": lambda r: 2 * r["a"] - median_b}),
            ["a", "b", "c"], [[4.0, 8, 5], [20.0, 0, 4], [14.0, 0, 7]],
        )

    def test_columnwise_thread_added_to_the_rows(self, tab):
        # TransformColumns[tab, "b" -> Function[#b + ColumnwiseThread[Accumulate[#b]]]]
        running_b = ColumnwiseThread(lambda t: t["b"].cumsum())
        assert_table(
            transform_columns(tab, {"b": lambda r: r["b"] + running_b}),
            ["a", "b", "c"], [[2, 16, 5], [10, 8, 4], [7, 8, 7]],
        )


# ====================================== 7. Rows

class TestRows:

    def test_row_values_by_attribute(self, tab):
        assert_table(
            transform_columns(tab, {"d": lambda r: r.a + r.b}),
            ["a", "b", "c", "d"], [[2, 8, 5, 10], [10, 0, 4, 10], [7, 0, 7, 7]],
        )

    def test_row_is_a_read_only_mapping(self, tab):
        seen = []
        transform_columns(tab, {"d": lambda r: seen.append(r)})
        assert seen == [{"a": 2, "b": 8, "c": 5}, {"a": 10, "b": 0, "c": 4}, {"a": 7, "b": 0, "c": 7}]
        first = seen[0]
        assert list(first) == ["a", "b", "c"] and len(first) == 3
        assert "a" in first and "zzz" not in first
        assert first.get("b") == 8 and first.get("zzz", -1) == -1
        assert list(first.items()) == [("a", 2), ("b", 8), ("c", 5)]
        assert sum(first.values()) == 15
        assert dict(first) == {"a": 2, "b": 8, "c": 5}
        assert "{a}-{c}".format(**first) == "2-5"
        assert repr(first) == "Row({'a': 2, 'b': 8, 'c': 5})"
        with pytest.raises(TypeError):
            first["a"] = 0

    def test_row_given_back_as_a_cell(self, tab):
        result = transform_columns(tab, {"d": lambda r: dict(r)})
        assert result["d"].tolist() == [
            {"a": 2, "b": 8, "c": 5}, {"a": 10, "b": 0, "c": 4}, {"a": 7, "b": 0, "c": 7}]

    def test_unknown_column(self, tab):
        with pytest.raises(KeyError, match="zzz"):
            transform_columns(tab, {"d": lambda r: r["zzz"]})

    def test_labels_that_are_not_identifiers(self):
        tab = pd.DataFrame([[1, 2, 3]], columns=["unit price", "_hidden", 7])
        result = transform_columns(tab, {"sum": lambda r: r["unit price"] + r["_hidden"] + r[7]})
        assert result["sum"].tolist() == [6]
        with pytest.raises(AttributeError, match="_hidden"):
            transform_columns(tab, {"sum": lambda r: r._hidden})

    def test_missing_attribute(self, tab):
        with pytest.raises(AttributeError, match="zzz"):
            transform_columns(tab, {"d": lambda r: r.zzz})

    def test_row_values_are_plain_python_objects(self):
        tab = pd.DataFrame({"n": [1, 2], "x": [1.5, 2.5], "s": ["u", "v"], "f": [True, False]})
        kinds = []
        transform_columns(tab, {"k": lambda r: kinds.append([type(v) for v in r.values()])})
        assert kinds == [[int, float, str, bool]] * 2

    def test_copy_of_a_row_can_be_changed(self, tab):
        def greedy(r):
            row = dict(r)
            row["b"] = 1000
            return row["a"] + row["b"]

        result = transform_columns(tab, {"d": greedy, "e": lambda r: r["b"]})
        assert result["d"].tolist() == [1002, 1010, 1007]
        assert result["e"].tolist() == [8, 0, 0]
        assert result["b"].tolist() == [8, 0, 0]

    def test_rows_of_a_wide_table(self):
        tab = pd.DataFrame([range(200), range(1, 201)], columns=[f"c{n}" for n in range(200)])
        result = transform_columns(tab, {"s": lambda r: r["c0"] + r.c199, "n": lambda r: len(r)})
        assert result["s"].tolist() == [199, 201]
        assert result["n"].tolist() == [200, 200]

    def test_missing_values(self):
        tab = pd.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, 30.0]})
        result = transform_columns(
            tab, {"a": lambda r: r["b"] if pd.isna(r["a"]) else r["a"]}
        )
        assert result["a"].tolist() == [1.0, 20.0, 3.0]


# ====================================== 8. Markers given as transformations

class TestMarkersAsTransformations:

    def test_thread(self, tab):
        result = transform_columns(tab, {"total": ColumnwiseThread(lambda t: t["a"].cumsum())})
        assert_table(result, ["a", "b", "c", "total"],
                     [[2, 8, 5, 2], [10, 0, 4, 12], [7, 0, 7, 19]])

    def test_thread_does_not_call_a_function_on_the_rows(self, tab):
        calls = []

        def doubled(t):
            calls.append(len(t))
            return 2 * t["a"]

        result = transform_columns(tab, {"a": ColumnwiseThread(doubled)})
        assert result["a"].tolist() == [4, 20, 14]
        assert calls == [3]

    def test_value(self, tab):
        result = transform_columns(tab, {"largest": ColumnwiseValue(lambda t: t["a"].max())})
        assert result["largest"].tolist() == [10, 10, 10]

    def test_thread_is_matched_by_position(self, tab):
        # the index of the returned Series plays no role
        tab.index = ["x", "y", "z"]
        result = transform_columns(tab, {"a": ColumnwiseThread(lambda t: t["a"].sort_values())})
        assert result["a"].tolist() == [2, 7, 10]
        assert list(result.index) == ["x", "y", "z"]

    def test_result_does_not_share_data_with_the_input(self, tab):
        result = transform_columns(tab, {"d": ColumnwiseThread(lambda t: t["a"])})
        result.loc[0, "d"] = -1
        assert tab["a"].tolist() == [2, 10, 7]

    def test_thread_keeps_the_type_of_its_result(self, tab):
        result = transform_columns(tab, {
            "kind": ColumnwiseThread(lambda t: pd.Categorical(["u", "v", "u"])),
            "half": ColumnwiseThread(lambda t: (t["a"] / 2).to_numpy()),
            "small": ColumnwiseThread(lambda t: t["a"].astype("int8")),
        })
        assert result["kind"].tolist() == ["u", "v", "u"]
        assert result["half"].dtype == "float64"
        assert result["small"].dtype == "int8"

    def test_thread_of_lists_and_generators(self, tab):
        result = transform_columns(tab, {
            "list": ColumnwiseThread(lambda t: [1, 2, 3]),
            "tuple": ColumnwiseThread(lambda t: (1, 2, 3)),
            "range": ColumnwiseThread(lambda t: range(3)),
            "generator": ColumnwiseThread(lambda t: (2 * v for v in t["a"])),
        })
        assert result["list"].tolist() == [1, 2, 3]
        assert result["tuple"].tolist() == [1, 2, 3]
        assert result["range"].tolist() == [0, 1, 2]
        assert result["generator"].tolist() == [4, 20, 14]

    def test_thread_of_vectors(self, tab):
        result = transform_columns(
            tab, {"pair": ColumnwiseThread(lambda t: t[["a", "b"]].to_numpy())}
        )
        assert [list(pair) for pair in result["pair"]] == [[2, 8], [10, 0], [7, 0]]

    def test_many_rows(self):
        generator = np.random.default_rng(1)
        tab = pd.DataFrame(generator.integers(0, 1000, size=(50000, 3)), columns=list("abc"))
        result = transform_columns(tab, {
            "a": ColumnwiseThread(lambda t: 2 * t["a"]),
            "d": ColumnwiseThread(lambda t: t["b"] - t["c"].mean()),
        })
        expected = tab.assign(a=2 * tab["a"], d=tab["b"] - tab["c"].mean())
        pd.testing.assert_frame_equal(result, expected)


# ====================================== 9. Tables

class TestTables:

    def test_input_is_not_modified(self, tab):
        before = tab.copy()
        transform_columns(tab, ["a", {"d": total, "a": product, "b": None}])
        pd.testing.assert_frame_equal(tab, before)

    def test_index_and_attributes_are_kept(self):
        tab = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=pd.Index(["x", "y"], name="key"))
        tab.attrs = {"source": "test"}
        for specification in ({"a": lambda r: -r["a"]}, {"c": total}, {"b": None},
                              [{"c": total}, "a"]):
            result = transform_columns(tab, specification)
            assert result.index.equals(tab.index) and result.index.name == "key"
            assert result.attrs == {"source": "test"}

    def test_index_with_repeated_labels(self):
        tab = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "x", "y"])
        result = transform_columns(tab, {"a": total, "c": ColumnwiseThread(lambda t: t["a"] * 2)})
        assert result["a"].tolist() == [5, 7, 9]
        assert result["c"].tolist() == [2, 4, 6]
        assert list(result.index) == ["x", "x", "y"]

    def test_untouched_columns_keep_their_type(self):
        tab = pd.DataFrame({
            "a": pd.array([1, 2, 3], dtype="int8"),
            "when": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "kind": pd.Categorical(["u", "v", "u"]),
        })
        result = transform_columns(tab, [{"b": lambda r: 2 * r["a"]}, "a"])
        assert list(result.columns) == ["b", "a", "when", "kind"]
        assert result[["a", "when", "kind"]].dtypes.tolist() == tab.dtypes.tolist()

    def test_type_of_a_computed_column(self, tab):
        result = transform_columns(tab, {
            "a": lambda r: r["a"] / 2, "s": lambda r: str(r["b"]), "t": lambda r: r["c"] > 4,
        })
        assert result["a"].dtype == "float64"
        assert result["b"].dtype == "int64"
        assert result["s"].tolist() == ["8", "0", "0"]
        assert result["t"].dtype == "bool"

    def test_result_can_be_modified(self, tab):
        result = transform_columns(tab, [{"d": total, "b": None}, "a"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result["e"] = 0
            result.iloc[0, 0] = -1
        assert tab.iloc[0, 0] == 2

    def test_table_without_rows(self):
        tab = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="float64")})
        result = transform_columns(tab, {"a": lambda r: 2 * r["a"], "c": total, "b": None})
        assert list(result.columns) == ["a", "c"]
        assert len(result) == 0
        assert result["a"].dtype == "int64"

    def test_table_without_columns(self):
        tab = pd.DataFrame(index=[10, 20, 30])
        result = transform_columns(tab, {"c": lambda r: len(r)})
        assert result["c"].tolist() == [0, 0, 0]
        assert list(result.index) == [10, 20, 30]

    def test_empty_specification(self, tab):
        for specification in ({}, []):
            result = transform_columns(tab, specification)
            pd.testing.assert_frame_equal(result, tab)
            assert result is not tab


# ====================================== 10. Errors

class TestErrors:

    def test_transformation_that_is_not_a_function(self, tab):
        # In Wolfram Language "e" -> 50 applies 50 as a function: a constant is (50 &)
        with pytest.raises(TypeError, match="column 'e'"):
            transform_columns(tab, {"e": 50})

    def test_unknown_anchor(self, tab):
        with pytest.raises(KeyError, match="anchor"):
            transform_columns(tab, ["zzz", {"d": total}])

    def test_anchor_that_is_a_function(self, tab):
        # without column names Wolfram Language takes functions alone; a DataFrame
        # always has column labels
        with pytest.raises(KeyError, match="anchor"):
            transform_columns(tab, [total])

    def test_unknown_column_to_remove(self, tab):
        with pytest.raises(KeyError, match="remove"):
            transform_columns(tab, {"zzz": None})

    @pytest.mark.parametrize("specification", [5, "a", lambda r: 1, {"a"}])
    def test_invalid_specification(self, tab, specification):
        with pytest.raises(TypeError, match="dictionary"):
            transform_columns(tab, specification)
        with pytest.raises(TypeError, match="dictionary"):
            transform_columns(specification)

    def test_missing_specification(self, tab):
        with pytest.raises(ValueError, match="required"):
            transform_columns(tab)

    def test_not_a_table(self):
        with pytest.raises(TypeError, match="DataFrame"):
            transform_columns([[1, 2]], {"a": one})

    def test_repeated_column_labels(self):
        tab = pd.DataFrame([[1, 2]], columns=["a", "a"])
        with pytest.raises(ValueError, match="repeated column labels"):
            transform_columns(tab, {"b": one})

    def test_columnwise_thread_of_the_wrong_length(self, tab):
        # TransformColumns::ctrlen: "... has length inconsistent with the number of rows, 3"
        with pytest.raises(ValueError, match="inconsistent with the number of rows, 3"):
            transform_columns(tab, {"d": ColumnwiseThread(lambda t: [1, 2])})

    @pytest.mark.parametrize("result", [19, 2.5, "abc", {"a": 1}, {1, 2, 3}, None])
    def test_columnwise_thread_that_is_not_a_column(self, tab, result):
        with pytest.raises(ValueError, match="one value per row"):
            transform_columns(tab, {"d": lambda r: ColumnwiseThread(lambda t: result)})

    def test_columnwise_thread_returning_a_table(self, tab):
        with pytest.raises(ValueError, match="one value per row"):
            transform_columns(tab, {"d": ColumnwiseThread(lambda t: t)})
