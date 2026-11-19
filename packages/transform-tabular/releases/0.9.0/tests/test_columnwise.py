"""Tests for the columnwise markers ColumnwiseValue and ColumnwiseThread."""

import math
import operator
import threading
import time
from functools import partial
from itertools import accumulate

import numpy as np
import pandas as pd
import pytest

from transform_tabular import (
    ColumnwiseThread,
    ColumnwiseValue,
    transform_columns,
    transform_rows,
    transform_tabular,
)

# Markers defined at module level: the functions below reach them as global variables.
# (Version 0.8.0 only recognised the markers that a function held in its closure.)
GLOBAL_MEAN = ColumnwiseValue(lambda col: sum(col) / len(col))
GLOBAL_RUNNING = ColumnwiseThread(lambda col: list(accumulate(col)))
GLOBAL_CALLS = []


def minus_global_mean(x):
    return x - GLOBAL_MEAN


class Counter:
    """A function that counts its calls."""

    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, argument):
        self.calls += 1
        return self.func(argument)


@pytest.fixture
def tab():
    return pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 60]})


def inside(thunk):
    """The value of ``thunk()`` evaluated while a transformation is running."""
    values = []
    transform_tabular(pd.DataFrame({"a": [7, 7]}), lambda x: values.append(thunk()))
    return values[0]


# ====================================== 1. Where the markers are defined

class TestWhereMarkersAreDefined:

    def test_global_variable_in_a_lambda(self, tab):
        result = transform_tabular(tab, lambda x: x - GLOBAL_MEAN)
        assert result["a"].tolist() == [-1.0, 0.0, 1.0]
        assert result["b"].tolist() == [-20.0, -10.0, 30.0]

    def test_global_variable_in_a_named_function(self, tab):
        assert transform_tabular(tab, minus_global_mean)["a"].tolist() == [-1.0, 0.0, 1.0]

    def test_global_thread(self, tab):
        result = transform_tabular(tab, lambda x: GLOBAL_RUNNING)
        assert result["a"].tolist() == [1, 3, 6]
        assert result["b"].tolist() == [10, 30, 90]

    def test_local_variable(self, tab):
        largest = ColumnwiseValue(max)
        assert transform_tabular(tab, lambda x: x / largest)["b"].tolist() == [1 / 6, 1 / 3, 1.0]

    def test_attribute_and_container(self, tab):
        class Markers:
            largest = ColumnwiseValue(max)

        markers = {"smallest": ColumnwiseValue(min)}
        result = transform_tabular(
            tab, lambda x: (x - markers["smallest"]) / (Markers.largest - markers["smallest"])
        )
        assert result["a"].tolist() == [0.0, 0.5, 1.0]
        assert result["b"].tolist() == [0.0, 0.2, 1.0]

    def test_default_argument_and_partial(self, tab):
        largest = ColumnwiseValue(max)
        assert transform_tabular(tab, lambda x, m=largest: m - x)["a"].tolist() == [2, 1, 0]
        assert transform_tabular(tab, partial(operator.sub, largest))["a"].tolist() == [2, 1, 0]

    def test_inside_a_helper_function(self, tab):
        largest = ColumnwiseValue(max)

        def distance(x):
            return largest - x

        assert transform_tabular(tab, lambda x: 2 * distance(x))["a"].tolist() == [4, 2, 0]

    def test_created_in_the_function(self, tab):
        # the form closest to the Wolfram one: # - ColumnwiseValue[Min[#]] &
        result = transform_tabular(tab, lambda x: x - ColumnwiseValue(lambda col: min(col)))
        assert result["a"].tolist() == [0, 1, 2]
        assert result["b"].tolist() == [0, 10, 50]

    def test_created_in_the_function_from_a_builtin(self, tab):
        assert transform_tabular(tab, lambda x: x - ColumnwiseValue(min))["b"].tolist() == [0, 10, 50]
        result = transform_tabular(tab, lambda x: ColumnwiseThread(sorted, ) - x)
        assert result["a"].tolist() == [0, 0, 0]

    def test_all_three_functions(self, tab):
        total_b = ColumnwiseValue(lambda t: t["b"].sum())
        share = {"share": lambda r: r["b"] / total_b}
        assert transform_columns(tab, share)["share"].tolist() == [1 / 9, 2 / 9, 6 / 9]
        result = transform_rows(tab, [2], share)
        assert result["share"].iloc[2] == 6 / 9


# ====================================== 2. Evaluated once

class TestEvaluatedOnce:

    def test_value_once_per_column(self, tab):
        largest = Counter(max)
        marker = ColumnwiseValue(largest)
        transform_tabular(tab, lambda x: x - marker)
        assert largest.calls == 2

    def test_thread_once_per_column(self, tab):
        ordered = Counter(sorted)
        marker = ColumnwiseThread(ordered)
        transform_tabular(tab, lambda x: x - marker)
        assert ordered.calls == 2

    def test_used_twice_in_an_expression(self, tab):
        smallest = Counter(min)
        marker = ColumnwiseValue(smallest)
        result = transform_tabular(tab, lambda x: (x - marker) * marker, "a")
        assert result["a"].tolist() == [0, 1, 2]
        assert smallest.calls == 1

    def test_once_for_all_the_columns_of_transform_columns(self, tab):
        total_b = Counter(lambda t: t["b"].sum())
        marker = ColumnwiseValue(total_b)
        transform_columns(tab, {"a": lambda r: r["a"] / marker, "b": lambda r: r["b"] / marker})
        assert total_b.calls == 1

    def test_not_evaluated_when_not_used(self, tab):
        largest = Counter(max)
        marker = ColumnwiseValue(largest)
        transform_tabular(tab, lambda x: x if x < 100 else marker)
        assert largest.calls == 0

    def test_created_in_the_function_from_one_function_object(self, tab):
        smallest = Counter(min)
        result = transform_tabular(tab, lambda x: x - ColumnwiseValue(smallest))
        assert result["b"].tolist() == [0, 10, 50]
        assert smallest.calls == 2

    def test_created_in_the_function_from_a_lambda(self, tab):
        del GLOBAL_CALLS[:]
        result = transform_tabular(
            tab, lambda x: x - ColumnwiseValue(lambda col: (GLOBAL_CALLS.append(0), min(col))[1])
        )
        assert result["b"].tolist() == [0, 10, 50]
        assert len(GLOBAL_CALLS) == 2

    def test_created_in_the_function_from_a_closure(self, tab):
        calls = []
        offset = 5
        result = transform_tabular(
            tab, lambda x: x - ColumnwiseThread(lambda col: [calls.append(0) or v - offset for v in col])
        )
        assert result["a"].tolist() == [5, 5, 5]
        # one call per element of a column, for each of the two columns: evaluated once
        assert len(calls) == 6

    def test_created_in_the_function_and_depending_on_the_row(self, tab):
        # the function of the marker holds the element: it cannot be evaluated once only
        calls = []
        result = transform_tabular(
            tab, lambda x: ColumnwiseValue(lambda col: (calls.append(0), x + max(col))[1]), "a"
        )
        assert result["a"].tolist() == [4, 5, 6]
        assert len(calls) == 3

    def test_two_markers_from_equal_lambdas(self, tab):
        del GLOBAL_CALLS[:]
        result = transform_tabular(
            tab,
            lambda x: ColumnwiseValue(lambda col: (GLOBAL_CALLS.append(0), max(col))[1])
            - ColumnwiseValue(lambda col: (GLOBAL_CALLS.append(0), max(col))[1]),
            "a",
        )
        assert result["a"].tolist() == [0, 0, 0]
        assert len(GLOBAL_CALLS) == 2

    def test_bound_methods_of_different_objects(self, tab):
        class Shift:
            def __init__(self, amount):
                self.amount = amount

            def largest(self, col):
                return max(col) + self.amount

        one, two = Shift(1), Shift(2)
        result = transform_tabular(
            tab, lambda x: ColumnwiseValue(one.largest) * 10 + ColumnwiseValue(two.largest), "a"
        )
        assert result["a"].tolist() == [45, 45, 45]

    def test_value_and_thread_of_the_same_function(self, tab):
        result = transform_tabular(
            tab, lambda x: [ColumnwiseValue(sorted).value, ColumnwiseThread(sorted).value], "a"
        )
        assert result["a"].tolist() == [[[1, 2, 3], 1], [[1, 2, 3], 2], [[1, 2, 3], 3]]

    def test_recomputed_for_every_table(self):
        largest = ColumnwiseValue(max)

        def below(x):
            return x - largest

        assert transform_tabular(pd.DataFrame({"a": [1, 5]}), below)["a"].tolist() == [-4, 0]
        assert transform_tabular(pd.DataFrame({"a": [1, 9]}), below)["a"].tolist() == [-8, 0]


# ====================================== 3. Markers as objects

class TestMarkerObjects:

    def test_value_inside_a_transformation(self, tab):
        largest = ColumnwiseValue(max)
        result = transform_tabular(tab, lambda x: math.gcd(x, largest.value), "b")
        assert result["b"].tolist() == [10, 20, 60]

    def test_value_of_a_thread_is_the_element_of_the_row(self, tab):
        ordered = ColumnwiseThread(lambda col: sorted(col, reverse=True))
        result = transform_tabular(tab, lambda x: str(ordered.value), "a")
        assert result["a"].tolist() == ["3", "2", "1"]

    def test_function_result_can_be_the_marker(self, tab):
        largest = ColumnwiseValue(max)
        assert transform_tabular(tab, lambda x: largest)["b"].tolist() == [60, 60, 60]

    @pytest.mark.parametrize("use", [
        lambda m: m.value, lambda m: m + 1, lambda m: 1 - m, lambda m: m < 1,
        lambda m: -m, lambda m: float(m), lambda m: round(m),
    ])
    def test_no_value_outside_a_transformation(self, use):
        for marker in (ColumnwiseValue(max), ColumnwiseThread(sorted)):
            with pytest.raises(RuntimeError, match="inside transform_tabular"):
                use(marker)

    def test_plain_object_outside_a_transformation(self):
        first, second = ColumnwiseValue(max), ColumnwiseValue(max)
        assert first == first and first != second
        assert first in [second, first]
        assert len({first, second, first}) == 2
        assert bool(first) is True
        assert repr(first) == f"ColumnwiseValue({max!r})"
        assert repr(ColumnwiseThread(sorted)) == f"ColumnwiseThread({sorted!r})"
        assert not callable(first)

    def test_function_is_required(self):
        with pytest.raises(TypeError, match="expects a function"):
            ColumnwiseValue(5)
        with pytest.raises(TypeError, match="expects a function"):
            ColumnwiseThread([1, 2, 3])


# ====================================== 4. Operators

SEVEN = ColumnwiseValue(lambda col: col[0])
THREE = ColumnwiseValue(lambda col: len(col) + 1)
EACH = ColumnwiseThread(lambda col: col)
REAL = ColumnwiseValue(lambda col: 7.5)

BINARY = [
    operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv,
    operator.mod, operator.pow, divmod, operator.and_, operator.or_, operator.xor,
    operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne,
]


class TestOperators:

    @pytest.mark.parametrize("op", BINARY)
    def test_binary(self, op):
        expected = op(7, 3)
        assert inside(lambda: op(SEVEN, 3)) == expected
        assert inside(lambda: op(7, THREE)) == expected
        assert inside(lambda: op(SEVEN, THREE)) == expected
        assert inside(lambda: op(EACH, 3)) == expected
        assert inside(lambda: op(EACH, THREE)) == expected
        assert type(inside(lambda: op(SEVEN, 3))) is type(expected)

    @pytest.mark.parametrize("op", [operator.neg, operator.pos, abs, operator.invert])
    def test_unary(self, op):
        assert inside(lambda: op(SEVEN)) == op(7)
        assert inside(lambda: op(EACH)) == op(7)

    @pytest.mark.parametrize("convert, expected", [
        (int, 7), (float, 7.5), (complex, 7.5 + 0j), (bool, True), (round, 8),
        (math.trunc, 7), (math.floor, 7), (math.ceil, 8), (math.sqrt, math.sqrt(7.5)),
        (lambda m: round(m, 0), 8.0), (lambda m: 1 if m else 0, 1),
    ])
    def test_conversions(self, convert, expected):
        value = inside(lambda: convert(REAL))
        assert value == expected and type(value) is type(expected)

    def test_index(self):
        assert inside(lambda: "abcdefgh"[SEVEN]) == "h"
        assert inside(lambda: list(range(THREE))) == [0, 1, 2]

    def test_equality_follows_the_value(self):
        assert inside(lambda: SEVEN == 7) is True
        assert inside(lambda: 7 == SEVEN) is True
        assert inside(lambda: SEVEN != 7) is False
        assert inside(lambda: SEVEN == EACH) is True
        assert inside(lambda: SEVEN == THREE) is False
        assert inside(lambda: {SEVEN: "kept"}[SEVEN]) == "kept"

    def test_strings(self):
        suffix = ColumnwiseValue(lambda col: "!" * len(col))
        tab = pd.DataFrame({"s": ["a", "b"]})
        assert transform_tabular(tab, lambda x: x + suffix)["s"].tolist() == ["a!!", "b!!"]
        assert transform_tabular(tab, lambda x: suffix + x)["s"].tolist() == ["!!a", "!!b"]

    def test_numpy_and_pandas_operands(self):
        value = inside(lambda: np.float64(2.0) * SEVEN)
        assert value == 14.0 and isinstance(value, float)
        assert inside(lambda: (np.array([1, 2]) + SEVEN).tolist()) == [8, 9]
        assert inside(lambda: (SEVEN - np.array([1, 2])).tolist()) == [6, 5]
        assert inside(lambda: (pd.Series([1, 2]) * SEVEN).tolist()) == [7, 14]


# ====================================== 5. ColumnwiseThread

class TestColumnwiseThread:

    @pytest.mark.parametrize("build", [
        lambda col: [v + 1 for v in col],
        lambda col: tuple(v + 1 for v in col),
        lambda col: (v + 1 for v in col),
        lambda col: map(lambda v: v + 1, col),
        lambda col: np.array(col) + 1,
        lambda col: pd.Series(col) + 1,
        lambda col: pd.Index(col) + 1,
        lambda col: pd.Series(col, index=[2, 0, 1]) + 1,
    ])
    def test_kinds_of_columns(self, tab, build):
        marker = ColumnwiseThread(build)
        for function in (lambda x: marker, lambda x: marker + 0, marker):
            assert transform_tabular(tab, function, "a")["a"].tolist() == [2, 3, 4]

    def test_elements_are_plain_python_objects(self, tab):
        kinds = []
        for build in (lambda col: np.array(col), lambda col: pd.Series(col)):
            marker = ColumnwiseThread(build)
            transform_tabular(tab, lambda x: kinds.append(type(marker.value)), "a")
        assert set(kinds) == {int}

    def test_wrong_length(self, tab):
        marker = ColumnwiseThread(lambda col: col[:-1])
        with pytest.raises(ValueError, match="length 2, inconsistent with the number of rows, 3"):
            transform_tabular(tab, lambda x: marker)

    def test_not_a_column(self, tab):
        marker = ColumnwiseThread(lambda col: sum(col))
        with pytest.raises(ValueError, match="one value per row"):
            transform_tabular(tab, lambda x: x - marker)

    def test_function_receives_a_list(self, tab):
        seen = []
        marker = ColumnwiseThread(lambda col: seen.append(col) or col)
        transform_tabular(tab, lambda x: marker)
        assert seen == [[1, 2, 3], [10, 20, 60]]

    def test_function_may_change_its_argument(self, tab):
        def in_place(col):
            col.sort(reverse=True)
            return col

        marker = ColumnwiseThread(in_place)
        result = transform_tabular(tab, lambda x: x * 100 + marker, "a")
        assert result["a"].tolist() == [103, 202, 301]


# ====================================== 6. Compositions

class TestCompositions:

    def test_value_inside_the_function_of_a_thread(self, tab):
        mean = ColumnwiseValue(lambda col: sum(col) / len(col))
        centred = ColumnwiseThread(lambda col: [v - mean for v in col])
        assert transform_tabular(tab, lambda x: centred, "a")["a"].tolist() == [-1.0, 0.0, 1.0]

    def test_value_inside_the_function_of_a_value(self, tab):
        smallest = ColumnwiseValue(min)
        spread = ColumnwiseValue(lambda col: max(col) - smallest)
        assert transform_tabular(tab, lambda x: spread)["b"].tolist() == [50, 50, 50]

    def test_value_inside_a_function_of_the_table(self, tab):
        mean_a = ColumnwiseValue(lambda t: t["a"].mean())
        centred = ColumnwiseThread(lambda t: t["a"] - mean_a)
        assert transform_columns(tab, {"a": centred})["a"].tolist() == [-1.0, 0.0, 1.0]

    def test_thread_inside_the_function_of_a_value(self, tab):
        ordered = ColumnwiseThread(sorted)
        marker = ColumnwiseValue(lambda col: ordered + 1)
        with pytest.raises(ValueError, match="Mixed compositions"):
            transform_tabular(tab, lambda x: marker)

    def test_thread_inside_the_function_of_a_thread(self, tab):
        ordered = ColumnwiseThread(sorted)
        marker = ColumnwiseThread(lambda col: [ordered + v for v in col])
        with pytest.raises(ValueError, match="Mixed compositions"):
            transform_tabular(tab, lambda x: marker)

    @pytest.mark.parametrize("outer", [ColumnwiseValue, ColumnwiseThread])
    @pytest.mark.parametrize("inner", [ColumnwiseValue(max), ColumnwiseThread(sorted)])
    def test_function_returning_a_marker(self, tab, outer, inner):
        marker = outer(lambda col: inner)
        with pytest.raises(ValueError, match="Mixed compositions"):
            transform_tabular(tab, lambda x: marker)


# ====================================== 7. Scopes

class TestScopes:

    def test_nested_transformations(self):
        smallest = ColumnwiseValue(min)
        inner = pd.DataFrame({"z": [100, 200]})

        def function(x):
            nested = transform_tabular(inner, lambda y: y - smallest)
            return x - smallest + nested["z"].sum()

        result = transform_tabular(pd.DataFrame({"a": [5, 7, 9]}), function)
        assert result["a"].tolist() == [100, 102, 104]

    def test_transformation_inside_the_function_of_a_marker(self):
        largest = ColumnwiseValue(max)
        scaled = ColumnwiseThread(
            lambda t: transform_tabular(t, lambda x: x / largest)["a"]
        )
        tab = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [8.0, 8.0, 8.0]})
        result = transform_columns(tab, {"c": lambda r: r["b"] * scaled})
        assert result["c"].tolist() == [2.0, 4.0, 8.0]

    def test_error_leaves_no_scope_behind(self):
        marker = ColumnwiseValue(max)
        with pytest.raises(ZeroDivisionError):
            transform_tabular(pd.DataFrame({"a": [1, 0]}), lambda x: 1 / x + marker)
        with pytest.raises(RuntimeError):
            marker + 1

    def test_error_in_the_function_of_a_marker(self, tab):
        marker = ColumnwiseValue(lambda col: col[10])
        with pytest.raises(IndexError):
            transform_tabular(tab, lambda x: x + marker)
        with pytest.raises(RuntimeError):
            marker + 1

    def test_threads(self):
        total = ColumnwiseValue(sum)
        running = ColumnwiseThread(lambda col: list(accumulate(col)))
        barrier = threading.Barrier(2)
        tables = [pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"a": [10, 20, 30]})]
        results = [None, None]

        def function(x):
            time.sleep(0.002)
            return total + running

        def work(number):
            barrier.wait()
            results[number] = transform_tabular(tables[number], function)

        workers = [threading.Thread(target=work, args=(number,)) for number in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        assert results[0]["a"].tolist() == [7, 9, 12]
        assert results[1]["a"].tolist() == [70, 90, 120]
