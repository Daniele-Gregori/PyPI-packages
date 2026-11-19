"""Benchmark the Python transform-tabular functions against their Wolfram
Language originals: the resource functions TransformTabular and TransformRows
and the built-in function TransformColumns.

Run from the package root (packages/transform-tabular):

    python benchmark/bench_python.py

and compare with the Wolfram side:

    wolframscript -file benchmark/bench_wolfram.wl

Both scripts build the same tables, run the same cases and print their
results as JSON; see BENCHMARK.md for the collected results.

Every case is run here in the "rowwise" style, which translates the Wolfram
code literally (a Python function called on every element or row).  Where a
case can also be written with whole-column operations, it is run a second
time in that "columnwise" style.
"""

import json
import platform
import sys
import time
from itertools import accumulate
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

import transform_tabular as package
from transform_tabular import (
    ColumnwiseThread,
    ColumnwiseValue,
    transform_columns,
    transform_rows,
    transform_tabular,
)

MAX_SECONDS = 2.0
MAX_REPS = 200


def table(rows):
    """A table of integers with columns "a" to "j", the same as on the Wolfram side."""
    i = np.arange(1, rows + 1)[:, None]
    j = np.arange(1, 11)[None, :]
    return pd.DataFrame((7919 * i + 104729 * j) % 1000003, columns=list("abcdefghij"))


def selection(count, rows):
    """Positions of *count* scattered rows (the Wolfram ones, minus 1)."""
    return np.unique((7919 * np.arange(1, count + 1)) % rows)


def build_cases():
    small, large = table(10**5), table(10**6)
    rows_small, rows_large = selection(10**4, 10**5), selection(10**5, 10**6)

    minimum = ColumnwiseValue(min)
    running = ColumnwiseThread(lambda col: list(accumulate(col)))
    median_b = ColumnwiseValue(lambda t: t["b"].median())
    double_a = {"a": lambda r: 2 * r["a"]}
    double_a_columnwise = {"a": ColumnwiseThread(lambda t: 2 * t["a"])}

    rowwise = [
        ("tabular-elementwise",
         lambda: transform_tabular(small, lambda x: x + 1)),
        ("tabular-columnwise-value",
         lambda: transform_tabular(small, lambda x: x - minimum)),
        ("tabular-columnwise-thread",
         lambda: transform_tabular(small, lambda x: running)),
        ("columns-arithmetic",
         lambda: transform_columns(large, double_a)),
        ("columns-new-column",
         lambda: transform_columns(large, {"k": lambda r: r["a"] + r["b"]})),
        ("columns-columnwise-value",
         lambda: transform_columns(large, {"a": lambda r: r["a"] - median_b})),
        ("columns-strings",
         lambda: transform_columns(small, {"s": lambda r: f"{r['a']}-{r['b']}"})),
        ("rows-list",
         lambda: transform_rows(large, rows_large, double_a)),
        ("rows-list-unordered",
         lambda: transform_rows(large, rows_large, double_a, keep_order=False)),
        ("rows-criterion",
         lambda: transform_rows(small, lambda r: r["c"] % 2 == 0,
                                {"a": lambda r: 2 * r["a"] + 2 * r["b"]})),
        ("rows-all-cells",
         lambda: transform_rows(small, rows_small, lambda x: x ** 2)),
    ]
    columnwise = [
        ("columns-arithmetic",
         lambda: transform_columns(large, double_a_columnwise)),
        ("columns-new-column",
         lambda: transform_columns(large, {"k": ColumnwiseThread(lambda t: t["a"] + t["b"])})),
        ("columns-columnwise-value",
         lambda: transform_columns(
             large, {"a": ColumnwiseThread(lambda t: t["a"] - t["b"].median())})),
        ("columns-strings",
         lambda: transform_columns(small, {"s": ColumnwiseThread(
             lambda t: t["a"].astype(str) + "-" + t["b"].astype(str))})),
        ("rows-list",
         lambda: transform_rows(large, rows_large, double_a_columnwise)),
        ("rows-list-unordered",
         lambda: transform_rows(large, rows_large, double_a_columnwise, keep_order=False)),
        ("rows-criterion",
         lambda: transform_rows(small, small["c"] % 2 == 0, {"a": ColumnwiseThread(
             lambda t: 2 * t["a"] + 2 * t["b"])})),
    ]
    return [("rowwise", rowwise), ("columnwise", columnwise)]


def checksum(result):
    """Total of the numeric cells, plus the total length of the strings."""
    total = 0.0
    for label in result.columns:
        column = result[label]
        if pd.api.types.is_numeric_dtype(column):
            total += float(column.sum())
        else:
            total += float(column.str.len().sum())
    return total


def time_case(thunk):
    result = thunk()  # warm-up
    times = []
    start = time.perf_counter()
    while time.perf_counter() - start < MAX_SECONDS and len(times) < MAX_REPS:
        t0 = time.perf_counter()
        thunk()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), len(times), result


def main():
    results = []
    for style, cases in build_cases():
        for name, thunk in cases:
            mean, reps, result = time_case(thunk)
            results.append({
                "case": name,
                "style": style,
                "ms": round(mean * 1000, 3),
                "reps": reps,
                "value": checksum(result),
                "head": [float(v) for v in result["a"].head(3)],
            })
    print(json.dumps({
        "python": platform.python_version(),
        "transform-tabular": package.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
