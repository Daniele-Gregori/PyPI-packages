# transform-tabular

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/transform-tabular.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/transform-tabular.yml)
[![PyPI version](https://badge.fury.io/py/transform-tabular.svg)](https://badge.fury.io/py/transform-tabular)
[![Python versions](https://img.shields.io/pypi/pyversions/transform-tabular.svg)](https://pypi.org/project/transform-tabular/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)



Transform pandas DataFrames with plain Python functions, in three ways:

- **`transform_tabular(df, func, columns)`** applies one function to every element of all the columns, or of a selection of them;
- **`transform_columns(df, {column: func})`** modifies existing columns or adds new ones, computing each cell from its row;
- **`transform_rows(df, rows, transformation)`** transforms only a selection of rows, leaving the others as they are.

Two markers bring whole columns into any of these transformations:

- **`ColumnwiseValue(func)`** stands for a value computed once from a whole column, such as a mean or a maximum;
- **`ColumnwiseThread(func)`** stands for a column computed once from a whole column, such as a sort or a cumulative sum, whose elements are matched to the rows.

This package is a Python port of the Wolfram Language resource functions [`TransformTabular`](https://resources.wolframcloud.com/FunctionRepository/resources/TransformTabular/) and `TransformRows`, and of the built-in function [`TransformColumns`](https://reference.wolfram.com/language/ref/TransformColumns.html).


## Installation

```bash
pip install transform-tabular
```


## Usage

```python
from transform_tabular import (
    transform_tabular, transform_columns, transform_rows,
    ColumnwiseValue, ColumnwiseThread,
)
import pandas as pd
```

All the functions return a new DataFrame and leave the given one untouched. The index of the table is kept as it is.


## `transform_tabular`

```python
transform_tabular(df, func)                # apply func to every element of all columns
transform_tabular(df, func, columns)       # apply func only to selected columns
transform_tabular(func)                    # operator form: returns a reusable transformer
transform_tabular(func, columns)           # operator form with column selection
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame` | Input DataFrame |
| `func` | callable | Function applied to each element. It may use `ColumnwiseValue` / `ColumnwiseThread` markers. |
| `columns` | optional | Column selection: a label, a position (`int`), a list of labels or positions, or a `slice` of positions. Defaults to all columns. |

Increment all the elements by 1:

```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
transform_tabular(df, lambda x: x + 1)
#    a  b
# 0  2  5
# 1  3  6
# 2  4  7
```

Transform only specific columns:

```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})
transform_tabular(df, lambda x: x * 2, ["a", "c"])
#    a   b    c
# 0  2  10  200
# 1  4  20  400
# 2  6  30  600
```

The function can be any callable, not only a `lambda`:

```python
import math

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
transform_tabular(df, math.sqrt, "b")
#    a         b
# 0  1  2.000000
# 1  2  2.236068
# 2  3  2.449490
```

### Columnwise values

`ColumnwiseValue(func)` wraps a function of a whole column, which here receives the column as a list. The function is evaluated once per column, and every element of that column sees the same result.

Subtract from each element the mean of its column:

```python
import statistics

df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
transform_tabular(df, lambda x: x - ColumnwiseValue(statistics.mean))
#    x   y
# 0 -2 -20
# 1 -1 -10
# 2  0   0
# 3  1  10
# 4  2  20
```

### Columnwise threads

`ColumnwiseThread(func)` wraps a function from a whole column to a column of the same length. The function is evaluated once per column, and each element is replaced by the element in the same position of the result.

Compute the cumulative sum of each column:

```python
from itertools import accumulate

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
transform_tabular(df, lambda x: ColumnwiseThread(lambda col: list(accumulate(col))))
#    a   b
# 0  1   4
# 1  3   9
# 2  6  15
```

Markers can also be defined beforehand, and combined. Compute the cumulative sum of each column and then subtract its mean:

```python
running = ColumnwiseThread(lambda col: list(accumulate(col)))
mean = ColumnwiseValue(statistics.mean)
transform_tabular(df, lambda x: running - mean)
#    a   b
# 0 -1  -1
# 1  1   4
# 2  4  10
```

### Operator form

`transform_tabular` can be given the function alone, to produce a reusable transformer:

```python
double_all = transform_tabular(lambda x: x * 2)
double_all(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
#    a  b
# 0  2  6
# 1  4  8
```


## `transform_columns`

```python
transform_columns(df, {column: func, ...})     # modify or add columns
transform_columns({column: func, ...})         # operator form: returns a reusable transformer
```

Each function receives a row and gives the cell of its column in that row. The row is a read-only mapping from column labels to values, which can also be read as attributes: `r["a"]` or `r.a`.

Transform two columns into a new one:

```python
tab = pd.DataFrame({"a": [2, 1, 3], "b": [2.78, 3.14, 1.68]})
transform_columns(tab, {"c": lambda r: r["b"] - r["a"]})
#    a     b     c
# 0  2  2.78  0.78
# 1  1  3.14  2.14
# 2  3  1.68 -1.32
```

An existing column is replaced where it is, and a new one is added after the existing ones. All the functions see the original table: here `c` is computed from the original `a`, not from the transformed one.

```python
transform_columns(tab, {"a": lambda r: 10 * r.a, "c": lambda r: r.a + 1})
#     a     b  c
# 0  20  2.78  3
# 1  10  3.14  2
# 2  30  1.68  4
```

`None` removes a column:

```python
transform_columns(tab, {"ratio": lambda r: r.b / r.a, "b": None})
#    a  ratio
# 0  2   1.39
# 1  1   3.14
# 2  3   0.56
```

### Placement of new columns

To place a new column elsewhere than at the end, give a list in which the dictionaries are mixed with labels of existing columns. A new column goes right after the column named before it:

```python
transform_columns(tab, ["a", {"c": lambda r: r.b - r.a}])
#    a     c     b
# 0  2  0.78  2.78
# 1  1  2.14  3.14
# 2  3 -1.32  1.68
```

and it goes first if it comes before any column name:

```python
transform_columns(tab, [{"c": lambda r: r.b - r.a}, "a"])
#       c  a     b
# 0  0.78  2  2.78
# 1  2.14  1  3.14
# 2 -1.32  3  1.68
```

### Columnwise values and threads

In `transform_columns` the function of a marker receives the whole table, that is the DataFrame, so that it can use any of its columns:

```python
mean_b = ColumnwiseValue(lambda t: t["b"].mean())
transform_columns(tab, {"b-mean": lambda r: r.b - mean_b})
#    a     b    b-mean
# 0  2  2.78  0.246667
# 1  1  3.14  0.606667
# 2  3  1.68 -0.853333
```

A column that does not depend on the single row can be given as the marker itself. No function is then called row by row, which makes this the fastest way to transform a large table:

```python
transform_columns(tab, {"total": ColumnwiseThread(lambda t: t["b"].cumsum())})
#    a     b  total
# 0  2  2.78   2.78
# 1  1  3.14   5.92
# 2  3  1.68   7.60
```

### Operator form

Given the transformations alone, `transform_columns` returns a transformer, which is convenient in a `pipe`:

```python
(tab
 .pipe(transform_columns({"c": lambda r: r.b - r.a}))
 .pipe(transform_columns({"d": lambda r: 2 * r.c})))
#    a     b     c     d
# 0  2  2.78  0.78  1.56
# 1  1  3.14  2.14  4.28
# 2  3  1.68 -1.32 -2.64
```


## `transform_rows`

```python
transform_rows(df, rows, transformation)                     # transform only some rows
transform_rows(df, rows, transformation, keep_order=False)   # transformed rows first
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame` | Input DataFrame |
| `rows` | row selection | A position (`int`), a list of positions, or a function of the row that is true for the rows to transform. Also a `slice` of positions, or a list or `Series` of booleans. |
| `transformation` | dict or callable | Column transformations as in `transform_columns`, or one function applied to every cell of the rows as in `transform_tabular`. |
| `keep_order` | `bool` | Whether to keep the original order of the rows (default `True`). |

Rows are given by position, starting from 0, whatever the index of the table.

Transform a single cell:

```python
tab = pd.DataFrame({"a": [2, 10, 7], "b": [8, 0, 0], "c": [5, 4, 7]})
transform_rows(tab, 2, {"c": lambda r: 2 * r.c})
#     a  b   c
# 0   2  8   5
# 1  10  0   4
# 2   7  0  14
```

Transform the first two rows of the first column:

```python
transform_rows(tab, [0, 1], {"a": lambda r: r.a + r.b})
#     a  b  c
# 0  10  8  5
# 1  10  0  4
# 2   7  0  7
```

Transform all the cells of the first and third rows with one function:

```python
tab = pd.DataFrame({"a": [9, 10, 7], "b": [1, 8, 0], "c": [6, 5, 5]})
transform_rows(tab, [0, 2], lambda x: x ** 3)
#      a  b    c
# 0  729  1  216
# 1   10  8    5
# 2  343  0  125
```

Select the rows by a condition on their values:

```python
transform_rows(tab, lambda r: r.c % 2 == 0, {"a": lambda r: 2 * r.a + 2 * r.b})
#     a  b  c
# 0  20  1  6
# 1  10  8  5
# 2   7  0  5
```

`ColumnwiseValue` and `ColumnwiseThread` are computed on the whole columns, including the rows that are not transformed. Here the median and the cumulative sum of `b` are those of all three rows:

```python
median_b = ColumnwiseValue(lambda t: t["b"].median())
running_b = ColumnwiseThread(lambda t: t["b"].cumsum())
transform_rows(tab, [1, 2], {"a": lambda r: 2 * r.a - median_b,
                             "b": lambda r: r.b + running_b})
#       a   b  c
# 0   9.0   1  6
# 1  19.0  17  5
# 2  13.0   9  5
```

The functions are called on the selected rows only, so they need not make sense for the other ones.

### Order of the rows

With `keep_order=False` the transformed rows come first, followed by the others:

```python
tab = pd.DataFrame({"Company": ["A", "B", "C"],
                    "Expense": [8012, 20111, 992],
                    "Revenue": [5178, 40222, 7032]})
transform_rows(tab, lambda r: r.Company != "A",
               {"Expense": lambda r: r.Expense + 432}, keep_order=False)
#   Company  Expense  Revenue
# 1       B    20543    40222
# 2       C     1424     7032
# 0       A     8012     5178
```

### Modifying individual records

`transform_rows` is suited to correcting single records of a table of data. For example, missing values can be replaced with others coming from a different source:

```python
data = pd.DataFrame({
    "municipality": ["Comune di Casnigo", "Comune di Cassiglio",
                     "Comune di Castelli Calepio", "Comune di Cavernago"],
    "population": [3075, 111, 10398, 2720],
    "amount": [48422.68, None, None, 9910.17],
})
other_source = {"Castelli": 11920.21, "Cassiglio": 13080.55}

def from_other_source(r):
    return next(value for name, value in other_source.items() if name in r.municipality)

transform_rows(data, lambda r: pd.isna(r.amount), {"amount": from_other_source})
#                  municipality  population    amount
# 0           Comune di Casnigo        3075  48422.68
# 1         Comune di Cassiglio         111  13080.55
# 2  Comune di Castelli Calepio       10398  11920.21
# 3         Comune di Cavernago        2720   9910.17
```


## More on the markers

- The function of a marker receives the columnwise counterpart of what the transformation receives: where the transformation gets an element (`transform_tabular`, and `transform_rows` with a single function) it receives the column, as a list; where the transformation gets a row (`transform_columns`, and `transform_rows` with a dictionary) it receives the table, as the DataFrame.
- A marker can be created inside the transformation, as in `lambda x: x - ColumnwiseValue(min)`, or defined beforehand and used by name. In both cases its function is evaluated once.
- Markers follow their value in arithmetic and comparisons. Where the plain value is needed, for instance as the argument of another function, it is given by `marker.value`:

  ```python
  largest = ColumnwiseValue(max)
  transform_tabular(df, lambda x: math.gcd(x, largest.value))
  ```

- The result of a `ColumnwiseThread` function can be a list, a NumPy array or a `Series`, with one value per row. Values are matched to the rows by position.


## Correspondence with Wolfram Language

| Wolfram Language | Python |
|------------------|--------|
| `TransformTabular[tab, # + 1 &]` | `transform_tabular(df, lambda x: x + 1)` |
| `TransformTabular[tab, f, {"a", "c"}]`, `TransformTabular[tab, f, 2 ;;]` | `transform_tabular(df, f, ["a", "c"])`, `transform_tabular(df, f, slice(1, None))` |
| `TransformTabular[tab, # - ColumnwiseValue[Min[#]] &]` | `transform_tabular(df, lambda x: x - ColumnwiseValue(min))` |
| `TransformColumns[tab, "c" -> Function[#b - #a]]` | `transform_columns(df, {"c": lambda r: r.b - r.a})` |
| `TransformColumns[tab, {"c" -> f, "b" -> Nothing}]` | `transform_columns(df, {"c": f, "b": None})` |
| `TransformColumns[tab, {"a", "c" -> f}]` | `transform_columns(df, ["a", {"c": f}])` |
| `TransformColumns["c" -> f][tab]` | `transform_columns({"c": f})(df)` |
| `TransformRows[tab, {1, 3}, "c" -> Function[2 #c]]` | `transform_rows(df, [0, 2], {"c": lambda r: 2 * r.c})` |
| `TransformRows[tab, EvenQ[#c] &, #^3 &]` | `transform_rows(df, lambda r: r.c % 2 == 0, lambda x: x ** 3)` |
| `TransformRows[tab, rows, f, "KeepOrder" -> False]` | `transform_rows(df, rows, f, keep_order=False)` |

The results are the same as those of the original functions, with these differences:

- Positions of rows and columns start from 0, and negative positions count from the end.
- A function of Wolfram Language becomes any Python callable. In particular `transform_rows` accepts a plain function such as `math.sqrt` for all the cells of the rows.
- A name that is not a column raises an error: in the column selection of `transform_tabular`, as the anchor of a placement, or as a column to remove.
- A DataFrame always has column labels, so the form of `TransformColumns` for tables without column keys does not apply: the columns of a DataFrame made from a matrix are labelled `0`, `1`, … and used as such (`{2: lambda r: r[0] + r[1]}`).
- The transformations of `transform_rows` are evaluated on the selected rows only.


## Performance

The three functions return the same results as their Wolfram Language originals on identical tables (verified case by case). On arithmetic over whole columns the Wolfram kernel, which vectorizes the function, is much faster than a Python function called on every row; giving the transformation as a `ColumnwiseThread` of whole-column operations brings the Python side within a factor of 10–45. On `transform_rows` with the order of the rows kept, the Python port is about 2× faster than `TransformRows`. See [benchmark/BENCHMARK.md](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/transform-tabular/benchmark/BENCHMARK.md) for the full table and methodology.


## Changes

See [CHANGELOG.md](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/transform-tabular/CHANGELOG.md).


## See also

For further examples and details, see the documentation of the original Wolfram Language functions: [TransformTabular](https://resources.wolframcloud.com/FunctionRepository/resources/TransformTabular/) (resource function contributed by the same author) and [TransformColumns](https://reference.wolfram.com/language/ref/TransformColumns.html).

## Author

Daniele Gregori

## License

MIT
