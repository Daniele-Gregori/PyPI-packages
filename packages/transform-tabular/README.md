# transform-tabular

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/tests.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/transform-tabular)](https://pypi.org/project/transform-tabular/)
[![Python versions](https://img.shields.io/pypi/pyversions/transform-tabular)](https://pypi.org/project/transform-tabular/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)



Apply a single function element-wise across an entire DataFrame, or a selection of its columns.

Two special markers extend this to column-aware operations:

- **`ColumnwiseValue(func)`** — computes a scalar aggregate per column (e.g. mean, max). The scalar is then used in the element-wise expression, so every row in that column sees the same aggregate.
- **`ColumnwiseThread(func)`** — applies a list-to-list transformation per column (e.g. sort, cumulative sum). Each row receives the corresponding element from the transformed list.

This package is a Python port of the Wolfram Language resource function [`TransformTabular`](https://resources.wolframcloud.com/FunctionRepository/resources/TransformTabular/).


## Usage

```python
from transform_tabular import transform_tabular, ColumnwiseValue, ColumnwiseThread
import pandas as pd
```

### Syntax

```python
transform_tabular(df, func)                # apply func element-wise to all columns
transform_tabular(df, func, columns)       # apply func only to selected columns
transform_tabular(func)                    # operator form: returns a reusable transformer
transform_tabular(func, columns)           # operator form with column selection
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame` | Input DataFrame |
| `func` | callable | Function applied element-wise to each cell. May reference `ColumnwiseValue` / `ColumnwiseThread` markers. |
| `columns` | optional | Column selection: a name (`str`), index (`int`), list of names/indices, or `slice`. Defaults to all columns. |

The function `func` is applied element-wise. The optional third argument can be either a list of columns, a list of column indices, a single column, or a slice.

### Basic transformation

Increment all numeric columns by 1:

```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
transform_tabular(df, lambda x: x + 1)
#    a  b
# 0  2  5
# 1  3  6
# 2  4  7
```

### Column selection

Transform only specific columns:

```python
df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})
transform_tabular(df, lambda x: x * 2, ["a", "c"])
#    a   b    c
# 0  2  10  200
# 1  4  20  400
# 2  6  30  600
```

### ColumnwiseValue (column-level aggregation)

`ColumnwiseValue(func)` wraps a function `func(column_as_list) -> scalar`. The scalar is pre-computed per column and then participates in the element-wise arithmetic — every row in a given column sees that column's aggregate.

Subtract the mean from each element (centering):

```python
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
cv_mean = ColumnwiseValue(lambda col: sum(col) / len(col))
transform_tabular(df, lambda x: x - cv_mean)
#      x     y
# 0 -2.0 -20.0
# 1 -1.0 -10.0
# 2  0.0   0.0
# 3  1.0  10.0
# 4  2.0  20.0
```

### ColumnwiseThread (column-level transformation)

`ColumnwiseThread(func)` wraps a function `func(column_as_list) -> list_of_same_length`. The transformation is pre-computed per column and each row receives its corresponding element from the resulting list.

Sort each column independently:

```python
df = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
ct_sorted = ColumnwiseThread(lambda col: sorted(col))
transform_tabular(df, lambda x: ct_sorted)
#    a  b
# 0  1  4
# 1  2  5
# 2  3  6
```

### Combined ColumnwiseValue and ColumnwiseThread

Both markers can be used together. For example, sort each column and then add its mean:

```python
df = pd.DataFrame({"a": [3, 1, 2], "b": [60, 40, 50]})
ct_sorted = ColumnwiseThread(lambda col: sorted(col))
cv_mean = ColumnwiseValue(lambda col: sum(col) / len(col))
transform_tabular(df, lambda x: ct_sorted + cv_mean)
#      a     b
# 0  3.0  90.0
# 1  4.0 100.0
# 2  5.0 110.0
```

### Operator form

`transform_tabular` can be curried to produce a reusable transformer:

```python
double_all = transform_tabular(lambda x: x * 2)
double_all(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
#    a  b
# 0  2  6
# 1  4  8
```

## See also

For further details and examples, see the documentation for the original Wolfram Language resource function: [TransformTabular](https://resources.wolframcloud.com/FunctionRepository/resources/TransformTabular/).

## Author

Daniele Gregori

## License

MIT
