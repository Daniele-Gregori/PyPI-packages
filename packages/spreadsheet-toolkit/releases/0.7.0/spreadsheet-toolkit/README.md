# spreadsheet-toolkit

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/spreadsheet-toolkit.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/spreadsheet-toolkit.yml)
[![PyPI version](https://badge.fury.io/py/spreadsheet-toolkit.svg)](https://badge.fury.io/py/spreadsheet-toolkit)
[![Python](https://img.shields.io/pypi/pyversions/spreadsheet-toolkit.svg)](https://pypi.org/project/spreadsheet-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tools and utilities for spreadsheet analysis: formula **dependency tracing**, workbook import, and cell reference conversion.

## Installation

```bash
pip install spreadsheet-toolkit
```

### Dependencies

- [**openpyxl**](https://openpyxl.readthedocs.io/) (>=3.1.0) — for reading `.xlsx` files

## Usage

### `spreadsheet_trace`

Trace all the formula dependencies of a spreadsheet cell, recursively, down to the elementary value cells. The output closely mirrors the Wolfram Language [`Trace`](https://reference.wolfram.com/language/ref/Trace.html): each traced cell produces `[cell, formula, subtraces...]` and each leaf cell produces `[cell, value]`.

```python
from spreadsheet_toolkit import spreadsheet_trace

spreadsheet_trace("book.xlsx", "D1")
# ['D1', 'C1*2', ['C1', 'A1+B1', ['A1', 10], ['B1', 20]]]
```

Cell ranges are expanded, and absolute references (`$B$10`) are followed like plain ones:

```python
spreadsheet_trace("sales.xlsx", "F2")
# ['F2', 'E2*$B$10', ['E2', 'C2*D2', ['C2', 25.5], ['D2', 100]], ['B10', 0.22]]

spreadsheet_trace("data.xlsx", "B1")
# ['B1', 'SUM(A1:A3)', ['A1', 15], ['A2', 22], ['A3', 8]]
```

Cross-sheet references and column ranges (e.g. `Products!A:C`, expanded across all rows of the referenced sheet) are supported:

```python
spreadsheet_trace("report.xlsx", "Summary!B3")
# ['Summary!B3', 'Input!B14', ['Input!B14', 'SUM(B2:B13)', ['B2', 12000.0], ...]]

spreadsheet_trace("orders.xlsx", "Orders!D2")
# ['Orders!D2', 'VLOOKUP(B2,Products!A:C,3,FALSE)', ['B2', 101],
#  ['Products!A1', 'ProductID'], ['Products!A2', 101], ...]
```

By default duplicate dependency branches are kept, mirroring the repeated occurrences of a cell in the formulas; pass `trace_duplicates=False` to trace each referenced cell only once per formula:

```python
spreadsheet_trace("data.xlsx", "C5", trace_duplicates=False)
```

Instead of a file path, a `(sheets, data, formulas)` triple as returned by `import_all` can be passed directly:

```python
from spreadsheet_toolkit import import_all, spreadsheet_trace

book = import_all("book.xlsx")
spreadsheet_trace(book, "D1")
```

### `import_all`

Import sheet names, cell values and formulas from a workbook, in one call. Repeated imports of the same (unmodified) file return a cached result.

```python
from spreadsheet_toolkit import import_all

sheets, data, formulas = import_all("book.xlsx")

sheets
# ['Data', 'Summary']

data[0]       # 2D list of cell values of the first sheet ("" for empty cells)
# [[10, 20, ''], ['hello', 'world', '']]

formulas[0]   # same shape: formula strings without "=", "" for non-formula cells
# [['', '', 'A1+B1'], ['', '', '']]
```

### `index_to_position`

Convert a cell reference (plain or absolute) to a `(row, column)` tuple with 1-based indices.

```python
from spreadsheet_toolkit import index_to_position

index_to_position("A1")
# (1, 1)

index_to_position("C5")
# (5, 3)

index_to_position("AA1")
# (1, 27)

index_to_position("$B$10")
# (10, 2)
```

### `position_to_index`

Convert a `(row, column)` position back to a cell reference string.

```python
from spreadsheet_toolkit import position_to_index

position_to_index((1, 1))
# 'A1'

position_to_index((5, 3))
# 'C5'

position_to_index((1, 27))
# 'AA1'
```

The 0.1.x function names also remain available; see the [changelog](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/spreadsheet-toolkit/CHANGELOG.md).

## Performance

The Python `spreadsheet_trace` produces identical traces to the Wolfram Language original while running roughly 40–130× faster; see the [benchmark](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/spreadsheet-toolkit/benchmark/BENCHMARK.md).

## See also

This Python package is a translation of the following **Wolfram Language** functions:

- [**SpreadsheetTrace**](https://resources.wolframcloud.com/FunctionRepository/resources/SpreadsheetTrace) (resource function by Daniele Gregori)
- the `SpreadsheetToolkit` package by Daniele Gregori, in turn based on:
  - [**SpreadsheetIndexToPosition**](https://resources.wolframcloud.com/FunctionRepository/resources/SpreadsheetIndexToPosition) (resource function by Sjoerd Smit)
  - [**PositionToSpreadsheetIndex**](https://resources.wolframcloud.com/FunctionRepository/resources/PositionToSpreadsheetIndex) (resource function by Sjoerd Smit)

## License

MIT
