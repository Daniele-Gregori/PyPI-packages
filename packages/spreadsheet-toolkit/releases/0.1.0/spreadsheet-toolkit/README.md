# spreadsheet-toolkit

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/spreadsheet-toolkit.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/spreadsheet-toolkit.yml)
[![PyPI version](https://badge.fury.io/py/spreadsheet-toolkit.svg)](https://badge.fury.io/py/spreadsheet-toolkit)
[![Python](https://img.shields.io/pypi/pyversions/spreadsheet-toolkit.svg)](https://pypi.org/project/spreadsheet-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Spreadsheet utilities for Python.

> **Status:** early development (v0.1.0).


## Installation

```bash
pip install spreadsheet-toolkit
```

### Dependencies

- [**openpyxl**](https://openpyxl.readthedocs.io/) (>=3.1.0) — for reading `.xlsx` files

## Usage

### `import_sheets`

Return the list of sheet names in a workbook.

```python
from spreadsheet_toolkit import import_sheets

import_sheets("budget.xlsx")
# ['Income', 'Expenses', 'Summary']
```

### `import_cells`

Import cell values from a spreadsheet. Optionally select a sheet (by name or 1-based index) and a row/column range.

```python
from spreadsheet_toolkit import import_cells

# All cells from the active sheet
import_cells("data.xlsx")
# [[10, 20, 30], ['hello', 'world', None], [3.14, True, None]]

# A specific sheet by name
import_cells("data.xlsx", sheet="Summary")
# [['total', 150]]

# A specific sheet by index
import_cells("data.xlsx", sheet=2)

# Rows 2-3, columns 1-2
import_cells("data.xlsx", rows=(2, 3), columns=(1, 2))
# [['hello', 'world'], [3.14, True]]
```

### `import_formulas`

Like `import_cells`, but formula cells return their formula string instead of the computed value.

```python
from spreadsheet_toolkit import import_formulas

import_formulas("data.xlsx")
# [[10, 20, '=A1+B1'], ['hello', 'world', None], [3.14, True, '=SUM(A1:B1)']]

import_formulas("data.xlsx", sheet="Summary")
# [['total', '=Data!A1+Data!B1']]
```

### `spreadsheet_index_to_position`

Convert a cell reference (e.g. `"A1"`, `"AA12"`) to a `(row, column)` tuple with 1-based indices.

```python
from spreadsheet_toolkit import spreadsheet_index_to_position

spreadsheet_index_to_position("A1")
# (1, 1)

spreadsheet_index_to_position("C5")
# (5, 3)

spreadsheet_index_to_position("AA1")
# (1, 27)
```

### `position_to_spreadsheet_index`

Convert a `(row, column)` position back to a cell reference string.

```python
from spreadsheet_toolkit import position_to_spreadsheet_index

position_to_spreadsheet_index(1, 1)
# 'A1'

position_to_spreadsheet_index(5, 3)
# 'C5'

position_to_spreadsheet_index(1, 27)
# 'AA1'
```

## See also

This Python package includes a translation of the following **Wolfram Language** functions:

- [**SpreadsheetIndexToPosition**](https://resources.wolframcloud.com/FunctionRepository/resources/SpreadsheetIndexToPosition) (resource function by S. Smit)
- [**PositionToSpreadsheetIndex**](https://resources.wolframcloud.com/FunctionRepository/resources/PositionToSpreadsheetIndex) (resource function by S. Smit)

## License

MIT
