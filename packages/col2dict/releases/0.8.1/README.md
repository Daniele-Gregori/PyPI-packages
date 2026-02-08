# col2dict

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/col2dict.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/col2dict.yml)
[![PyPI version](https://badge.fury.io/py/col2dict.svg)](https://badge.fury.io/py/col2dict)
[![Python versions](https://img.shields.io/pypi/pyversions/col2dict.svg)](https://pypi.org/project/col2dict/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Create nested dictionaries from columns of tabular data.

Python translation of the **Wolfram** [resource function "AssociateColumns"](https://resources.wolframcloud.com/FunctionRepository/resources/AssociateColumns/).

## Installation

```bash
pip install col2dict
```

## Overview

Data is well collected in DataFrames, but its manipulation and analysis often requires the use of nested dictionaries. `associate_columns` allows you to create (possibly nested) dictionaries from columns of a DataFrame or a list of dicts, with control over how duplicate keys are merged at each nesting level.

## Usage

```python
from col2dict import associate_columns
```

### Signatures

```python
associate_columns(tab, (col1, col2))
associate_columns(tab, (col1, col2), merge=func)
associate_columns(tab, (col1, col2, col3, ...))
associate_columns(tab, (col1, col2, col3, ...), merge=func)
associate_columns(tab, (col1, col2, col3, ...), merge=[f12, f23, ...])
associate_columns(tab, ([colA, colB], col2))          # multi-column keys
associate_columns(tab, (col1, [colA, colB]))           # multi-column values
associate_columns(tab, ([colA, colB], [colC, colD]))   # both
```

### Parameters

| Parameter | Description |
|---|---|
| `tab` | A `pandas.DataFrame` or a list of dicts. |
| `cols` | A tuple/list of column specs. Each element is a column name (`str`) or a list of column names. With 2 elements: first → keys, second → values. With 3+ elements: creates nested dicts. |
| `merge` | Merging function(s) for duplicate keys. `None` (default): auto-merge to list with a warning. A single callable: applied at every level. A list of callables: one per nesting transition. |

### Options

| Option | Default | Description |
|---|---|---|
| `duplicates_warning` | `True` | Warns when duplicate keys are found and no merge function is provided. Set to `False` to silence the warning and collect values into lists silently. |

## Examples

### Basic: two columns

```python
import pandas as pd
from col2dict import associate_columns

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Carol"],
    "Age": [30, 25, 35],
})

associate_columns(df, ("Name", "Age"))
# {'Alice': 30, 'Bob': 25, 'Carol': 35}
```

### Duplicate keys with merging

```python
df = pd.DataFrame({
    "Year": [2020, 2020, 2021, 2021, 2022],
    "Name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "Score": [90, 85, 92, 88, 95],
})

associate_columns(df, ("Year", "Name"), merge=sorted)
# {2020: ['Alice', 'Bob'], 2021: ['Carol', 'Dave'], 2022: ['Eve']}

associate_columns(df, ("Year", "Score"), merge=sum)
# {2020: 175, 2021: 180, 2022: 95}
```

### Nested association (3+ columns)

```python
associate_columns(df, ("Year", "Name", "Score"))
# {2020: {'Alice': 90, 'Bob': 85},
#  2021: {'Carol': 92, 'Dave': 88},
#  2022: {'Eve': 95}}
```

### Multi-column keys

```python
df = pd.DataFrame({
    "Store": ["A", "A", "B"],
    "Dept": ["Elec", "Food", "Elec"],
    "Revenue": [500, 100, 450],
})

associate_columns(df, (["Store", "Dept"], "Revenue"))
# {('A', 'Elec'): 500, ('A', 'Food'): 100, ('B', 'Elec'): 450}
```

### Multi-column values

```python
df = pd.DataFrame({
    "Name": ["Alice", "Bob"],
    "Age": [30, 25],
    "City": ["NYC", "LA"],
})

associate_columns(df, ("Name", ["Age", "City"]))
# {'Alice': [30, 'NYC'], 'Bob': [25, 'LA']}
```

### Per-level merge functions

```python
df = pd.DataFrame({
    "A": [1, 1, 1, 2],
    "B": ["x", "x", "y", "z"],
    "C": [100, 200, 300, 400],
})

# sorted at level 1 (sort inner dict keys), sum at level 2 (merge dup values)
associate_columns(df, ("A", "B", "C"), merge=[sorted, sum])
# {1: {'x': 300, 'y': 300}, 2: {'z': 400}}
```

### Deep nesting with mixed multi-column specs

```python
df = pd.DataFrame({
    "Store": ["A", "A", "B", "B"],
    "Dept": ["Elec", "Food", "Elec", "Elec"],
    "Item": ["TV", "Milk", "TV", "TV"],
    "Brand": ["Sony", "Org", "LG", "Sony"],
    "Price": [500, 3, 450, 500],
})

# Store -> {Dept, Item} -> Brand -> Price
associate_columns(df, ("Store", ["Dept", "Item"], "Brand", "Price"))
# {'A': {('Elec', 'TV'): {'Sony': 500}, ('Food', 'Milk'): {'Org': 3}},
#  'B': {('Elec', 'TV'): {'LG': 450, 'Sony': 500}}}
```

For full documentation and more examples, see the [original Wolfram Language resource page](https://resources.wolframcloud.com/FunctionRepository/resources/AssociateColumns/), contributed by the same author and vetted by the Wolfram Review Team.

## Testing

The test suite lives in `tests/test_associate_columns.py` and uses [pytest](https://docs.pytest.org/). To run it:

```bash
pip install col2dict[dev]
pytest
```

Or from a source checkout:

```bash
git clone https://github.com/Daniele-Gregori/PyPI-packages.git
cd PyPI-packages/packages/col2dict
pip install -e ".[dev]"
pytest
```

The suite includes 68 tests organised in 14 groups covering basic associations, duplicate-key merging, nested dicts up to 5 levels deep, multi-column keys and values, per-level merge functions, input-type compatibility, edge cases, and error handling.

## License

MIT — see [LICENSE](LICENSE).
