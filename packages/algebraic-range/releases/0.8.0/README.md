# algebraic-range


[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/algebraic-range.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/algebraic-range.yml)
[![PyPI version](https://img.shields.io/pypi/v/algebraic-range.svg)](https://pypi.org/project/algebraic-range/)
[![Python](https://img.shields.io/pypi/pyversions/algebraic-range.svg)](https://pypi.org/project/algebraic-range/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generate ranges of algebraic numbers.

Python port of the **Wolfram** [resource function "AlgebraicRange"](https://resources.wolframcloud.com/FunctionRepository/resources/AlgebraicRange/) resource function.

Requires [SymPy](https://www.sympy.org/) ≥ 1.12.

## Overview

`algebraic_range` creates ranges made of [algebraic numbers](https://en.wikipedia.org/wiki/Algebraic_number). This extends the basic concept of `range()` to include, besides rational numbers, also roots — always restricted to the real domain.

The first two arguments represent the bounds of the range (minimum and maximum values), while the optional third and fourth arguments (by default equal to 1 and 0) regulate the upper and lower bounds of the steps (differences between successive elements).



## Usage

```python
algebraic_range(x)                # Sqrt[Range[1, x²]]  for x ≥ 1
algebraic_range(x, y)             # Sqrt[Range[x², y²]]  for 0 ≤ x ≤ y
algebraic_range(x, y, s)          # step upper bound s,  0 < s ≤ y
algebraic_range(x, y, s, d)       # step lower bound d,  0 ≤ d ≤ s
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r1` | *(required)* | Start of range (or single argument) |
| `r2` | `None` | End of range |
| `s` | `None` | Step upper bound (negative → descending) |
| `d` | `0` | Step lower bound |
| `root_order` | `2` | `int r` → orders 2..r; `[r]` → only order r; `[r1,r2,…]` → listed orders |
| `step_method` | `"Outer"` | `"Outer"` or `"Root"` |
| `farey_range` | `False` | Use Farey-sequence–based rational multipliers |
| `formula_complexity_threshold` | `inf` | Discard elements above this complexity |
| `algebraics_only` | `True` | Reject transcendental inputs |

### Options

#### `root_order`

```python
# Include square and cubic roots
algebraic_range(2, root_order=3)

# Only cubic roots
algebraic_range(2, root_order=[3])

# Cubic and fifth roots
algebraic_range(1, Rational(3, 2), root_order=[3, 5])
```

#### `step_method`

```python
# "Root" method: Sqrt[Range[x², y², s²]]
algebraic_range(0, 3, Rational(1, 3), step_method="Root")
```

The default `"Outer"` method uses the outer product construction. The `"Root"` method is generally a superset.

#### `farey_range`

```python
algebraic_range(0, 3, Rational(1, 3), farey_range=True)
```

Generalises `FareyRange` by combining algebraic ranges over all Farey-sequence steps.

#### `formula_complexity_threshold`

```python
# Only keep simple expressions
algebraic_range(4, root_order=4, formula_complexity_threshold=8)
```

#### `algebraics_only`

```python
from sympy import sqrt, E

# This raises NotAlgebraicError:
# algebraic_range(0, 5, sqrt(E))

# Allow transcendental step:
algebraic_range(0, 5, sqrt(E), algebraics_only=False)
```

## Properties

- **Extends `range()`**: `set(range(x, y+1))` ⊆ `set(algebraic_range(x, y))`
- **Negative reflection**: `algebraic_range(-y, -x)` = `list(reversed([-v for v in algebraic_range(x, y)]))`
- **All outputs are algebraic** (when `algebraics_only=True`) and real
- **Sorted** in ascending order (or descending for negative step)
- **No duplicates**


## See also

More details and examples can be found in the documentation for the original Wolfram Language resource function [`AlgebraicRange`](https://resources.wolframcloud.com/FunctionRepository/resources/AlgebraicRange/), contributed by the same author and vetted by the Wolfram Review Team. 

