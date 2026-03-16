# farey

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/farey.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/farey.yml)
[![PyPI version](https://badge.fury.io/py/farey.svg)](https://badge.fury.io/py/farey)
[![Python](https://img.shields.io/pypi/pyversions/farey.svg)](https://pypi.org/project/farey/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generate Farey sequences and Farey ranges.


## Overview

The [Farey sequence](https://en.wikipedia.org/wiki/Farey_sequence) F_n is the sequence of all completely reduced fractions between 0 and 1, with denominators ≤ n, in ascending order.

A **Farey range** applies a Farey sequence to subdivide an arbitrary interval into rational points. In other words, it is the (sorted) union of all the rational ranges with (the reciprocal of their) steps equal to the elements of the Farey sequence. 

## `farey_sequence`

### Usage

```python
from farey import farey_sequence

farey_sequence(1)
# [Fraction(0, 1), Fraction(1, 1)]

farey_sequence(5)
# [Fraction(0, 1), Fraction(1, 5), Fraction(1, 4), Fraction(1, 3),
#  Fraction(2, 5), Fraction(1, 2), Fraction(3, 5), Fraction(2, 3),
#  Fraction(3, 4), Fraction(4, 5), Fraction(1, 1)]
```

## `farey_range`

### Usage

```python
from farey import farey_range
from fractions import Fraction

farey_range(0, 1, 3)
# [Fraction(0, 1), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1, 1)]

farey_range(0, 2, 4)
# [Fraction(0, 1), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
#  Fraction(2, 3), Fraction(3, 4), Fraction(1, 1), Fraction(5, 4),
#  Fraction(4, 3), Fraction(3, 2), Fraction(5, 3), Fraction(7, 4),
#  Fraction(2, 1)]

# Reversed
farey_range(0, 1, -5)
# [Fraction(1, 1), Fraction(4, 5), ..., Fraction(0, 1)]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start` | *(required)* | Left endpoint of the interval |
| `end` | *(required)* | Right endpoint of the interval |
| `step` | `None` | Controls the Farey order and direction (see below) |

### Step conventions

The `step` parameter controls which Farey order is used and whether the result is ascending or descending:

| Step value | Order | Direction |
|------------|-------|-----------|
| `n` (positive int) | n | ascending |
| `-n` (negative int) | n | descending |
| `1/n` (float or `Fraction(1, n)`) | n | ascending |
| `-1/n` (float or `Fraction(-1, n)`) | n | descending |
| `None` | 1 | ascending |


All equivalent forms produce the same result:

```python
farey_range(0, 2, 3)
farey_range(0, 2, Fraction(1, 3))
farey_range(0, 2, 1/3)
# all return the same list
```

## See also

This Python package has been inspired by the following **Wolfram** functions:

- [**FareySequence**](https://reference.wolfram.com/language/ref/FareySequence.html) (built-in Wolfram Language function);
- [**FareyRange**](https://resources.wolframcloud.com/FunctionRepository/resources/FareyRange/) (resource function contributed by Jan Mangaldan).


## License

MIT
