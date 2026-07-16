# farey

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/farey.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/farey.yml)
[![PyPI version](https://badge.fury.io/py/farey.svg)](https://badge.fury.io/py/farey)
[![Python](https://img.shields.io/pypi/pyversions/farey.svg)](https://pypi.org/project/farey/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generate Farey sequences and Farey ranges.


## Overview

The [Farey sequence](https://en.wikipedia.org/wiki/Farey_sequence) F_n is the sequence of all completely reduced fractions between 0 and 1, with denominators ≤ n, in ascending order.

A **Farey range** `farey_range(x, y, n)` gives every rational number with denominator ≤ `n` inside the interval `[x, y]`. Equivalently, it is the (sorted) union of all the rational ranges whose reciprocal steps are the elements of the Farey sequence `F_n`.

The result is **exact**: a list of `Fraction` objects computed with the Farey next-term recurrence (no floating-point interval scaling). This is a faithful port of the improved Wolfram Language [`FareyRange`](https://resources.wolframcloud.com/FunctionRepository/resources/FareyRange/) used internally by the [`GeneralizedRange`](https://resources.wolframcloud.com/PacletRepository/resources/DanieleGregori/GeneralizedRange/) paclet, extended to negative and unit-fraction steps.

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

farey_range(2, 4, 3)   # bounds greater than 1
# [Fraction(2, 1), Fraction(7, 3), Fraction(5, 2), Fraction(8, 3), Fraction(3, 1),
#  Fraction(10, 3), Fraction(7, 2), Fraction(11, 3), Fraction(4, 1)]

farey_range(0, Fraction(3, 2), 2)   # exact, non-integer bound
# [Fraction(0, 1), Fraction(1, 2), Fraction(1, 1), Fraction(3, 2)]

# Descending: first bound larger, negative step
farey_range(1, 0, -3)
# [Fraction(1, 1), Fraction(2, 3), Fraction(1, 2), Fraction(1, 3), Fraction(0, 1)]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start` | *(required)* | First bound — the starting point of the range |
| `end` | *(required)* | Second bound |
| `step` | `1` | Controls the Farey order and direction (see below) |

### Step conventions

The magnitude of `step` selects the Farey **order**; the sign selects the
**direction**. `n` and `1/n` are equivalent (both mean order `n`):

| Step value | Order | Direction |
|------------|-------|-----------|
| `n` (positive int) | n | ascending |
| `1/n` (float or `Fraction(1, n)`) | n | ascending |
| `-n` (negative int) | n | descending |
| `-1/n` (float or `Fraction(-1, n)`) | n | descending |

A step that is zero, or a non-unit fraction such as `2/3`, raises `FareyError`.

## Performance

`farey_range` produces the same output as the Wolfram Language
`FareyRange` resource function (verified case-for-case). Because the work is
pure exact-rational arithmetic — with no parsing hot-path to exploit — the
compiled Wolfram kernel is about 3× faster on large ranges, while the Python
port wins on tiny ones; the trade-off is dependency-free portability. See
[benchmark/BENCHMARK.md](benchmark/BENCHMARK.md) for the full table and
methodology.

## See also

This Python package has been inspired by the following **Wolfram** functions:

- [**FareySequence**](https://reference.wolfram.com/language/ref/FareySequence.html) (built-in Wolfram Language function);
- [**FareyRange**](https://resources.wolframcloud.com/FunctionRepository/resources/FareyRange/) (resource function contributed by Jan Mangaldan).


## License

MIT
