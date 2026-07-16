# algebraic-range

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/algebraic-range.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/algebraic-range.yml)
[![PyPI version](https://badge.fury.io/py/algebraic-range.svg)](https://badge.fury.io/py/algebraic-range)
[![Python](https://img.shields.io/pypi/pyversions/algebraic-range.svg)](https://pypi.org/project/algebraic-range/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Generate ranges of algebraic numbers.**

Python port of the **Wolfram Language** resource function
[`AlgebraicRange`](https://resources.wolframcloud.com/FunctionRepository/resources/AlgebraicRange/)
**2.0**, by the same author. Requires [SymPy](https://www.sympy.org/) ≥ 1.12.

```
pip install algebraic-range
```

## Overview

`algebraic_range` creates ranges made of
[algebraic numbers](https://en.wikipedia.org/wiki/Algebraic_number):
it extends the basic concept of `range()` to include, besides rational
numbers, also roots — always restricted to the real domain.

The first two arguments are the bounds of the range, while the optional
third and fourth arguments (by default 1 and 0) set the **upper and lower
bounds of the steps**, the differences between successive elements:

```python
algebraic_range(x)             # Sqrt[Range[1, x²]],   for x ≥ 1
algebraic_range(x, y)          # Sqrt[Range[x², y²]],  for 0 ≤ x ≤ y
algebraic_range(x, y, s)       # steps bounded above by s,  0 < s ≤ y − x
algebraic_range(x, y, s, d)    # steps bounded below by d,  0 ≤ d ≤ |s|
```

If no step is given, the elementary range is the square-root grid anchored
at the bounds — `Sqrt[Range[x², y²]]` — since among irrationals no constant
step exists. Negative bounds are handled by reflection (real roots are
defined for positive arguments), and a negative `s` produces a descending
range:

```python
>>> from algebraic_range import algebraic_range
>>> algebraic_range(3)
[1, sqrt(2), sqrt(3), 2, sqrt(5), sqrt(6), sqrt(7), 2*sqrt(2), 3]

>>> algebraic_range(-3, -1)
[-3, -2*sqrt(2), -sqrt(7), -sqrt(6), -sqrt(5), -2, -sqrt(3), -sqrt(2), -1]
```

With a step upper bound the range is filled by rational multiples of the
elementary roots — conceptually the outer product of
`algebraic_range(min(x, x/s), y)` with the step-`s` rational grid,
restricted to `[x, y]`:

```python
>>> from sympy import Rational
>>> algebraic_range(0, 2, Rational(1, 2))
[0, 1/2, sqrt(2)/2, sqrt(3)/2, 1, sqrt(2), 3/2, sqrt(3), 2]
```

Taken literally, that outer product scales quadratically; like the Wolfram
2.0 implementation, this port instead selects for each root only the
admissible multipliers, by binary search over the sorted factor grid, and
effectively scales linearly (see
[benchmark/BENCHMARK.md](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/algebraic-range/benchmark/BENCHMARK.md)).

The fourth argument imposes a **minimum** absolute difference between
successive elements, taming the accumulation of algebraics towards certain
points and producing nearly uniform distributions:

```python
>>> algebraic_range(2, 5, Rational(1, 3), Rational(1, 4))
[2, 4*sqrt(3)/3, 2*sqrt(15)/3, 2*sqrt(19)/3, sqrt(10), 2*sqrt(3),
 5*sqrt(5)/3, 4, sqrt(19), 8*sqrt(3)/3, 2*sqrt(6)]
```

## Options

| Option | Default | Description |
|---|---|---|
| `root_order` | `2` | the root orders to be included |
| `step_method` | `"Outer"` | how the step bound is interpreted |
| `farey_range` | `False` | steps given by the Farey sequence |
| `formula_complexity` | `inf` | discard elements above this complexity |
| `algebraics_only` | `True` | accept only algebraic parameters |
| `working_precision` | machine | precision of internal numerical decisions |

### `root_order`

An integer `r` includes all roots up to order *r*; `[r]` only order *r*;
`[r1, r2, …]` all listed orders.

```python
algebraic_range(2, root_order=3)               # square and cubic roots
algebraic_range(2, root_order=[3])             # cube roots only:
# [1, 2**(1/3), 3**(1/3), 2**(2/3), 5**(1/3), 6**(1/3), 7**(1/3), 2]
algebraic_range(1, Rational(3, 2), root_order=[3, 5])
```

### `step_method`

The default `"Outer"` uses the rational-multiplier construction above;
`"Root"` steps directly in the power domain,
`(Range[x^n, y^n, s^n])^(1/n)` — generally a superset:

```python
>>> algebraic_range(0, 2, Rational(2, 3), step_method="Root")
[0, 2/3, 2*sqrt(2)/3, 2*sqrt(3)/3, 4/3, 2*sqrt(5)/3, 2*sqrt(6)/3,
 2*sqrt(7)/3, 4*sqrt(2)/3, 2]
```

### `farey_range`

Generalizes the resource function
[`FareyRange`](https://resources.wolframcloud.com/FunctionRepository/resources/FareyRange/):
the range combines the algebraic ranges of **all** steps in the Farey
sequence of the given order (an integer step `s` means order `s`; a step
`1/n` means order `n`):

```python
>>> algebraic_range(0, 3, Rational(1, 3), farey_range=True)  # F₃ steps: 1/3, 1/2, 2/3, 1
```

equals the union of the plain ranges with steps `1/3`, `1/2`, `2/3` and `1`,
and contains `FareyRange[0, 3, 3]` as its rational backbone.

### `formula_complexity`

An alternative way to thin a range: discard elements whose symbolic form
exceeds a heuristic complexity threshold.

```python
algebraic_range(4, root_order=4, formula_complexity=8)
```

### `algebraics_only`

By default, transcendental parameters are rejected (`NotAlgebraicError`);
disable to deliberately allow them:

```python
from sympy import sqrt, E
algebraic_range(0, 5, sqrt(E), algebraics_only=False)
```

### `working_precision`

With steps below machine resolution, nearby algebraics may collide
numerically and be omitted; raising the precision keeps them distinct —
it can be pushed arbitrarily high:

```python
algebraic_range(1 - Rational(1, 10**13), 1 + Rational(1, 10**13),
                Rational(1, 10**17), working_precision=30)
```

## Behaviour notes

- Approximate inputs are exactified before use (the WL `RootApproximant`
  step): `algebraic_range(0.1, 3.1)` starts at `1/10`.
- Results are plain lists of exact SymPy expressions, sorted ascending
  (descending for negative `s`), without duplicates.
- Errors mirror the WL failure modes: `NotRealError`, `NotAlgebraicError`,
  `FareyStepError`, `LowerBoundError` (negative `d`), `StepBoundError`
  (`d > |s|`).

## Performance

The five reference cases of the Wolfram verification suite run in about
1–7 s each (40 000–80 000 elements) — up to ~1.7× **faster** than the
Wolfram Language 2.0 implementation on the stepped cases, and ~40×–100×
faster than algebraic-range 0.8. Details and methodology:
[benchmark/BENCHMARK.md](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/algebraic-range/benchmark/BENCHMARK.md).

## Applications

A natural application is the search for closed forms of floating-point
numbers, providing structured search ranges of algebraic candidates.
The sibling package
[`find-closed-form`](https://pypi.org/project/find-closed-form/)
(port of the Wolfram resource function
[`FindClosedForm`](https://resources.wolframcloud.com/FunctionRepository/resources/FindClosedForm/),
by the same author) accepts these ranges since version 0.5.0 through its
`search_range` option, searching functions of exact algebraic arguments —
here recovering a formula from its bare machine digits:

```python
from find_closed_form import find_closed_form

find_closed_form(4.1132503787829275, search_range="Algebraic")
# exp(sqrt(2))
```

Generator options reach `algebraic_range` through `search_range_options`
(e.g. `{"root_order": 3}` to search cube-root arguments), and an explicit
`algebraic_range(...)` output can be passed as fixed `search_arguments`.

## See also

- Full documentation of the original
  [`AlgebraicRange`](https://resources.wolframcloud.com/FunctionRepository/resources/AlgebraicRange/)
  (contributed by the same author and vetted by the Wolfram Review Team),
  whose 2.0 verification suite this package transcribes in
  [tests/](https://github.com/Daniele-Gregori/PyPI-packages/tree/main/packages/algebraic-range/tests).
- The paclet
  [`GeneralizedRange`](https://resources.wolframcloud.com/PacletRepository/resources/DanieleGregori/GeneralizedRange/)
  for extended functionality and syntax.
- The [`farey`](https://pypi.org/project/farey/) package, by the same
  author, for standalone Farey sequences and ranges.
- [CHANGELOG.md](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/algebraic-range/CHANGELOG.md) for the 0.9 rewrite notes.
