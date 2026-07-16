# transcendental-range

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/transcendental-range.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/transcendental-range.yml)
[![PyPI version](https://badge.fury.io/py/transcendental-range.svg)](https://badge.fury.io/py/transcendental-range)
[![Python](https://img.shields.io/pypi/pyversions/transcendental-range.svg)](https://pypi.org/project/transcendental-range/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Generate ranges of transcendental numbers.**

A Python port of the Wolfram Language resource function
[TranscendentalRange](https://resources.wolframcloud.com/FunctionRepository/resources/TranscendentalRange)
(version 1.1.0), returning exact [sympy](https://www.sympy.org/) expressions.

```
pip install transcendental-range
```

```python
from transcendental_range import transcendental_range

transcendental_range(10)
# [E, 2*E, exp(2), 3*E]
```

## Overview

`transcendental_range` creates ranges of
[transcendental numbers](https://en.wikipedia.org/wiki/Transcendental_number)
— numbers that provably cannot be the root of any polynomial equation
with integer coefficients. Whole families of such numbers follow from
classical theorems (**Lindemann–Weierstrass**, **Gelfond–Schneider**,
**Baker**), and this function generates them systematically over 27
methods covering exponential, logarithmic, trigonometric, hyperbolic and
inverse forms, plus algebraic powers:

```python
>>> from transcendental_range import transcendental_range
>>> transcendental_range(10)
[E, 2*E, exp(2), 3*E]

>>> transcendental_range(-2, 2)
[-2*exp(-1), -exp(-1), -2*exp(-2), -exp(-2), exp(-2), 2*exp(-2), exp(-1), 2*exp(-1)]
```

```python
>>> from sympy import Rational
>>> transcendental_range(0, 4, Rational(1, 2))          # with step
[exp(1/2)/2, E/2, exp(1/2), exp(3/2)/2, 3*exp(1/2)/2, E, 2*exp(1/2), exp(2)/2]

>>> transcendental_range(100, 1, -1)                    # descending
[3*exp(3), ..., 2*exp(2), ..., E]
```

| Form | Description |
|------|-------------|
| `transcendental_range(x)` | transcendentals *t* = *b* f(*a*) with 1 ≤ *t* ≤ *x*, *a* and *b* in `range(1, x+1)` |
| `transcendental_range(x, y)` | with *x* ≤ *t* ≤ *y*, *a* and *b* in `range(x, y+1)` |
| `transcendental_range(x, y, s)` | *a* and *b* in `range(x, y+1, s)` (negative *s* for descending) |
| `transcendental_range(x, y, s, d)` | minimum difference *d* between successive elements |

Every element is an exact sympy expression, sorted by numerical value;
numerically coincident alternatives are reduced to the representative
with the smallest function argument. The implementation uses a
monotonicity-aware scan that is far more efficient than the naive outer
product (see [Performance](#performance)).

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `method` | `'exp'` | transcendental function generating the numbers |
| `multiplicity` | `1` | number of terms to combine linearly |
| `generators_domain` | `'rationals'` | domain of the algebraic generators |
| `farey_range` | `False` | step denominators as in the Farey sequence |
| `formula_complexity_threshold` | `inf` | limit the complexity of the expressions |
| `working_precision` | `15` | precision of all internal numerical evaluations |

### method

| Method | Generated forms |
|--------|-----------------|
| `'exp'` | exponential forms *b* e<sup>*a*</sup> |
| `'log'` | logarithmic forms *b* log(*a*) |
| `'power'` | power forms *a*<sup>*b*</sup>, for irrational *b* only |
| `'sin'`, `'cos'`, `'tan'`, … | trigonometric forms *b* sin(*a*), *b* cos(*a*), … |
| `'asin'`, `'acos'`, `'atan'`, … | inverse trigonometric forms |
| `'sinh'`, `'cosh'`, `'tanh'`, … | hyperbolic forms |
| `'asinh'`, `'acosh'`, `'atanh'`, … | inverse hyperbolic forms |
| a list of the above | combined range |
| `'all'` | all of the above types |

```python
transcendental_range(-10, 10, method='sinh')
# [-8*sinh(1), -7*sinh(1), -2*sinh(2), -6*sinh(1), ..., 2*sinh(2), 7*sinh(1), 8*sinh(1)]

transcendental_range(-2, 2, method='atan')
# [-pi/2, -atan(2), -pi/4, pi/4, atan(2), pi/2]
```

Different types can be freely combined:

```python
transcendental_range(-4, 4, method=['acot', 'exp'])
# [-pi, -E, -3*pi/4, -4*acot(2), -pi/2, -4*exp(-1), ..., 4*acot(2), 3*pi/4, E, pi]

transcendental_range(3, method='all')
# [coth(3), 3*acsc(3), coth(2), 3*acoth(3), pi/3, 2*cos(1), log(3), csc(2), ...]
```

The method `'power'` produces nontrivial results together with
`generators_domain='algebraics'` (only irrational exponents generate
transcendental powers, by Gelfond–Schneider):

```python
transcendental_range(-3, 3, method='power', generators_domain='algebraics')
# [3**(-2*sqrt(2)), 2**(-3*sqrt(2)), 3**(-sqrt(7)), 7**(-sqrt(2)), ...]
```

### multiplicity

Linear combinations of the generated numbers are still transcendental:

```python
from sympy import Rational

transcendental_range(0, 6, Rational(1, 2), multiplicity=2)
# [exp(1/2), exp(1/2)/2 + E/2, 3*exp(1/2)/2, E, E/2 + exp(1/2), ...]
```

For the method `'power'` the combinations are products; different methods
of a list combine only within themselves.

### generators_domain

With `'algebraics'`, the generator arguments and coefficients extend from
the rationals of `range(x, y, s)` to the real algebraic numbers of
[`algebraic_range`](https://pypi.org/project/algebraic-range/):

```python
transcendental_range(8, generators_domain='algebraics')
# [E, sqrt(2)*E, exp(sqrt(2)), sqrt(3)*E, 2*E, exp(sqrt(3)), ...]
```

The rational range is always a subset of the algebraic one.

### farey_range

Step denominators can follow the Farey sequence, as generated by the
[`farey`](https://pypi.org/project/farey/) package; the resulting range is
a strict superset of the union of the plain ranges with the corresponding
steps:

```python
transcendental_range(1, 10, Rational(1, 3), farey_range=True)
# [E, 4*E/3, exp(4/3), 3*E/2, exp(3/2), 5*E/3, 4*exp(4/3)/3, ...]
```

A Farey step must be an integer ≥ 1 or of the form 1/*n* (or −1/*n* for a
descending range): anything else raises `FareyStepError`.

### formula_complexity_threshold

The output can be restricted to expressions below a heuristic complexity
score (from `algebraic_range.formula_complexity`):

```python
transcendental_range(0, 8, Rational(1, 2), generators_domain='algebraics',
                     formula_complexity_threshold=5)
# 14 simple elements, against 382 with the default infinite threshold
```

### working_precision

Symbolically distinct elements may numerically collide at machine
precision, in which case only one representative is kept:

```python
transcendental_range(20, 25, method='tanh')
# [21*tanh(20), 22*tanh(20), 23*tanh(20), 24*tanh(20), 25*tanh(20)]
```

Raising the precision resolves the collisions:

```python
transcendental_range(20, 25, method='tanh', working_precision=30)
# [21*tanh(20), 21*tanh(21), ..., 25*tanh(24), 25*tanh(25)]  (30 elements)
```

## Properties and relations

`transcendental_range(x, y, s)` equals the outer product of
*b* e<sup>*a*</sup> over the generator range, restricted to the bounds,
cleared of algebraic values, deduplicated and sorted — but the actual
monotonicity-aware implementation is far more efficient than the outer
product over large exponentially-growing ranges, and every method is
verified against that naive baseline in the test suite.

## Performance

The Python port reproduces the WL output exactly (80 604 elements match
one for one) within 2–3× of the native WL timings. See
[benchmark/BENCHMARK.md](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/transcendental-range/benchmark/BENCHMARK.md)
for per-method timings and methodology.

## Possible issues

By definition, the range arguments must be algebraic numbers:

```python
from sympy import E
transcendental_range(0, E, Rational(1, 3))
# NotAlgebraicError: the range arguments provided are not all algebraic numbers

transcendental_range(0, 3, Rational(1, 3))   # ceiling(E) = 3
# [exp(1/3)/3, exp(2/3)/3, E/3, 2*exp(1/3)/3, ...]
```

## Applications

`transcendental_range` was designed especially as a search space for
[`find-closed-form`](https://pypi.org/project/find-closed-form/),
for exhaustively searching closed forms of raw numeric values in terms of
arbitrary mathematical functions with transcendental arguments. Since
find-closed-form 0.5.0 (`pip install "find-closed-form[ranges]"`) this
works out of the box through the `search_range` option, with
`search_range_options` forwarded to the generator.

Formulae like ζ(√e) — a higher mathematical function wrapped around a
transcendental number — are out of reach of the default rational search,
but are recovered from their bare machine digits over this range:

```python
from find_closed_form import find_closed_form

find_closed_form(2.1638308208408383, search_range="Transcendental")
# zeta(exp(1/2))
```

The range elements themselves are candidate closed forms, matched up to
algebraic factors and addends — here recognizing the Gelfond–Schneider
constant among the `'power'` elements over algebraic generators:

```python
find_closed_form(2.665144142690225, search_range="Transcendental",
                 search_range_options={"method": "power",
                                       "generators_domain": "algebraics"})
# 2**sqrt(2)
```

An explicit `transcendental_range(...)` output can also be passed directly
as fixed `search_arguments`.


## References

1. Alan Baker, *Transcendental Number Theory*, Cambridge University Press, 1975.

## See also

For the full documentation see the
[Wolfram Language resource function](https://resources.wolframcloud.com/FunctionRepository/resources/TranscendentalRange)
this package ports, and [CHANGELOG.md](CHANGELOG.md) for version history.

## Dependencies

- [sympy](https://www.sympy.org/) ≥ 1.12 (and mpmath ≥ 1.3)
- [algebraic-range](https://pypi.org/project/algebraic-range/) ≥ 0.9.0
- [farey](https://pypi.org/project/farey/) ≥ 0.7.0

## Author

Daniele Gregori

## License

MIT
