# find-closed-form

[![find-closed-form](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/find-closed-form.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/find-closed-form.yml)
[![PyPI](https://badge.fury.io/py/find-closed-form.svg)](https://pypi.org/project/find-closed-form/)
[![Python](https://img.shields.io/pypi/pyversions/find-closed-form)](https://pypi.org/project/find-closed-form/)

A Python port of the Wolfram Language ResourceFunction
[FindClosedForm](https://resources.wolframcloud.com/FunctionRepository/resources/FindClosedForm/),
contributed by the same author.

`find_closed_form` helps solve the fundamental problem of
[number recognition](https://mathworld.wolfram.com/NumberRecognition.html),
by searching for a possible closed-form formula for a given number `y`,
in terms of arbitrary combinations of elementary and higher mathematical
functions.

The fundamental strategy is that, given a callable `f`, progressively more
complex rational arguments are tried, until a numerical match with the given
value `y` is found. By default, this match is searched up to linear
combinations with algebraic numbers (rationals or roots).

When no functional form is specified, for each round of argument search,
a further search goes through the following common mathematical functions:
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `acot`, `log`, `exp`,
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `acoth`,
`zeta`, `gamma`, `polygamma`, `erf`, `erfinv`,
`elliptic_k`, `elliptic_e`, `airyai`, `airybi`.
In addition, `find_closed_form` searches among algebraic combinations of the
following mathematical constants:
`pi`, `EulerGamma`, `Catalan`, `GoldenRatio`.

## Installation

```bash
pip install find-closed-form
```

## Quick start

Find a possible mathematical function for a number:

```python
from find_closed_form import find_closed_form

find_closed_form(0.405465)          # log(3/2)
```

Find possible closed forms in terms of common mathematical functions:

```python
find_closed_form(3.792277)          # 1/6 + gamma(1/4)
```

Find formulae in terms of mathematical constants:

```python
find_closed_form(1.044866)          # 1/sqrt(Catalan)
```

Specify the functional form as a callable:

```python
from sympy import zeta

find_closed_form(1.85653, functions=lambda x: 1/zeta(x)**2)
# zeta(1/5)**(-2)
```

## Scope

The numerical match with the functional form is searched up to addition or
multiplication by an algebraic number (that is, a rational or root):

```python
from sympy import asinh, log, exp

find_closed_form(0.780653, functions=lambda x: asinh(x))
# sqrt(5)*asinh(4)/6

find_closed_form(7.443967, functions=lambda x: log(1 + exp(x)))
# 10*log(1 + exp(1/10))
```

Multi-argument functions are supported:

```python
from sympy import gamma, log

find_closed_form(6.263643,
    functions=lambda x, y: log(x)*log(y),
    search_range="Integer")
# 2*log(5)*log(7)

find_closed_form(14.911818,
    functions=lambda x, y: gamma(x)*gamma(y),
    search_range="Plain")
# gamma(1/6)*gamma(1/3)
```

Search through a list of functional forms:

```python
from sympy import sinh, cosh, sech, csch

find_closed_form(5.550045, functions=[
    lambda x: sinh(x), lambda x: cosh(x),
    lambda x: sech(x), lambda x: csch(x),
])
# 6*sech(2/5)
```

Multiple results can be requested through `max_results`:

```python
find_closed_form(0.405465, functions=lambda x: log(x), max_results=10)
# returns multiple results, first = log(3/2)
```

## Options

### `algebraic_add`

Setting `algebraic_add=False` restricts the search to the specified functional
form up to multiplication (but not addition) by an algebraic number. This
can speed up the search, since special range properties are exploited for
certain known functions:

```python
from sympy import gamma

find_closed_form(0.1013578,
    functions=lambda x, y: 1/(gamma(x)*gamma(y)),
    algebraic_add=False)
# 1/(sqrt(pi)*gamma(1/6))
```

### `algebraic_factor`

Setting `algebraic_factor=False` restricts the search to the specified
functional form up to addition (but not multiplication) of an algebraic
number.

If both `algebraic_add` and `algebraic_factor` are set to `False`, the
search can be faster but may miss linear combinations of the functional form.

### `formula_complexity_threshold`

If not enough digits are specified, a careful balance between precision and
complexity of the result should be reached through `formula_complexity_threshold`.
Often the desired formula is the simplest. For example:

```python
from sympy import gamma

find_closed_form(38.94017, functions=lambda x: gamma(x),
    formula_complexity_threshold=15)
# 2*gamma(1/20)
```

The formula complexity is a positive real value which ranks complexity as
follows: take all integers appearing in the formula (expanding rationals,
roots, etc.); for each integer, compute the mean among the square root of
its absolute value, its digit sum, and 5 times its number of digits; then
take the total of these means.

### `max_search_rounds`

The maximum number of argument search rounds is 50 by default. This also
determines the largest integer argument and rational denominator reachable:

```python
from sympy import gamma

find_closed_form(49.44221, functions=lambda x: gamma(x),
    algebraic_add=False, algebraic_factor=False, search_range="Plain")
# gamma(1/50)
```

By default, larger arguments are not reachable:

```python
find_closed_form(59.43902, functions=lambda x: gamma(x),
    algebraic_add=False, algebraic_factor=False, search_range="Plain")
# None
```

Changing the value of `max_search_rounds` allows a solution to be found:

```python
find_closed_form(59.43902, functions=lambda x: gamma(x),
    max_search_rounds=100, algebraic_add=False, algebraic_factor=False,
    search_range="Plain")
# gamma(1/60)
```

### `rational_solutions`

By default, simple rational solutions are not returned, and more sophisticated
solutions are searched for. If `rational_solutions=True`, simple exact
rational solutions are allowed:

```python
from sympy import sin, pi

find_closed_form(0.25, functions=lambda x: sin(pi*x),
    rational_solutions=True, algebraic_add=False)
# 1/4
```

If the functional form is the identity, there is no need for this option:

```python
find_closed_form(0.25, functions=lambda x: x)
# 1/4
```

### `search_arguments`

Through `search_arguments` you can specify each particular argument
to be tried:

```python
from sympy import gamma
from fractions import Fraction

find_closed_form(4.678938, functions=lambda x: gamma(x),
    search_arguments=[Fraction(3), Fraction(1), Fraction(1, 3)])
# 2 + gamma(1/3)
```

This can speed up the search and serves as a debugging tool.

### `search_range`

By default, for each search round the arguments span the Farey range
`farey_range(-round, round, round)`, which consists of rationals of
uniform complexity. The following values are supported:

| Value | Range per round |
|-------|-----------------|
| `"Farey"` | `farey_range(-cut, cut, cut)` — a rational Farey range |
| `"Plain"` | `range(-cut, cut, 1/cut)` — the shorter rational range |
| `"Integer"` | `range(-cut, cut)` — purely integer arguments |

```python
from sympy import log

find_closed_form(6.263643,
    functions=lambda x, y: log(x)*log(y),
    search_range="Integer")
# 2*log(5)*log(7)
```

### `search_range_fn`

It is possible to specify a custom range function of the search round number:

```python
from sympy import log
from fractions import Fraction

find_closed_form(13.165149, functions=lambda x: log(x),
    search_range_fn=lambda cut: [Fraction(i) for i in range(0, 100*cut+1, 25)])
# sqrt(3)*log(2000)
```

### `significant_digits`

The precision of the numerical match is automatically set to the number
of significant digits in the given number. If you want to ignore some
numerical error, you can specify a lower value:

```python
from sympy import zeta

find_closed_form(0.81248057539,
    functions=lambda x: 1/zeta(x)**2,
    significant_digits=7)
# zeta(11/3)**(-2)
```

### `search_time_limit`

The maximum time in seconds spent by the search algorithm. Default is 3600.

### Summary table

| Parameter | Default | Description |
|-----------|---------|-------------|
| `functions` | `None` | Functional forms to search; `None` uses ~29 common functions. |
| `max_results` | `1` | Number of results to return. |
| `significant_digits` | Auto | Precision target; auto-detected from input digits. |
| `formula_complexity_threshold` | Auto | Maximum formula complexity; auto-scaled per round. |
| `algebraic_factor` | `True` | Search up to multiplication by algebraic numbers. |
| `algebraic_add` | `True` | Search up to addition of algebraic numbers. |
| `rational_solutions` | `False` | Allow simple rational solutions. |
| `max_search_rounds` | `50` | Maximum argument-range expansion rounds. |
| `search_range` | `"Farey"` | Argument range type: `"Farey"`, `"Plain"`, or `"Integer"`. |
| `search_range_fn` | `None` | Custom `f(cut) → list` for argument generation. |
| `search_arguments` | `None` | Fixed argument list (bypasses auto ranges). |
| `search_time_limit` | `3600` | Maximum seconds for the search. |

## Properties and relations

`find_closed_form` with the identity function generalizes rationalization
and works with fewer digits:

```python
find_closed_form(0.666, functions=lambda x: x)   # 2/3
```

When the given number approximates a simple root, it also generalizes
root approximation:

```python
find_closed_form(4.243, functions=lambda x: x)    # 3*sqrt(2)
find_closed_form(0.5848, functions=lambda x: x)   # 5**(-1/3)
```

## Auxiliary functions

The `formula_complexity` function is also exported and can be used directly
to compute the complexity of any sympy expression:

```python
from find_closed_form import formula_complexity
from sympy import Rational

formula_complexity(2*gamma(Rational(1, 20)))
```

The `farey_range` function generates Farey-based argument ranges:

```python
from find_closed_form import farey_range

farey_range(-3, 3, 3)
# [-3, -8/3, -5/2, ..., 5/2, 8/3, 3]
```

## Dependencies

- [sympy](https://www.sympy.org/) ≥ 1.12
- [farey](https://pypi.org/project/farey/) ≥ 0.6.0

## License

MIT
