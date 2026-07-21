# leaf-complexity

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/leaf-complexity.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/leaf-complexity.yml)
[![PyPI version](https://badge.fury.io/py/leaf-complexity.svg)](https://badge.fury.io/py/leaf-complexity)
[![Python](https://img.shields.io/pypi/pyversions/leaf-complexity.svg)](https://pypi.org/project/leaf-complexity/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compute a sum or other complexity measure over all atoms of a symbolic expression.

## Overview

Any [SymPy](https://www.sympy.org) expression is internally a tree of heads (operators, function symbols, containers) and leaves (numbers, symbols), which can be inspected with `sympy.srepr` or printed with `sympy.printing.tree`. The total number of indivisible subexpressions of that tree is the simplest measure of complexity — but expressions with the same number of leaves can be very far apart in simplicity: `x + 2` and `x + 1000000` count the same.

`leaf_complexity` directly extends the leaf count by weighing each leaf by its absolute numeric value and then taking the sum. If a leaf is non-numeric, it is counted by default as 1. Even if already atomic, rational numbers are decomposed into numerator and denominator, complex numeric literals into real and imaginary part, and dictionaries are scanned over their values.

This is a faithful port of the Wolfram Language resource function [`LeafComplexity`](https://resources.wolframcloud.com/FunctionRepository/resources/LeafComplexity/), by the same author.

## Installation

```bash
pip install leaf-complexity
```

## Usage

```python
from sympy import symbols, log
from leaf_complexity import leaf_complexity

x, y = symbols('x y')
```

`leaf_complexity(expr)` gives the sum of all the numeric indivisible subexpressions in `expr` — so expressions with the same leaf count may have different complexity:

```python
leaf_complexity(x + 2)    # 5
leaf_complexity(x + 10)   # 13
```

`leaf_complexity(expr, f)` applies a function `f` to each indivisible subexpression in `expr` and takes the total:

```python
leaf_complexity(x + 1000000, lambda v: log(v, 10))   # 7
```

`leaf_complexity(expr, f, g)` applies `f` to each indivisible subexpression and recursively applies to it another wrapping function `g`, as `s = g(s, f(leaf))`:

```python
from sympy import exp, Mul

rng = [1, 2, 3, 4, 5]
leaf_complexity(rng, exp, Mul)                    # exp(16)
leaf_complexity(rng, exp, lambda s, v: exp(v))    # exp(exp(5))
leaf_complexity(rng, exp, lambda s, v: exp(s))    # exp(exp(exp(exp(exp(E)))))
```

Setting `g` to an additive function recovers the other usage cases up to two arguments.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expr` | *(required)* | SymPy expression, number, or (nested) container |
| `f` | `None` | function applied to each leaf before totaling (`None` means absolute value) |
| `g` | `None` | binary wrapping function recursively applied as `g(s, f(leaf))` in place of the sum; requires `f` |
| `heads` | `True` | consider also the subexpression heads as leaves |

Through the option `heads` set to `False`, only proper leaf nodes are counted (parent nodes count 0):

```python
leaf_complexity(x + 1, heads=False)   # 3
leaf_complexity(x + 1)                # 4
```

### Details

* It works on any expression: SymPy expressions, plain numbers, `fractions.Fraction`, `complex`, and arbitrarily nested lists, tuples and dictionaries (the analog of WL Associations — scanned over their values only).

* A custom `f` receives the signed value of each numeric leaf, `1` for heads and non-numeric leaves, and the infinity itself for infinite leaves. The Wolfram "unary" wrapping usage corresponds in Python to a two-argument callable ignoring one of its arguments.

* An important detail in the definition and design of `leaf_complexity` is the following. Since in the most general usage case one may sometimes want to choose as wrapping function `g` a product instead of the default sum, then in order to avoid returning identically null results, all usage cases are implemented as recursions starting from initial condition 1 instead of 0. This in practice induces the somewhat awkward "correspondence principle" that if all leaves are equal to 1 or −1, `leaf_complexity(expr)` equals the leaf count (heads included) plus 1. One may desire to see complete equality with the leaf count in this special case, but a better trade-off seems to maintain the internal consistency and generality of `leaf_complexity` itself.

## Applications

`leaf_complexity` can work as a measure of complexity for algebraic expressions — for instance to steer `sympy.simplify` away from large numbers, the analog of `ComplexityFunction` in the Wolfram Language:

```python
from sympy import simplify
a, b = symbols('a b')

simplify(2*log(a) + 4*log(-4))                            # 2*log(a) + log(256) + 4*I*pi
simplify(2*log(a) + 4*log(-4), measure=leaf_complexity)   # 2*log(a) + 8*log(2) + 4*I*pi
```

Functions like [`algebraic-range`](https://pypi.org/project/algebraic-range/) tend to overproduce complex closed forms; `leaf_complexity` allows to select only the simpler ones:

```python
simple = [e for e in closed_forms if leaf_complexity(e) <= 30]
```

The resulting numeric distribution can still be nearly uniform.

## Performance

`leaf_complexity` produces exactly the same values as the Wolfram Language `LeafComplexity` resource function on identical expression trees (verified case-for-case). Because the work is a pure recursive tree scan — with no parsing hot-path to exploit — the compiled Wolfram kernel is about 2–3× faster, e.g. on large expanded polynomials; the trade-off is the lightweight SymPy-only portability. See [benchmark/BENCHMARK.md](https://github.com/Daniele-Gregori/PyPI-packages/blob/main/packages/leaf-complexity/benchmark/BENCHMARK.md) for the full table and methodology.

## See also

This Python package has been inspired by the following **Wolfram** functions:

- [**LeafComplexity**](https://resources.wolframcloud.com/FunctionRepository/resources/LeafComplexity/) (resource function contributed by the same author);
- [**LeafCount**](https://reference.wolfram.com/language/ref/LeafCount.html) (built-in Wolfram Language function).


## License

MIT
