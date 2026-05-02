# find-closed-form



**DISCLAIMER**

**This is a very early vibe-coded version and its use is not recommended. The reason for such *pre-crastination* is just that the maintainer *dangregori* realized that in these days (first weekend of May 2026 A.D.) the *Python Package Index* topped the 800k projects. When he joined this open mess just half a year before there were about 100k less. Clearly there are too many other vibe coders around here... but the maintainer will not allow them to spam the namespace of his own expert-reviewed and published *Wolfram* *[ResourceFunction["FindClosedForm"]](https://resources.wolframcloud.com/FunctionRepository/resources/FindClosedForm/)*.**

---

Given a numerical value, `find_closed_form` searches for closed-form
mathematical expressions that match it — testing candidates directly by
comparing digits and filtering by a formula-complexity heuristic.

**This will progressively become a Python port** of the Wolfram Language ResourceFunction
[FindClosedForm](https://resources.wolframcloud.com/FunctionRepository/resources/FindClosedForm/), contributed by the same author and vetted by the Wolfram reviewers.




## Quick start

```python
from find_closed_form import find_closed_form

# Identify sqrt(2)/2
find_closed_form(0.7071067811865476)
# [sqrt(2)/2]

# Identify the Euler–Mascheroni constant
find_closed_form(0.5772156649015329)
# [EulerGamma]


# bad: less digits do not work automatically
# bad: single output should not be a list
```

## How it works

1. **Argument-range generation** — For each search round the algorithm
   expands a Farey-based range of rational arguments (built-in
   `farey_range` implementation).

2. **Function evaluation** — A library of ~27 common mathematical
   functions (trig, exponential, logarithmic, special functions, and
   mathematical constants) is evaluated over the argument range.

3. **Digit matching** — Candidate expressions are accepted when
   `|1 − candidate/target| ≤ 10^(−digits+1)`, where *digits* is
   auto-detected from the input or set explicitly.

4. **Algebraic combinations** — Beyond direct `f(arg)` matches, the
   algorithm also searches for:
   - **Multiplicative**: `a · f(b) ≈ target`, where *a* is a simple
     algebraic number (integer, rational, or root).
   - **Additive**: `a + f(b) ≈ target`.

5. **Complexity filtering** — Every candidate is scored using the
   built-in `formula_complexity` heuristic and discarded if it exceeds
   the threshold.

## API

```python
find_closed_form(
    y,                                  # target number (int, float, Fraction, sympy)
    functions=None,                     # callable, list of callables, or list of (name, fn, filter) tuples
    max_results=1,                      # how many results to return
    *,
    significant_digits=None,            # precision target (auto-detected if None)
    formula_complexity_threshold=None,  # max allowed complexity (auto-scaled if None)
    algebraic_factor=True,              # enable a·f(b,...) search
    algebraic_add=True,                 # enable a+f(b,...) search
    max_search_rounds=50,               # max argument-range expansion rounds
)
```

## Examples

### 1. Recognise a trigonometric value

```python
import math
find_closed_form(math.cos(math.pi / 5))
# [GoldenRatio/2]
```

### 2. Identify a logarithmic expression

```python
find_closed_form(0.6931471805599453)
# [log(2)]
```

### 3. Pass a pure function to search over

```python
from sympy import sin, pi

find_closed_form(0.8660254037844386, functions=lambda x: sin(pi * x))
# [sin(pi/3)]
# bad actual result is [] 
```

### 4. Search with a named function and domain filter

```python
from sympy import asin

find_closed_form(
    1.0471975511965976,
    functions=[("asin(#)", lambda x: asin(x), lambda x: 0 <= x <= 1)],
)
# [asin(sqrt(3)/2)]
```

### 5. Combine multiple functional forms

```python
from sympy import cos, exp, pi

find_closed_form(
    0.5,
    functions=[lambda x: sin(pi * x), lambda x: cos(pi * x)],
    max_results=3,
)
# [sin(pi/6), cos(pi/3), ...]
```

### 6. Two-argument function — multiplicative

```python
from sympy import log

# 2 * log(3) ≈ 2.1972...
find_closed_form(2.1972245773362196, functions=lambda x, y: x * log(y))
# [2*log(3)]
```

### 7. Two-argument function with domain filter

```python
find_closed_form(
    1.0,
    functions=[(
        "x*sin(pi*y)",
        lambda x, y: x * sin(pi * y),
        lambda x, y: x > 0 and 0 < y < 1,
    )],
)
# [2*sin(pi/6)]
```

### 8. Three-argument function

```python
# Search for x + y*log(z)
find_closed_form(
    2.3862943611198906,
    functions=lambda x, y, z: x + y * log(z),
    max_search_rounds=5,
)
# [1 + 2*log(2)] or equivalent
```

### 9. Use a regular Python function (not just lambdas)

```python
from sympy import exp, sqrt

def my_form(x):
    return exp(x) / sqrt(2)

find_closed_form(1.9221276790498548, functions=my_form)
# [exp(1)/sqrt(2)] or equivalent
```

### 10. Fewer significant digits — find approximate matches

```python
find_closed_form(1.414, significant_digits=4, max_results=5)
# [sqrt(2), ...] and other expressions close to 1.414
```

### 11. Stricter matching with more significant digits

```python
find_closed_form(1.4142135623730951, significant_digits=15)
# [sqrt(2)]
```

### 12. Controlling complexity to prefer simpler expressions

```python
find_closed_form(0.7071067811865476, formula_complexity_threshold=10)
# [sqrt(2)/2]
```

## Formula complexity

The complexity of a candidate expression is computed as the sum of
per-integer complexities for every integer appearing in the expression:

```
int_complexity(n) = 0.5 × mean(digit_sum, 5×len, #prime_factors, √n)
```

This is the same definition as the WL
[AlgebraicRange](https://resources.wolframcloud.com/FunctionRepository/resources/AlgebraicRange/)
resource function.

## Dependencies

- [sympy](https://www.sympy.org/) — symbolic mathematics (only runtime dependency)

## License

MIT
