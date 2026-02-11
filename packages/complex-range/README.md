# complex-range

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/complex-range.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/complex-range.yml)
[![PyPI version](https://badge.fury.io/py/complex-range.svg)](https://badge.fury.io/py/complex-range)
[![Python versions](https://img.shields.io/pypi/pyversions/complex-range.svg)](https://pypi.org/project/complex-range/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Generate ranges of complex numbers - rectangular grids and linear sequences in the complex plane.

Python port of the **Wolfram** [resource function "ComplexRange"](https://resources.wolframcloud.com/FunctionRepository/resources/ComplexRange/).


## Installation

```bash
pip install complex-range
```


## Overview

`complex_range` generates ranges of complex numbers, extending Python's `range` functionality to the complex plane. It supports two modes:

- **Rectangular ranges**: Generate all points on a 2D grid spanning from minimum to maximum real and imaginary parts
- **Linear ranges**: Generate points along a line between two complex numbers

Step sizes can be specified as a complex number (e.g., `0.5+0.5j`) or as separate real and imaginary steps in list form (e.g., `[0.5, 0.5]`). By default, the imaginary part is incremented first throughout its range with the real part fixed, then the real part is incremented—this behavior can be reversed via the `increment_first` option.

The `farey_range` option enables a complex generalization of the Farey sequence, creating mathematically refined point distributions in the complex plane.


## Usage

### Rectangular Ranges

Generate a grid of complex numbers between two corners:

```
Im ↑
 4 │  ·  ·  ·
 3 │  ·  ·  ·
 2 │  ·  ·  ·
 1 │  ·  ·  ·
   └──────────→ Re
      1  2  3
```

```python
from complex_range import complex_range

# Basic rectangular range
complex_range(0, 2+2j)
# Returns: [0j, 1j, 2j, (1+0j), (1+1j), (1+2j), (2+0j), (2+1j), (2+2j)]

# Single argument: range from 0 to z
complex_range(2+3j)
# Returns grid from 0 to 2+3j (3×4 = 12 points)

# With custom step size (complex number)
complex_range(0, 2+2j, 0.5+0.5j)
# Real step = 0.5, Imaginary step = 0.5

# With separate real and imaginary steps
complex_range(0, 4+6j, [2, 3])
# Real step = 2, Imaginary step = 3
```

With `increment_first='im'` (default), points are generated column by column:
`(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), ...`

### Linear Ranges

Generate points along a line (diagonal) in the complex plane:

```
Im ↑
 4 │        ·
 3 │     ·
 2 │  ·
 1 │·
   └──────────→ Re
     1  2  3
```

Points increment along both axes simultaneously.

```python
# Linear from 0 to endpoint
complex_range([3+3j])
# Returns: [0j, (1+1j), (2+2j), (3+3j)]

# Linear between two points
complex_range([-1-1j, 2+2j])
# Returns: [(-1-1j), 0j, (1+1j), (2+2j)]

# Linear with custom step
complex_range([0, 4+4j], [2, 2])
# Returns: [0j, (2+2j), (4+4j)]

# Linear with complex step
complex_range([0, 4+4j], 2+2j)
# Returns: [0j, (2+2j), (4+4j)]

# Descending linear range with negative step
complex_range([2+2j, 0], -1-1j)
# Returns: [(2+2j), (1+1j), 0j]
```

### Options

#### `increment_first`

Control the iteration order for rectangular ranges:

```python
# Default: increment imaginary first
complex_range(0, 2+2j, 1+1j, increment_first='im')
# [0j, 1j, 2j, (1+0j), (1+1j), (1+2j), (2+0j), (2+1j), (2+2j)]

# Increment real first
complex_range(0, 2+2j, 1+1j, increment_first='re')
# [0j, (1+0j), (2+0j), 1j, (1+1j), (2+1j), 2j, (1+2j), (2+2j)]
```

#### `farey_range`

Use Farey sequence to create a finer subdivision of the grid:

```python
# Regular grid
regular = complex_range(0, 4+4j, 2+2j)
# Returns 9 points

# Farey subdivision (step must be integer)
farey = complex_range(0, 4+4j, 2+2j, farey_range=True)
# Returns 81 points using Farey sequence F_2

# Farey sequence of order n creates fractions 0, 1/n, ..., 1
from complex_range import farey_sequence
farey_sequence(3)
# [Fraction(0,1), Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(1,1)]
```



## Applications

### Mandelbrot Set Visualization

```python
from complex_range import complex_range
import matplotlib.pyplot as plt
import numpy as np

def mandelbrot_escape(c, max_iter=100):
    z = 0
    for i in range(max_iter):
        z = z*z + c
        if abs(z) > 2:
            return i
    return max_iter

# Grid parameters
re_min, re_max = -2, 1
im_min, im_max = -1.5, 1.5
step = 0.005

# Generate grid and compute escape times
grid = complex_range(re_min + im_min*1j, re_max + im_max*1j, [step, step])
escapes = [mandelbrot_escape(c) for c in grid]

# Reshape and plot
n_re = int((re_max - re_min) / step) + 1
n_im = int((im_max - im_min) / step) + 1
escape_array = np.array(escapes).reshape(n_re, n_im).T

plt.figure(figsize=(10, 8))
plt.imshow(escape_array, extent=[re_min, re_max, im_min, im_max],
           origin='lower', cmap='hot', aspect='equal')
plt.colorbar(label='Escape iterations')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Mandelbrot Set')
plt.savefig('mandelbrot.png', dpi=150)
plt.show()
```


## See Also

For more examples and details, see the documentation for the corresponding [Wolfram Language "ComplexRange"](https://resources.wolframcloud.com/FunctionRepository/resources/ComplexRange/) resource function, contributed by the same author and vetted by the Wolfram Review Team.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

