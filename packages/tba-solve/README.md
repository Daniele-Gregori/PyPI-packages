# tba-solve

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/tba-solve.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/tba-solve.yml)
[![PyPI version](https://badge.fury.io/py/tba-solve.svg)](https://badge.fury.io/py/tba-solve)
[![Python](https://img.shields.io/pypi/pyversions/tba-solve.svg)](https://pypi.org/project/tba-solve/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Numerical solver for Thermodynamic Bethe Ansatz integral equations.

## Installation

```bash
pip install tba-solve
```

Requires NumPy and SciPy.

## The equation

The Thermodynamic Bethe Ansatz (TBA) is a class of nonlinear integral equations studied in mathematical physics — in particular in statistical field theory, quantum integrability, and gauge theories. The general form is:

$$y_j(x) = f_j(x) + \sum_k \int_{-\infty}^{\infty} \varphi_{j,k}(x - t)\,\log\!\bigl[1 + c_{j,k}\,e^{\,\sigma_{j,k}\,y_k(t)}\bigr]\,dt$$

where $y_j(x)$ are the unknowns, $f_j(x)$ the forcing terms, $\varphi_{j,k}(x)$ the convolution kernels, $c_{j,k}$ some constants, and $\sigma_{j,k} = \pm 1$.

Due to its nonlinear structure, the TBA cannot be solved symbolically and requires a dedicated numerical treatment.

## Usage

### Pre-built models

```python
from tba_solve import sinh_gordon
import numpy as np

sols = sinh_gordon(r=0.1)
y = sols[0]

x = np.linspace(-5, 5, 200)
print(y(x))
```

Each model function returns a list of callable solution objects (one per dependent variable), which can be evaluated at arbitrary points within the grid domain.

Available models: `sinh_gordon`, `liouville`, `seiberg_witten_su2`.

### Custom equations

Any TBA equation can be solved by providing its decomposed components directly:

```python
from tba_solve import TBASolver
import numpy as np

solver = TBASolver(
    forcing=[lambda x: 0.1 * np.cosh(x)],
    kernels=[[lambda x: -1 / (2 * np.pi * np.cosh(x))]],
    crossing=[[0]],
)
sols = solver.solve()
```

### Coupled systems

Systems of coupled TBA equations are specified in the same way. The `crossing` parameter indicates which dependent variable each kernel term acts on:

```python
from tba_solve import TBASolver
import numpy as np

solver = TBASolver(
    forcing=[
        lambda x: 3.0 * np.exp(x),
        lambda x: 3.0 * np.exp(x),
    ],
    kernels=[
        [lambda x: -1 / (np.pi * np.cosh(x))],
        [lambda x: -1 / (np.pi * np.cosh(x))],
    ],
    crossing=[[1], [0]],   # equation 0 couples to y_1, equation 1 to y_0
)
y1, y2 = solver.solve()
```

## Parameters

| Parameter | Default | Description |
|:---|:---:|:---|
| `forcing` | — | Forcing terms $f_j(x)$, one callable per equation |
| `kernels` | — | Convolution kernels $\varphi_{j,k}(x)$, nested list of callables |
| `crossing` | — | Index of the dependent variable in each kernel term |
| `constants` | all 1 | Coefficients $c_{j,k}$ inside the log terms |
| `signs` | all −1 | Signs $\sigma_{j,k}$ inside the exponentials |

## Options

| Option | Default | Description |
|:---|:---:|:---|
| `grid_cutoff` | 100.2 | Half-width of the symmetric grid $[-L, L]$ |
| `grid_resolution` | 1024 | Number of grid points (ideally a power of 2) |
| `stopping_accuracy` | 10⁻¹⁰ | Convergence threshold on relative iteration error |
| `max_iterations` | 4000 | Hard upper bound on iterations |
| `damping` | 0.1 | Relaxation parameter for iteration stability |
| `boundary_ext` | 0 | External boundary terms, added to forcing |
| `boundary_int` | 0 | Internal boundary terms, added inside each convolution |
| `monitor` | False | Print iteration progress |
| `labels` | None | Names for the solution components |

Higher `grid_resolution` and `grid_cutoff` improve accuracy at the cost of speed. Lower `stopping_accuracy` requires more iterations; increase `max_iterations` accordingly.

## Method

The solver implements the method of successive approximations:

1. The solution is initialised to the forcing terms.
2. At each iteration, the log-terms are evaluated and convolved with the kernels using FFT, then mixed with the previous solution via a damping factor.
3. Convergence is checked every 100 iterations against the relative difference between successive iterates.
4. The converged solution is returned as a cubic spline interpolation over the grid.

The convolutions are computed in Fourier space for efficiency: $O(N \log N)$ per convolution per iteration, where $N$ is the grid resolution.

## Origin

This package is a Python translation of the [ThermodynamicBetheAnsatzSolve](https://resources.wolframcloud.com/FunctionRepository/resources/ThermodynamicBetheAnsatzSolve/) Wolfram Language resource function. The Wolfram version additionally includes an automatic symbolic equation parser; the Python version requires the user to provide the equation components explicitly.

## References

- A. B. Zamolodchikov, "Thermodynamic Bethe Ansatz in Relativistic Models", *Nucl. Phys. B* **342** (1990) 695–720.
- D. Fioravanti, D. Gregori, H. Shu, "Integrability, susy SU(2) matter gauge theories and black holes", *Nucl. Phys. B* **1021** (2025) 117200, [doi:10.1016/j.nuclphysb.2025.117200](https://doi.org/10.1016/j.nuclphysb.2025.117200), [arXiv:2208.14031](https://arxiv.org/abs/2208.14031).
- D. Gregori, "Resource function for the numerical solution of certain nonlinear integral equations", [Wolfram Community, Staff Picks, September 23, 2025](https://community.wolfram.com/groups/-/m/t/3549711).

## License

MIT
