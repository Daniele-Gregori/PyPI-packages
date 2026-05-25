"""Core TBA solver.

Implements the method of successive approximations with FFT-based convolution,
faithfully translated from the ThermodynamicBetheAnsatzSolve Wolfram Language
resource function (1.0.0) by Daniele Gregori.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy.interpolate import interp1d


class TBASolution:
    """An interpolated TBA solution component.

    Callable wrapper around a cubic spline interpolation of the numerical
    solution on the discretized grid.

    Parameters
    ----------
    x_grid : ndarray
        The grid points.
    values : ndarray
        Solution values (may be complex).
    label : str, optional
        Name for this solution component (e.g. ``"y1"``).
    """

    def __init__(
        self,
        x_grid: np.ndarray,
        values: np.ndarray,
        label: Optional[str] = None,
    ):
        self._x = x_grid.copy()
        self._values = values.copy()
        self._interp = interp1d(
            x_grid, values, kind="cubic",
            bounds_error=False, fill_value="extrapolate",
        )
        self.label = label
        self.domain = (float(x_grid[0]), float(x_grid[-1]))

    def __call__(self, x):
        return self._interp(x)

    def __repr__(self):
        tag = f"'{self.label}' " if self.label else ""
        return (
            f"TBASolution({tag}"
            f"domain=[{self.domain[0]:.6g}, {self.domain[1]:.6g}])"
        )


class TBASolver:
    """Numerical solver for Thermodynamic Bethe Ansatz integral equations.

    Solves systems of nonlinear integral equations of the form::

        y_j(x) = f_j(x)
                  + sum_k int phi_{j,k}(x-t)
                    * log[1 + c_{j,k} * exp(sigma_{j,k} * y_{cross_k}(t))] dt

    using the method of successive approximations with FFT-based convolution.

    Parameters
    ----------
    forcing : list of callable
        Forcing (non-homogeneous) terms ``f_j(x)``, one per equation.
        Each callable accepts and returns an ndarray.
    kernels : list of list of callable
        Convolution kernels ``phi_{j,k}(x)``.  ``kernels[j]`` is the list
        of kernel functions for equation *j*.
    crossing : list of list of int
        ``crossing[j][k]`` is the 0-based index of the dependent variable
        that appears inside the *k*-th log-term of equation *j*.
    constants : list of list of float, optional
        Coefficients ``c_{j,k}`` inside the log terms (default all 1).
    signs : list of list of int, optional
        Signs ``sigma_{j,k}`` inside the exponentials (default all -1).
    grid_cutoff : float
        Half-width of the symmetric grid ``[-cutoff, cutoff]``.
    grid_resolution : int
        Number of grid points (should be a power of 2 for FFT efficiency).
    stopping_accuracy : float
        Stop iterating when the relative error drops below this threshold.
    max_iterations : int
        Hard upper bound on iterations (must be a multiple of 100).
    boundary_ext : list of callable or float, optional
        External boundary terms added to forcing (default 0).
    boundary_int : list of list of callable or float, optional
        Internal boundary terms added inside each convolution (default 0).
    damping : float
        Relaxation parameter alpha in ``(0, 1)``.  The update rule is
        ``sol_new = alpha * sol_old + (1 - alpha) * (forcing + conv)``.
    monitor : bool
        Print iteration progress.
    labels : list of str, optional
        Names for the solution components.
    """

    def __init__(
        self,
        forcing: list[Callable],
        kernels: list[list[Callable]],
        crossing: list[list[int]],
        constants: Optional[list[list[float]]] = None,
        signs: Optional[list[list[int]]] = None,
        *,
        grid_cutoff: float = 100.2,
        grid_resolution: int = 1024,
        stopping_accuracy: float = 1e-10,
        max_iterations: int = 4000,
        boundary_ext: Optional[list[Union[Callable, float]]] = None,
        boundary_int: Optional[list[list[Union[Callable, float]]]] = None,
        damping: float = 0.1,
        monitor: bool = False,
        labels: Optional[list[str]] = None,
    ):
        self.n_eq = len(forcing)
        self.forcing = forcing
        self.kernels = kernels
        self.crossing = crossing

        if constants is None:
            constants = [[1.0] * len(k) for k in kernels]
        self.constants = constants

        if signs is None:
            signs = [[-1] * len(k) for k in kernels]
        self.signs = signs

        self.grid_cutoff = float(grid_cutoff)
        self.grid_resolution = int(grid_resolution)
        self.stopping_accuracy = float(stopping_accuracy)
        self.max_iterations = int(max_iterations)
        self.damping = float(damping)
        self.monitor = bool(monitor)
        self.labels = labels

        if boundary_ext is None:
            boundary_ext = [0.0] * self.n_eq
        self.boundary_ext = boundary_ext

        if boundary_int is None:
            boundary_int = [[0.0] * len(k) for k in kernels]
        self.boundary_int = boundary_int

        self._validate()

    def _validate(self):
        n = self.n_eq
        if len(self.kernels) != n:
            raise ValueError(
                f"Expected {n} kernel lists, got {len(self.kernels)}"
            )
        if len(self.crossing) != n:
            raise ValueError(
                f"Expected {n} crossing lists, got {len(self.crossing)}"
            )
        for j in range(n):
            nk = len(self.kernels[j])
            if len(self.crossing[j]) != nk:
                raise ValueError(
                    f"Equation {j}: {nk} kernels but "
                    f"{len(self.crossing[j])} crossing entries"
                )
            if len(self.constants[j]) != nk:
                raise ValueError(
                    f"Equation {j}: {nk} kernels but "
                    f"{len(self.constants[j])} constants"
                )
            if len(self.signs[j]) != nk:
                raise ValueError(
                    f"Equation {j}: {nk} kernels but "
                    f"{len(self.signs[j])} signs"
                )
            for idx in self.crossing[j]:
                if not (0 <= idx < n):
                    raise ValueError(
                        f"Crossing index {idx} out of range "
                        f"for {n} equations"
                    )

    def _build_grid(self):
        res = self.grid_resolution
        cut = self.grid_cutoff
        return np.linspace(-cut, -cut + 2 * cut * (res - 1) / res, res)

    def _tabulate(self, func, grid):
        if callable(func):
            result = np.asarray(func(grid), dtype=complex)
            if result.shape == ():
                return np.full(len(grid), result, dtype=complex)
            return result
        return np.full(len(grid), complex(func), dtype=complex)

    def solve(self) -> list[TBASolution]:
        """Run the iterative solver and return interpolated solutions.

        Returns
        -------
        list of TBASolution
            One callable interpolation per equation.

        Raises
        ------
        RuntimeError
            If overflow or NaN is detected (TBA likely non-convergent).
        """
        res = self.grid_resolution
        cut = self.grid_cutoff
        n_eq = self.n_eq
        alpha = self.damping

        grid = self._build_grid()

        # --- tabulate and FFT kernels ---
        ker_fft = []
        for j in range(n_eq):
            ker_fft_j = []
            for k in range(len(self.kernels[j])):
                kv = self._tabulate(self.kernels[j][k], grid)
                ker_fft_j.append(np.fft.fft(kv))
            ker_fft.append(ker_fft_j)

        # --- tabulate forcing + external boundary ---
        forc_tab = []
        for j in range(n_eq):
            fv = self._tabulate(self.forcing[j], grid)
            bv = self._tabulate(self.boundary_ext[j], grid)
            forc_tab.append(fv + bv)

        # --- tabulate internal boundary ---
        bdy_int_tab = []
        for j in range(n_eq):
            row = []
            for k in range(len(self.kernels[j])):
                row.append(self._tabulate(self.boundary_int[j][k], grid))
            bdy_int_tab.append(row)

        # --- phase table for FFT-based convolution on centred grid ---
        phase_tab = (2 * cut / res) * np.exp(
            -1j * np.pi * np.arange(res)
        )

        def conv_fourier(kernel_f, func_vals):
            return np.fft.ifft(
                phase_tab * kernel_f * np.fft.fft(func_vals)
            )

        # --- iteration ---
        solution = [f.copy() for f in forc_tab]

        def iterate(sol_old):
            sol_new = []
            for j in range(n_eq):
                n_kern = len(self.kernels[j])
                conv_sum = np.zeros(res, dtype=complex)
                for k in range(n_kern):
                    idx = self.crossing[j][k]
                    c = self.constants[j][k]
                    s = self.signs[j][k]
                    fL = np.log(1.0 + c * np.exp(s * sol_old[idx]))
                    conv_sum += conv_fourier(
                        ker_fft[j][k], fL + bdy_int_tab[j][k]
                    )
                sol_new.append(
                    alpha * sol_old[j]
                    + (1 - alpha) * (forc_tab[j] + conv_sum)
                )
            return sol_new

        total_iterations = 0

        with np.errstate(over="ignore", invalid="ignore"):
            for block in range(1, self.max_iterations // 100 + 1):
                for _ in range(100):
                    prev = [s.copy() for s in solution]
                    solution = iterate(solution)

                total_iterations = block * 100

                # check for overflow / NaN
                for j in range(n_eq):
                    if not np.all(np.isfinite(solution[j])):
                        raise RuntimeError(
                            "Overflow detected: the Thermodynamic Bethe "
                            "Ansatz is likely non-convergent for the "
                            "specified parameters."
                        )

                # convergence check
                errs = self._compute_errors(solution, prev)
                if self.monitor:
                    print(f"Iteration {total_iterations}, "
                          f"max error: {max(errs):.2e}")

                if all(e < self.stopping_accuracy for e in errs):
                    break

        if self.monitor:
            print(f"Total iterations: {total_iterations}")

        # --- build interpolated solutions ---
        results = []
        for j in range(n_eq):
            label = self.labels[j] if self.labels else None
            results.append(TBASolution(grid, solution[j], label=label))
        return results

    @staticmethod
    def _compute_errors(sol_new, sol_old):
        errs = []
        for j in range(len(sol_new)):
            denom = sol_new[j] + sol_old[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = 2.0 * (sol_new[j] - sol_old[j]) / denom
                rel = np.where(np.isfinite(rel), rel, 0.0)
            max_re = float(np.max(np.real(rel)))
            max_im = float(np.max(np.imag(rel)))
            errs.append(max(max_re, max_im))
        return errs
