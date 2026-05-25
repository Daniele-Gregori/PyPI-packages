"""Pre-built TBA models.

Convenience functions that configure and solve well-known Thermodynamic
Bethe Ansatz equations.  Each returns a list of :class:`TBASolution`
objects, one per dependent variable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gamma, hyp2f1

from tba_solve.solver import TBASolution, TBASolver


def sinh_gordon(
    r: float,
    b: float = 1.0,
    **solver_kwargs,
) -> list[TBASolution]:
    """Solve the Sinh-Gordon TBA.

    .. math::

        y(x) = r\\cosh(x)
               - \\int_{-\\infty}^{\\infty}
                 \\varphi_b(x-t)\\,\\log[1+e^{-y(t)}]\\,dt

    where the kernel at the self-dual point (*b* = 1) is
    ``sech(x) / (2 pi)``.

    Parameters
    ----------
    r : float
        Coupling parameter (must be positive for convergence).
    b : float
        Sinh-Gordon coupling.  ``b = 1`` is the self-dual point.
    **solver_kwargs
        Forwarded to :class:`TBASolver`.
    """
    if b == 1.0:
        def kernel(x):
            return -1.0 / (2.0 * np.pi * np.cosh(x))
    else:
        q = b + 1.0 / b
        p = b / q
        def kernel(x):
            return -(4.0 * np.sin(np.pi * p) * np.cosh(x)) / (
                2.0 * np.pi * (np.cosh(2 * x) - np.cos(2 * np.pi * p))
            )

    solver = TBASolver(
        forcing=[lambda x, _r=r: _r * np.cosh(x)],
        kernels=[[kernel]],
        crossing=[[0]],
        labels=["y"],
        **solver_kwargs,
    )
    return solver.solve()


def liouville(
    P: Optional[float] = None,
    b: float = 1.0,
    **solver_kwargs,
) -> list[TBASolution]:
    """Solve the Liouville TBA.

    Without *P*, solves the standard Liouville TBA.  With *P*, adds
    boundary conditions that parametrise the solution family.

    Parameters
    ----------
    P : float, optional
        Boundary-condition parameter.
    b : float
        Coupling.  ``b = 1`` is the self-dual point.
    **solver_kwargs
        Forwarded to :class:`TBASolver`.
    """
    gamma_quarter = float(gamma(0.25))
    coeff = 16.0 * np.pi ** 1.5 / gamma_quarter ** 2

    if b == 1.0:
        def kernel(x):
            return -1.0 / (np.pi * np.cosh(x))
    else:
        q = b + 1.0 / b
        p = b / q
        def kernel(x):
            return -(4.0 * np.sin(np.pi * p) * np.cosh(x)) / (
                2.0 * np.pi * (np.cosh(2 * x) - np.cos(2 * np.pi * p))
            )

    bdy_ext = None
    bdy_int = None
    if P is not None:
        def _bdy_ext(x, _P=P):
            return -8.0 * _P * np.log(1.0 + np.exp(-x))
        def _bdy_int(x, _P=P):
            return -4.0 * _P * np.log(1.0 + np.exp(-2.0 * x))
        bdy_ext = [_bdy_ext]
        bdy_int = [[_bdy_int]]

    solver = TBASolver(
        forcing=[lambda x: coeff * np.exp(x)],
        kernels=[[kernel]],
        crossing=[[0]],
        labels=["y"],
        boundary_ext=bdy_ext,
        boundary_int=bdy_int,
        **solver_kwargs,
    )
    return solver.solve()


def seiberg_witten_su2(
    u: float,
    Lambda: float = 1.0,
    **solver_kwargs,
) -> list[TBASolution]:
    """Solve the pure SU(2) Seiberg-Witten (Nekrasov-Shatashvili) TBA.

    Two coupled equations with kernel ``sech(x) / pi``.

    Parameters
    ----------
    u : float
        Coulomb-branch parameter (convergent for ``|u| < Lambda**2``).
    Lambda : float
        Dynamical scale.
    **solver_kwargs
        Forwarded to :class:`TBASolver`.
    """
    ratio = u / Lambda ** 2

    c1 = float(
        -2.0 * np.pi * (-1.0 + ratio)
        * hyp2f1(0.5, 0.5, 2.0, 0.5 * (1.0 - ratio))
    )
    c2 = float(
        -2.0 * np.pi * (-1.0 - ratio)
        * hyp2f1(0.5, 0.5, 2.0, 0.5 * (1.0 + ratio))
    )

    def kernel(x):
        return -1.0 / (np.pi * np.cosh(x))

    solver = TBASolver(
        forcing=[
            lambda x, _c=c1: _c * np.exp(x),
            lambda x, _c=c2: _c * np.exp(x),
        ],
        kernels=[[kernel], [kernel]],
        crossing=[[1], [0]],
        labels=["y1", "y2"],
        **solver_kwargs,
    )
    return solver.solve()
