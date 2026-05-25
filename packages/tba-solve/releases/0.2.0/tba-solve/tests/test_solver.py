"""Tests for tba-solve.

Validates the TBA solver against known structural properties of TBA
solutions and self-consistency checks.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from tba_solve import TBASolver, TBASolution
from tba_solve.models import sinh_gordon, liouville, seiberg_witten_su2


# ===================================================================
# Helpers
# ===================================================================

FAST_OPTS = dict(
    grid_cutoff=20.0,
    grid_resolution=256,
    stopping_accuracy=1e-6,
    max_iterations=2000,
)

MEDIUM_OPTS = dict(
    grid_cutoff=40.0,
    grid_resolution=512,
    stopping_accuracy=1e-8,
    max_iterations=4000,
)


def _assert_real(sol: TBASolution, tol: float = 1e-8):
    """Check that the solution is real-valued on the interior grid."""
    x = np.linspace(sol.domain[0] * 0.9, sol.domain[1] * 0.9, 200)
    vals = sol(x)
    assert np.max(np.abs(np.imag(vals))) < tol, (
        f"Imaginary part too large: {np.max(np.abs(np.imag(vals))):.2e}"
    )


def _assert_symmetric(sol: TBASolution, tol: float = 1e-6):
    """Check y(x) ≈ y(-x) on the interior grid."""
    x = np.linspace(0.1, sol.domain[1] * 0.5, 100)
    diff = np.max(np.abs(sol(x) - sol(-x)))
    assert diff < tol, f"Symmetry violation: {diff:.2e}"


# ===================================================================
# 1. TBASolver validation
# ===================================================================

class TestValidation:
    def test_mismatched_kernels(self):
        with pytest.raises(ValueError, match="kernel lists"):
            TBASolver(
                forcing=[lambda x: x],
                kernels=[],
                crossing=[[0]],
            )

    def test_mismatched_crossing(self):
        with pytest.raises(ValueError, match="crossing"):
            TBASolver(
                forcing=[lambda x: x],
                kernels=[[lambda x: x]],
                crossing=[[0, 1]],
            )

    def test_crossing_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            TBASolver(
                forcing=[lambda x: x],
                kernels=[[lambda x: x]],
                crossing=[[5]],
            )

    def test_mismatched_constants(self):
        with pytest.raises(ValueError, match="constants"):
            TBASolver(
                forcing=[lambda x: x],
                kernels=[[lambda x: x]],
                crossing=[[0]],
                constants=[[1, 2]],
            )

    def test_mismatched_signs(self):
        with pytest.raises(ValueError, match="signs"):
            TBASolver(
                forcing=[lambda x: x],
                kernels=[[lambda x: x]],
                crossing=[[0]],
                signs=[[-1, 1]],
            )


# ===================================================================
# 2. TBASolution basics
# ===================================================================

class TestTBASolution:
    def test_callable(self):
        x = np.linspace(-5, 5, 100)
        sol = TBASolution(x, np.sin(x), label="test")
        assert np.abs(sol(0.0)) < 1e-10

    def test_repr(self):
        x = np.linspace(-1, 1, 10)
        sol = TBASolution(x, x ** 2, label="y")
        assert "'y'" in repr(sol)
        assert "TBASolution" in repr(sol)

    def test_domain(self):
        x = np.linspace(-3, 3, 50)
        sol = TBASolution(x, np.ones_like(x))
        assert sol.domain == (-3.0, 3.0)

    def test_complex_values(self):
        x = np.linspace(-1, 1, 50)
        vals = np.exp(1j * x)
        sol = TBASolution(x, vals)
        result = sol(0.0)
        assert np.abs(result - 1.0) < 1e-7


# ===================================================================
# 3. Sinh-Gordon TBA
# ===================================================================

class TestSinhGordon:
    def test_convergence(self):
        sols = sinh_gordon(r=0.1, **FAST_OPTS)
        assert len(sols) == 1
        assert isinstance(sols[0], TBASolution)

    def test_real_valued(self):
        sols = sinh_gordon(r=0.1, **FAST_OPTS)
        _assert_real(sols[0])

    def test_symmetric(self):
        sols = sinh_gordon(r=0.1, **FAST_OPTS)
        _assert_symmetric(sols[0])

    def test_positive_for_positive_r(self):
        sols = sinh_gordon(r=1.0, **FAST_OPTS)
        x = np.linspace(-5, 5, 100)
        vals = np.real(sols[0](x))
        assert np.all(vals > 0), "Solution should be positive for r > 0"

    def test_different_r_gives_different_solution(self):
        sol1 = sinh_gordon(r=0.1, **FAST_OPTS)
        sol2 = sinh_gordon(r=1.0, **FAST_OPTS)
        diff = np.abs(sol1[0](0.0) - sol2[0](0.0))
        assert diff > 0.01

    def test_self_consistency(self):
        """Plug solution back into the integral equation at x=0."""
        r = 0.5
        sols = sinh_gordon(r=r, **MEDIUM_OPTS)
        y = sols[0]
        lo, hi = y.domain

        lhs = float(np.real(y(0.0)))

        def integrand(t):
            yt = float(np.real(y(t)))
            return (1.0 / (2.0 * np.pi * np.cosh(t))) * np.log(
                1.0 + np.exp(-yt)
            )

        integral, _ = quad(integrand, lo, hi, limit=200)
        rhs = r * np.cosh(0.0) - integral

        assert abs(lhs - rhs) < 1e-3, (
            f"Self-consistency: LHS={lhs:.8f}, RHS={rhs:.8f}, "
            f"diff={abs(lhs - rhs):.2e}"
        )

    def test_label(self):
        sols = sinh_gordon(r=0.1, **FAST_OPTS)
        assert sols[0].label == "y"


# ===================================================================
# 4. Seiberg-Witten SU(2) TBA
# ===================================================================

class TestSeiberWittenSU2:
    def test_convergence(self):
        sols = seiberg_witten_su2(u=0.1, Lambda=1.0, **FAST_OPTS)
        assert len(sols) == 2
        assert sols[0].label == "y1"
        assert sols[1].label == "y2"

    def test_real_valued(self):
        sols = seiberg_witten_su2(u=0.1, Lambda=1.0, **FAST_OPTS)
        _assert_real(sols[0])
        _assert_real(sols[1])

    def test_u_zero_symmetry(self):
        """When u=0, forcing terms are equal so y1 = y2 by symmetry."""
        sols = seiberg_witten_su2(u=0.0, Lambda=1.0, **FAST_OPTS)
        x = np.linspace(-5, 5, 100)
        diff = np.max(np.abs(sols[0](x) - sols[1](x)))
        assert diff < 1e-4, f"y1 ≠ y2 at u=0: max diff = {diff:.2e}"

    def test_overflow_for_large_u(self):
        with pytest.raises(RuntimeError, match="non-convergent"):
            seiberg_witten_su2(
                u=5.0,
                Lambda=1.0,
                grid_cutoff=20.0,
                grid_resolution=256,
                max_iterations=200,
            )


# ===================================================================
# 5. Liouville TBA
# ===================================================================

class TestLiouville:
    def test_convergence_no_boundary(self):
        sols = liouville(**FAST_OPTS)
        assert len(sols) == 1

    def test_convergence_with_boundary(self):
        sols = liouville(P=1.0, **FAST_OPTS)
        assert len(sols) == 1

    def test_real_valued(self):
        sols = liouville(**FAST_OPTS)
        _assert_real(sols[0])

    def test_boundary_changes_solution(self):
        sol_no_bdy = liouville(**FAST_OPTS)
        sol_bdy = liouville(P=1.0, **FAST_OPTS)
        diff = abs(sol_no_bdy[0](0.0) - sol_bdy[0](0.0))
        assert diff > 0.1, "Boundary condition should change the solution"


# ===================================================================
# 6. Grid options
# ===================================================================

class TestGridOptions:
    def test_cutoff_affects_domain(self):
        sol1 = sinh_gordon(r=0.1, grid_cutoff=10.0, grid_resolution=128,
                           stopping_accuracy=1e-4, max_iterations=500)
        sol2 = sinh_gordon(r=0.1, grid_cutoff=30.0, grid_resolution=128,
                           stopping_accuracy=1e-4, max_iterations=500)
        assert abs(sol1[0].domain[0] - (-10.0)) < 0.5
        assert abs(sol2[0].domain[0] - (-30.0)) < 0.5

    def test_higher_resolution_same_result(self):
        sol_lo = sinh_gordon(
            r=0.5,
            grid_cutoff=15.0,
            grid_resolution=128,
            stopping_accuracy=1e-6,
            max_iterations=2000,
        )
        sol_hi = sinh_gordon(
            r=0.5,
            grid_cutoff=15.0,
            grid_resolution=512,
            stopping_accuracy=1e-6,
            max_iterations=2000,
        )
        x_test = np.linspace(-3, 3, 20)
        diff = np.max(np.abs(sol_lo[0](x_test) - sol_hi[0](x_test)))
        assert diff < 0.01, f"Low/high res differ by {diff:.2e}"


# ===================================================================
# 7. Monitor option
# ===================================================================

class TestMonitor:
    def test_monitor_prints(self, capsys):
        sinh_gordon(r=0.1, monitor=True, **FAST_OPTS)
        captured = capsys.readouterr()
        assert "Iteration" in captured.out or "Total" in captured.out


# ===================================================================
# 8. Custom TBASolver
# ===================================================================

class TestCustomSolver:
    def test_manual_sinh_gordon(self):
        """Build the Sinh-Gordon TBA manually and compare with model."""
        r = 0.3
        sol_model = sinh_gordon(r=r, **FAST_OPTS)

        solver = TBASolver(
            forcing=[lambda x, _r=r: _r * np.cosh(x)],
            kernels=[[lambda x: -1.0 / (2.0 * np.pi * np.cosh(x))]],
            crossing=[[0]],
            **FAST_OPTS,
        )
        sol_manual = solver.solve()

        x_test = np.linspace(-5, 5, 50)
        diff = np.max(np.abs(sol_model[0](x_test) - sol_manual[0](x_test)))
        assert diff < 1e-8, f"Model vs manual differ by {diff:.2e}"

    def test_constant_forcing(self):
        """TBA with constant forcing and weak kernel should converge."""
        solver = TBASolver(
            forcing=[lambda x: np.full_like(x, 5.0)],
            kernels=[[lambda x: -0.01 / np.cosh(x)]],
            crossing=[[0]],
            grid_cutoff=10.0,
            grid_resolution=128,
            stopping_accuracy=1e-6,
            max_iterations=1000,
        )
        sols = solver.solve()
        x = np.linspace(-3, 3, 20)
        vals = np.real(sols[0](x))
        assert np.all(vals > 3.0), "Solution should be near forcing value"
