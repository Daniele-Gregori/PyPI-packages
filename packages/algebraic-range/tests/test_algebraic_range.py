"""
Comprehensive tests for algebraic_range.

Translated from the Wolfram Language .wlt test suite covering all argument
forms, options, edge cases, domain properties, and error handling.
"""

import math

import pytest
import sympy
from sympy import Rational, sqrt, S, E, Integer, Abs

from algebraic_range import (
    algebraic_range,
    formula_complexity,
    AlgebraicRangeError,
    NotRealError,
    NotAlgebraicError,
    StepBoundError,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _nv(x):
    return float(x.evalf())


def _nvals(lst):
    return [_nv(v) for v in lst]


def _is_subset_numeric(small, big, tol=1e-10):
    """Check that every element of *small* appears in *big* (numerically)."""
    big_set = {round(_nv(v), 10) for v in big}
    for v in small:
        if round(_nv(v), 10) not in big_set:
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Single-argument form  algebraic_range(x)
# ═════════════════════════════════════════════════════════════════════════════

class TestSingleArg:
    def test_1a_basic_ar3(self):
        result = algebraic_range(3)
        expected = [Integer(n) ** Rational(1, 2) for n in range(1, 10)]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(_nv(r) - _nv(e)) < 1e-12

    def test_1b_edge_ar1(self):
        assert algebraic_range(1) == [S.One]

    def test_1c_ar2(self):
        result = algebraic_range(2)
        assert len(result) == 4

    def test_1d_ar5_length(self):
        assert len(algebraic_range(5)) == 25

    def test_1e_ar5_endpoints(self):
        r = algebraic_range(5)
        assert _nv(r[0]) == pytest.approx(1.0)
        assert _nv(r[-1]) == pytest.approx(5.0)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Two-argument form  algebraic_range(x, y)
# ═════════════════════════════════════════════════════════════════════════════

class TestTwoArg:
    def test_2a_positive_length(self):
        assert len(algebraic_range(1, 3)) == 9

    def test_2b_zero_start(self):
        assert _nv(algebraic_range(0, 2)[0]) == pytest.approx(0.0)

    def test_2c_zero_start_last(self):
        assert _nv(algebraic_range(0, 2)[-1]) == pytest.approx(2.0)

    def test_2d_negative_reflection(self):
        lhs = algebraic_range(-3, -1)
        rhs_raw = algebraic_range(1, 3)
        rhs = list(reversed([-v for v in rhs_raw]))
        assert len(lhs) == len(rhs)
        for a, b in zip(lhs, rhs):
            assert abs(_nv(a) - _nv(b)) < 1e-12

    def test_2e_extends_range(self):
        ar = algebraic_range(2, 5)
        ar_nums = {round(_nv(v), 10) for v in ar}
        for n in range(2, 6):
            assert round(float(n), 10) in ar_nums

    def test_2f_single_point(self):
        r = algebraic_range(3, 3)
        assert len(r) == 1
        assert _nv(r[0]) == pytest.approx(3.0)

    def test_2g_zero_zero(self):
        r = algebraic_range(0, 0)
        assert len(r) == 1
        assert _nv(r[0]) == pytest.approx(0.0)

    def test_2h_mixed_bounds_contains_zero(self):
        r = algebraic_range(-2, 3)
        assert any(abs(_nv(v)) < 1e-12 for v in r)

    def test_2i_neg_to_zero(self):
        r = algebraic_range(-3, 0)
        assert _nv(r[-1]) == pytest.approx(0.0)

    def test_2j_irrational_bounds(self):
        r = algebraic_range(sqrt(2), sqrt(7))
        assert len(r) == 6

    def test_2k_negative_reflection_large(self):
        lhs = algebraic_range(-5, -2)
        rhs = list(reversed([-v for v in algebraic_range(2, 5)]))
        assert len(lhs) == len(rhs)
        for a, b in zip(lhs, rhs):
            assert abs(_nv(a) - _nv(b)) < 1e-12

    def test_2l_all_negative(self):
        r = algebraic_range(-4, -1)
        assert all(_nv(v) < 0 for v in r)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Three-argument form  algebraic_range(x, y, s)
# ═════════════════════════════════════════════════════════════════════════════

class TestThreeArg:
    def test_3a_fractional_step_nonempty(self):
        assert len(algebraic_range(0, 3, Rational(1, 2))) > 0

    def test_3b_step_upper_bound(self):
        r = algebraic_range(0, 3, Rational(1, 2))
        diffs = [_nv(r[i + 1]) - _nv(r[i]) for i in range(len(r) - 1)]
        assert max(diffs) <= 0.5 + 1e-10

    def test_3c_negative_step_descending(self):
        r = algebraic_range(2, -2, Rational(-1, 2))
        vals = _nvals(r)
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-10

    def test_3d_irrational_step(self):
        assert len(algebraic_range(0, 3, 1 / sqrt(2))) > 0

    def test_3e_nested_sqrt_bounds(self):
        assert len(algebraic_range(sqrt(1 + sqrt(2)), 4, 1 / sqrt(2))) > 0

    def test_3f_negative_irrational_step(self):
        assert len(algebraic_range(3, Rational(3, 2), -1 / sqrt(3))) > 0

    def test_3g_step_equals_span(self):
        r = algebraic_range(0, 2, 2)
        vals = _nvals(r)
        assert pytest.approx(vals[0], abs=1e-10) == 0.0
        assert pytest.approx(vals[-1], abs=1e-10) == 2.0

    def test_3h_step_larger_than_span(self):
        r = algebraic_range(0, 1, 3)
        assert len(r) >= 1
        assert _nv(r[0]) == pytest.approx(0.0)

    def test_3i_first_element(self):
        r = algebraic_range(0, 3, Rational(1, 2))
        assert _nv(r[0]) == pytest.approx(0.0)

    def test_3j_last_element(self):
        r = algebraic_range(0, 3, Rational(1, 2))
        assert _nv(r[-1]) == pytest.approx(3.0)

    def test_3k_very_small_step(self):
        assert len(algebraic_range(0, 1, Rational(1, 10))) > 0


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 4 — Four-argument form  algebraic_range(x, y, s, d)
# ═════════════════════════════════════════════════════════════════════════════

class TestFourArg:
    def test_4a_step_bounds_verified(self):
        r = algebraic_range(2, 7, Rational(1, 3), Rational(1, 4))
        diffs = [abs(_nv(r[i + 1]) - _nv(r[i])) for i in range(len(r) - 1)]
        assert min(diffs) >= 0.25 - 1e-10
        assert max(diffs) <= 1 / 3 + 1e-10

    def test_4b_nonempty(self):
        assert len(algebraic_range(2, 7, Rational(1, 3), Rational(1, 4))) > 0

    def test_4c_with_root_order(self):
        r = algebraic_range(
            0, 4, 2, Rational(1, 4), root_order=[4]
        )
        assert len(r) > 0

    def test_4d_step_bounds_with_root_order(self):
        r = algebraic_range(
            1, 5, Rational(1, 2), Rational(1, 5), root_order=[3]
        )
        diffs = [abs(_nv(r[i + 1]) - _nv(r[i])) for i in range(len(r) - 1)]
        assert min(diffs) >= 0.2 - 1e-10
        assert max(diffs) <= 0.5 + 1e-10


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 5 — Option root_order
# ═════════════════════════════════════════════════════════════════════════════

class TestRootOrder:
    def test_5a_default_equals_order2(self):
        a = algebraic_range(2)
        b = algebraic_range(2, root_order=[2])
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert abs(_nv(x) - _nv(y)) < 1e-12

    def test_5b_higher_order_more_elements(self):
        assert len(algebraic_range(2, root_order=3)) > len(algebraic_range(2))

    def test_5c_upto_contains_only(self):
        full = algebraic_range(2, root_order=3)
        only3 = algebraic_range(2, root_order=[3])
        assert _is_subset_numeric(only3, full)

    def test_5d_upto_contains_order2(self):
        full = algebraic_range(2, root_order=3)
        only2 = algebraic_range(2, root_order=[2])
        assert _is_subset_numeric(only2, full)

    def test_5e_only_cubic_roots(self):
        r = algebraic_range(2, root_order=[3])
        assert len(r) == 8  # 1, 2^(1/3), 3^(1/3), ..., 2

    def test_5f_multiple_orders(self):
        assert len(algebraic_range(1, Rational(3, 2), root_order=[3, 5])) > 0

    def test_5g_fourth_roots_only(self):
        assert len(algebraic_range(0, 4, 2, root_order=[4])) > 0

    def test_5h_order4_more_than_order2_with_step(self):
        l4 = len(algebraic_range(1, 4, 1, root_order=4))
        l2 = len(algebraic_range(1, 4, 1, root_order=2))
        assert l4 > l2


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 6 — Option step_method
# ═════════════════════════════════════════════════════════════════════════════

class TestStepMethod:
    def test_6a_root_nonempty(self):
        r = algebraic_range(0, 3, Rational(1, 3), step_method="Root")
        assert len(r) > 0

    def test_6b_outer_subset_of_root(self):
        root = algebraic_range(0, 3, Rational(1, 3), step_method="Root")
        outer = algebraic_range(0, 3, Rational(1, 3), step_method="Outer")
        assert _is_subset_numeric(outer, root)

    def test_6c_root_at_least_as_large(self):
        lr = len(algebraic_range(0, 3, Rational(1, 3), step_method="Root"))
        lo = len(algebraic_range(0, 3, Rational(1, 3), step_method="Outer"))
        assert lr >= lo


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 7 — Option farey_range
# ═════════════════════════════════════════════════════════════════════════════

class TestFareyRange:
    def test_7a_nonempty(self):
        r = algebraic_range(0, 3, Rational(1, 3), farey_range=True)
        assert len(r) > 0

    def test_7b_at_least_as_large_as_default(self):
        lf = len(algebraic_range(0, 3, Rational(1, 3), farey_range=True))
        ld = len(algebraic_range(0, 3, Rational(1, 3)))
        assert lf >= ld


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 8 — Option formula_complexity_threshold
# ═════════════════════════════════════════════════════════════════════════════

class TestFormulaComplexity:
    def test_8a_lower_fewer(self):
        l4 = len(algebraic_range(4, root_order=4, formula_complexity_threshold=4))
        l8 = len(algebraic_range(4, root_order=4, formula_complexity_threshold=8))
        assert l4 < l8

    def test_8b_finite_less_than_infinity(self):
        lf = len(algebraic_range(4, root_order=4, formula_complexity_threshold=8))
        li = len(algebraic_range(4, root_order=4))
        assert lf < li

    def test_8c_monotone(self):
        l4 = len(algebraic_range(4, root_order=4, formula_complexity_threshold=4))
        l8 = len(algebraic_range(4, root_order=4, formula_complexity_threshold=8))
        li = len(algebraic_range(4, root_order=4))
        assert l4 < l8 < li


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 9 — Option algebraics_only
# ═════════════════════════════════════════════════════════════════════════════

class TestAlgebraicsOnly:
    def test_9a_rejects_transcendental(self):
        with pytest.raises(NotAlgebraicError):
            algebraic_range(0, 5, sqrt(E))

    def test_9b_accepts_with_option(self):
        r = algebraic_range(0, 5, sqrt(E), algebraics_only=False)
        assert len(r) > 0


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 10 — Domain verification
# ═════════════════════════════════════════════════════════════════════════════

class TestDomain:
    def test_10a_all_real(self):
        r = algebraic_range(0, 5, Rational(1, 2), root_order=3)
        for v in r:
            nv = complex(v.evalf())
            assert abs(nv.imag) < 1e-12

    def test_10b_single_arg_algebraic(self):
        r = algebraic_range(3)
        for v in r:
            # All elements are radicals of integers → algebraic
            assert v.is_algebraic is not False


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 11 — Sorting & uniqueness
# ═════════════════════════════════════════════════════════════════════════════

class TestSortingUniqueness:
    def test_11a_sorted_ascending_positive(self):
        vals = _nvals(algebraic_range(1, 5))
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-12

    def test_11b_sorted_ascending_negative(self):
        vals = _nvals(algebraic_range(-5, -1))
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-12

    def test_11c_sorted_ascending_mixed(self):
        vals = _nvals(algebraic_range(-2, 3))
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-12

    def test_11d_no_duplicates_three_arg(self):
        r = algebraic_range(0, 3, Rational(1, 3))
        rounded = [round(_nv(v), 10) for v in r]
        assert len(rounded) == len(set(rounded))

    def test_11e_no_duplicates_single_arg(self):
        r = algebraic_range(5)
        rounded = [round(_nv(v), 10) for v in r]
        assert len(rounded) == len(set(rounded))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 12 — Negative ranges & reflection
# ═════════════════════════════════════════════════════════════════════════════

class TestNegativeReflection:
    def test_12a_reflection_small(self):
        lhs = algebraic_range(-3, -1)
        rhs = list(reversed([-v for v in algebraic_range(1, 3)]))
        assert len(lhs) == len(rhs)
        for a, b in zip(lhs, rhs):
            assert abs(_nv(a) - _nv(b)) < 1e-12

    def test_12b_reflection_large(self):
        lhs = algebraic_range(-5, -2)
        rhs = list(reversed([-v for v in algebraic_range(2, 5)]))
        assert len(lhs) == len(rhs)

    def test_12c_reflection_larger(self):
        lhs = algebraic_range(-10, -7)
        rhs = list(reversed([-v for v in algebraic_range(7, 10)]))
        assert len(lhs) == len(rhs)

    def test_12d_neg_step_asymmetry(self):
        """Negative step produces a valid descending range."""
        a = algebraic_range(Rational(3, 2), Rational(-3, 2), Rational(-1, 3))
        b = algebraic_range(Rational(-3, 2), Rational(3, 2), Rational(1, 3))
        # Both should be non-empty and have the same length
        assert len(a) > 0
        assert len(b) > 0
        # a is descending, b is ascending
        a_vals = _nvals(a)
        b_vals = _nvals(b)
        assert a_vals[0] > a_vals[-1]
        assert b_vals[0] < b_vals[-1]

    def test_12e_neg_step_descending(self):
        r = algebraic_range(Rational(3, 2), Rational(-3, 2), Rational(-1, 3))
        vals = _nvals(r)
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-10

    def test_12f_neg_step_irrational_bounds(self):
        r = algebraic_range(sqrt(10), -sqrt(10), -2)
        assert len(r) > 0

    def test_12g_neg_step_complex_irrational_bounds(self):
        r = algebraic_range(
            2 * sqrt(2 + sqrt(3)), -2 * sqrt(2 + sqrt(3)), -sqrt(5)
        )
        assert len(r) > 0


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 13 — Error handling
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_13a_complex_input(self):
        with pytest.raises(NotRealError):
            algebraic_range(sqrt(1 - sqrt(2)), 4, 1 / sqrt(2))

    def test_13b_transcendental_input(self):
        with pytest.raises(NotAlgebraicError):
            algebraic_range(0, 5, sqrt(E))

    def test_13c_step_bound_error(self):
        with pytest.raises(StepBoundError):
            algebraic_range(0, 5, Rational(1, 4), Rational(1, 2))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 14 — Numeric (approximate) inputs
# ═════════════════════════════════════════════════════════════════════════════

class TestNumericInputs:
    def test_14a_simple_decimal(self):
        r = algebraic_range(0.1, 3.1)
        assert len(r) > 0

    def test_14b_decimal_recognized(self):
        r = algebraic_range(0.1, 3.1)
        # First element is sqrt(1) = 1 (smallest integer root >= 0.1)
        assert _nv(r[0]) == pytest.approx(1.0, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 15 — Superset / subset structural properties
# ═════════════════════════════════════════════════════════════════════════════

class TestStructuralProperties:
    def test_15a_superset_small(self):
        ar = algebraic_range(2, 5)
        assert _is_subset_numeric([Integer(n) for n in range(2, 6)], ar)

    def test_15b_superset_with_zero(self):
        ar = algebraic_range(0, 4)
        assert _is_subset_numeric([Integer(n) for n in range(0, 5)], ar)

    def test_15c_superset_large(self):
        ar = algebraic_range(1, 10)
        assert _is_subset_numeric([Integer(n) for n in range(1, 11)], ar)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 16 — Combined options
# ═════════════════════════════════════════════════════════════════════════════

class TestCombinedOptions:
    def test_16a_root_order_formula_complexity(self):
        r = algebraic_range(0, 3, root_order=4, formula_complexity_threshold=6)
        assert len(r) > 0

    def test_16b_root_order_step_method(self):
        r = algebraic_range(
            0, 3, Rational(1, 3), root_order=3, step_method="Root"
        )
        assert len(r) > 0

    def test_16c_root_order_farey_range(self):
        r = algebraic_range(
            0, 2, Rational(1, 2), root_order=3, farey_range=True
        )
        assert len(r) > 0

    def test_16d_four_args_root_order_bounds(self):
        r = algebraic_range(
            1, 5, Rational(1, 2), Rational(1, 5), root_order=[3]
        )
        diffs = [abs(_nv(r[i + 1]) - _nv(r[i])) for i in range(len(r) - 1)]
        assert min(diffs) >= 0.2 - 1e-10
        assert max(diffs) <= 0.5 + 1e-10


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 17 — Stress / large output
# ═════════════════════════════════════════════════════════════════════════════

class TestStress:
    def test_17a_large_range_length(self):
        assert len(algebraic_range(0, 10)) == 101

    def test_17b_large_range_endpoints(self):
        r = algebraic_range(0, 10)
        assert _nv(r[0]) == pytest.approx(0.0)
        assert _nv(r[-1]) == pytest.approx(10.0)

    def test_17c_high_root_order_many_elements(self):
        assert len(algebraic_range(1, 4, 1, root_order=4)) > 100


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 18 — formula_complexity function
# ═════════════════════════════════════════════════════════════════════════════

class TestFormulaComplexityFunction:
    def test_18a_integer_complexity(self):
        assert formula_complexity(Integer(1)) > 0

    def test_18b_simple_sqrt(self):
        c = formula_complexity(sqrt(2))
        assert c > 0

    def test_18c_higher_root_more_complex(self):
        c2 = formula_complexity(sqrt(2))
        c4 = formula_complexity(Integer(2) ** Rational(1, 4))
        assert c4 >= c2

    def test_18d_larger_integers_more_complex(self):
        c1 = formula_complexity(sqrt(2))
        c2 = formula_complexity(sqrt(1000))
        assert c2 > c1
