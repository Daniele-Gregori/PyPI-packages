"""Tests for the farey package.

Expected values are the outputs of the improved Wolfram Language
``FareyRange`` resource function (exact rationals with denominator <= n
inside the interval).  The endpoint-symmetric extension (bounds in either
order, sign of step = direction) is validated here as well.
"""

import pytest
from fractions import Fraction

from farey import farey_sequence, farey_range, FareyError


# ===========================================================================
# FareySequence
# ===========================================================================

class TestFareySequence:
    def test_order_1(self):
        assert farey_sequence(1) == [Fraction(0, 1), Fraction(1, 1)]

    def test_order_2(self):
        assert farey_sequence(2) == [
            Fraction(0, 1), Fraction(1, 2), Fraction(1, 1),
        ]

    def test_order_3(self):
        assert farey_sequence(3) == [
            Fraction(0, 1), Fraction(1, 3), Fraction(1, 2),
            Fraction(2, 3), Fraction(1, 1),
        ]

    def test_order_4(self):
        assert farey_sequence(4) == [
            Fraction(0, 1), Fraction(1, 4), Fraction(1, 3),
            Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
            Fraction(1, 1),
        ]

    def test_order_5(self):
        assert farey_sequence(5) == [
            Fraction(0, 1), Fraction(1, 5), Fraction(1, 4),
            Fraction(1, 3), Fraction(2, 5), Fraction(1, 2),
            Fraction(3, 5), Fraction(2, 3), Fraction(3, 4),
            Fraction(4, 5), Fraction(1, 1),
        ]

    def test_order_7(self):
        assert farey_sequence(7) == [
            Fraction(0, 1), Fraction(1, 7), Fraction(1, 6),
            Fraction(1, 5), Fraction(1, 4), Fraction(2, 7),
            Fraction(1, 3), Fraction(2, 5), Fraction(3, 7),
            Fraction(1, 2), Fraction(4, 7), Fraction(3, 5),
            Fraction(2, 3), Fraction(5, 7), Fraction(3, 4),
            Fraction(4, 5), Fraction(5, 6), Fraction(6, 7),
            Fraction(1, 1),
        ]

    def test_lengths(self):
        # |F_n| = 1 + sum_{k=1}^{n} phi(k)
        assert len(farey_sequence(5)) == 11
        assert len(farey_sequence(6)) == 13
        assert len(farey_sequence(7)) == 19
        assert len(farey_sequence(8)) == 23

    def test_sorted_and_reduced(self):
        from math import gcd
        seq = farey_sequence(8)
        for a, b in zip(seq, seq[1:]):
            assert a < b
        for f in seq:
            assert gcd(f.numerator, f.denominator) == 1

    def test_mediant_property(self):
        """Consecutive Farey fractions a/b, c/d satisfy |bc - ad| = 1."""
        seq = farey_sequence(6)
        for i in range(len(seq) - 1):
            a, b = seq[i].numerator, seq[i].denominator
            c, d = seq[i + 1].numerator, seq[i + 1].denominator
            assert abs(b * c - a * d) == 1

    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            farey_sequence(0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            farey_sequence(-3)

    def test_invalid_float(self):
        with pytest.raises(ValueError):
            farey_sequence(2.5)

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            farey_sequence("3")


# ===========================================================================
# FareyRange — values verified against the WL FareyRange resource function
# ===========================================================================

class TestFareyRangeExact:
    def test_unit_interval_order_3(self):
        assert farey_range(0, 1, 3) == farey_sequence(3)

    def test_default_step_order_1(self):
        assert farey_range(0, 1) == [Fraction(0, 1), Fraction(1, 1)]
        assert farey_range(0, 3) == [Fraction(i) for i in range(4)]

    def test_0_2_order_4(self):
        assert farey_range(0, 2, 4) == [
            Fraction(0, 1), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
            Fraction(2, 3), Fraction(3, 4), Fraction(1, 1), Fraction(5, 4),
            Fraction(4, 3), Fraction(3, 2), Fraction(5, 3), Fraction(7, 4),
            Fraction(2, 1),
        ]

    def test_neg1_1_order_5(self):
        assert farey_range(-1, 1, 5) == [
            Fraction(-1, 1), Fraction(-4, 5), Fraction(-3, 4), Fraction(-2, 3),
            Fraction(-3, 5), Fraction(-1, 2), Fraction(-2, 5), Fraction(-1, 3),
            Fraction(-1, 4), Fraction(-1, 5), Fraction(0, 1), Fraction(1, 5),
            Fraction(1, 4), Fraction(1, 3), Fraction(2, 5), Fraction(1, 2),
            Fraction(3, 5), Fraction(2, 3), Fraction(3, 4), Fraction(4, 5),
            Fraction(1, 1),
        ]

    def test_2_5_order_5_first_unit(self):
        result = farey_range(2, 5, 5)
        assert len(result) == 31
        assert result[:11] == [
            Fraction(2, 1), Fraction(11, 5), Fraction(9, 4), Fraction(7, 3),
            Fraction(12, 5), Fraction(5, 2), Fraction(13, 5), Fraction(8, 3),
            Fraction(11, 4), Fraction(14, 5), Fraction(3, 1),
        ]

    def test_neg3_3_order_4_symmetry(self):
        result = farey_range(-3, 3, 4)
        assert len(result) == 37
        assert result[0] == Fraction(-3) and result[-1] == Fraction(3)
        for i in range(len(result)):
            assert result[i] + result[-(i + 1)] == 0

    def test_10_13_order_6(self):
        result = farey_range(10, 13, 6)
        assert len(result) == 37
        assert result[0] == Fraction(10) and result[-1] == Fraction(13)

    def test_noninteger_upper_bound(self):
        # denominators <= 2 inside [0, 3/2]
        assert farey_range(0, Fraction(3, 2), 2) == [
            Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
        ]

    def test_noninteger_both_bounds(self):
        # denominators <= 3 inside [1/2, 5/2]
        assert farey_range(Fraction(1, 2), Fraction(5, 2), 3) == [
            Fraction(1, 2), Fraction(2, 3), Fraction(1), Fraction(4, 3),
            Fraction(3, 2), Fraction(5, 3), Fraction(2), Fraction(7, 3),
            Fraction(5, 2),
        ]

    def test_fractional_lower_upper(self):
        # denominators <= 5 inside [0, 7/10]
        assert farey_range(0, Fraction(7, 10), 5) == [
            Fraction(0), Fraction(1, 5), Fraction(1, 4), Fraction(1, 3),
            Fraction(2, 5), Fraction(1, 2), Fraction(3, 5), Fraction(2, 3),
        ]

    def test_result_is_all_fractions(self):
        assert all(isinstance(v, Fraction) for v in farey_range(0, Fraction(7, 4), 4))


class TestFareyRangeStepConventions:
    def test_int_and_unit_fraction_agree(self):
        assert farey_range(0, 2, 3) == farey_range(0, 2, Fraction(1, 3))
        assert farey_range(-1, 1, 5) == farey_range(-1, 1, Fraction(1, 5))
        assert farey_range(0, 3, 7) == farey_range(0, 3, Fraction(1, 7))

    def test_negative_int_and_unit_fraction_agree(self):
        # descending: first bound larger, negative step
        assert farey_range(2, 0, -3) == farey_range(2, 0, Fraction(-1, 3))
        assert farey_range(1, -1, -5) == farey_range(1, -1, Fraction(-1, 5))

    def test_magnitude_is_only_order(self):
        assert farey_range(2, 4, 3) == farey_range(2, 4, Fraction(1, 3))
        assert farey_range(4, 2, -3) == farey_range(4, 2, Fraction(-1, 3))

    def test_reciprocal_step_one_sided_and_degenerate(self):
        assert farey_range(0, Fraction(1, 4), 3) == farey_range(0, Fraction(1, 4), Fraction(1, 3))
        assert farey_range(Fraction(1, 4), 0, -3) == farey_range(Fraction(1, 4), 0, Fraction(-1, 3))
        assert farey_range(5, 5, 3) == farey_range(5, 5, Fraction(1, 3))
        assert farey_range(5, 5, -3) == farey_range(5, 5, Fraction(-1, 3))
        assert farey_range(Fraction(2, 5), Fraction(2, 5), 3) == farey_range(Fraction(2, 5), Fraction(2, 5), Fraction(1, 3))

    def test_float_steps(self):
        assert farey_range(0, 2, 4.0) == farey_range(0, 2, 4)
        assert farey_range(0, 2, 1 / 3) == farey_range(0, 2, 3)
        assert farey_range(-1, 1, 0.2) == farey_range(-1, 1, 5)
        assert farey_range(4, 2, -1 / 3) == farey_range(4, 2, -3)


class TestFareyRangeDirection:
    """Directional, like Range[a, b, step]: start at the first bound, move in
    the direction of the step; empty when the bounds run the other way."""

    def test_positive_step_ascending(self):
        assert farey_range(0, 1, 3) == [
            Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1),
        ]

    def test_negative_step_descending(self):
        assert farey_range(1, 0, -3) == [
            Fraction(1), Fraction(2, 3), Fraction(1, 2), Fraction(1, 3), Fraction(0),
        ]

    def test_descending_is_reverse_of_ascending(self):
        assert farey_range(1, 0, -3) == list(reversed(farey_range(0, 1, 3)))
        assert farey_range(4, 2, -3) == list(reversed(farey_range(2, 4, 3)))

    def test_bounds_against_positive_step_are_empty(self):
        assert farey_range(1, 0, 3) == []
        assert farey_range(4, 2, 3) == []
        assert farey_range(3, Fraction(3, 2), 3) == []

    def test_bounds_against_negative_step_are_empty(self):
        assert farey_range(0, 1, -3) == []
        assert farey_range(2, 4, -3) == []

    def test_works_for_bounds_greater_than_one(self):
        assert farey_range(2, 4, 3) == [
            Fraction(2), Fraction(7, 3), Fraction(5, 2), Fraction(8, 3),
            Fraction(3), Fraction(10, 3), Fraction(7, 2), Fraction(11, 3),
            Fraction(4),
        ]
        assert farey_range(Fraction(3, 2), 3, 3) == [
            Fraction(3, 2), Fraction(5, 3), Fraction(2), Fraction(7, 3),
            Fraction(5, 2), Fraction(8, 3), Fraction(3),
        ]


class TestFareyRangeOneSidedAndEmpty:
    def test_one_sided_single_point(self):
        # interval too short for a second rational of order 3: only 0 fits
        assert farey_range(0, Fraction(1, 4), 3) == [Fraction(0)]

    def test_one_sided_descending(self):
        assert farey_range(Fraction(1, 4), 0, -3) == [Fraction(0)]

    def test_one_sided_wrong_direction_is_empty(self):
        assert farey_range(Fraction(1, 4), 0, 3) == []
        assert farey_range(0, Fraction(1, 4), -3) == []

    def test_degenerate_on_grid_returns_point(self):
        # x == y yields {x} when x lies on the order-n Farey grid (den(x) <= n),
        # the canonical convention shared with algebraic-range.
        assert farey_range(1, 1, 3) == [Fraction(1)]
        assert farey_range(7, 7, 5) == [Fraction(7)]
        assert farey_range(7, 7, -5) == [Fraction(7)]
        assert farey_range(Fraction(1, 2), Fraction(1, 2), 3) == [Fraction(1, 2)]
        assert farey_range(Fraction(-4, 3), Fraction(-4, 3), 9) == [Fraction(-4, 3)]

    def test_degenerate_off_grid_is_empty(self):
        assert farey_range(Fraction(2, 5), Fraction(2, 5), 3) == []
        assert farey_range(Fraction(2, 5), Fraction(2, 5), -3) == []
        assert farey_range(Fraction(7, 3), Fraction(7, 3), 2) == []


class TestFareyRangeErrors:
    def test_step_zero_int(self):
        with pytest.raises(FareyError):
            farey_range(0, 1, 0)

    def test_step_zero_fraction(self):
        with pytest.raises(FareyError):
            farey_range(0, 1, Fraction(0))

    def test_step_bad_fraction(self):
        with pytest.raises(FareyError):
            farey_range(0, 1, Fraction(2, 3))

    def test_step_bad_negative_fraction(self):
        with pytest.raises(FareyError):
            farey_range(0, 1, Fraction(-2, 3))
