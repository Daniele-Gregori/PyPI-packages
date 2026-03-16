"""Tests for the farey package."""

import pytest
from fractions import Fraction

from farey import farey_sequence, farey_range, FareyError


# ===========================================================================
# FareySequence
# ===========================================================================

class TestFareySequence:
    def test_order_1(self):
        """F_1 = [0/1, 1/1]"""
        assert farey_sequence(1) == [Fraction(0, 1), Fraction(1, 1)]

    def test_order_2(self):
        """F_2 = [0/1, 1/2, 1/1]"""
        assert farey_sequence(2) == [
            Fraction(0, 1), Fraction(1, 2), Fraction(1, 1),
        ]

    def test_order_3(self):
        """F_3 has 5 elements."""
        result = farey_sequence(3)
        assert result == [
            Fraction(0, 1), Fraction(1, 3), Fraction(1, 2),
            Fraction(2, 3), Fraction(1, 1),
        ]

    def test_order_5(self):
        """F_5 has 11 elements."""
        result = farey_sequence(5)
        assert len(result) == 11
        assert result[0] == Fraction(0, 1)
        assert result[-1] == Fraction(1, 1)

    def test_order_7_length(self):
        """F_7 has 19 elements."""
        assert len(farey_sequence(7)) == 19

    def test_sorted(self):
        """Result is strictly increasing."""
        seq = farey_sequence(6)
        for a, b in zip(seq, seq[1:]):
            assert a < b

    def test_all_reduced(self):
        """All fractions are in lowest terms."""
        from math import gcd
        for f in farey_sequence(8):
            assert gcd(f.numerator, f.denominator) == 1

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
# FareyRange
# ===========================================================================

class TestFareyRangeBasic:
    def test_unit_interval_order_3(self):
        """farey_range(0, 1, 3) == farey_sequence(3)"""
        result = farey_range(0, 1, 3)
        expected = farey_sequence(3)
        assert result == expected

    def test_default_step(self):
        """Default step uses order 1."""
        result = farey_range(0, 1)
        assert result == [Fraction(0, 1), Fraction(1, 1)]

    def test_two_unit_intervals(self):
        """Order 2 over [0, 2] tiles each unit interval."""
        result = farey_range(0, 2, 2)
        assert result == [
            Fraction(0, 1), Fraction(1, 2), Fraction(1, 1),
            Fraction(3, 2), Fraction(2, 1),
        ]

    def test_negative_to_positive(self):
        """Interval [-1, 1] with order 2."""
        result = farey_range(-1, 1, 2)
        assert len(result) == 5
        assert result[0] == Fraction(-1, 1)
        assert result[-1] == Fraction(1, 1)

    def test_returns_fractions_for_integer_endpoints(self):
        """Integer endpoints produce Fraction results."""
        result = farey_range(0, 3, 2)
        assert all(isinstance(v, Fraction) for v in result)

    def test_non_integer_span_returns_floats(self):
        """Non-integer span returns float results."""
        result = farey_range(0, 1.5, 3)
        assert all(isinstance(v, float) for v in result)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.5)

    def test_sorted_ascending(self):
        """Default direction is ascending."""
        result = farey_range(0, 5, 4)
        for a, b in zip(result, result[1:]):
            assert a < b


class TestFareyRangeReverse:
    def test_negative_step_reverses(self):
        """Negative integer step reverses the order."""
        fwd = farey_range(0, 1, 3)
        rev = farey_range(0, 1, -3)
        assert rev == list(reversed(fwd))

    def test_negative_fraction_step_reverses(self):
        """Negative Fraction step reverses the order."""
        fwd = farey_range(0, 2, Fraction(1, 3))
        rev = farey_range(0, 2, Fraction(-1, 3))
        assert rev == list(reversed(fwd))


class TestFareyRangeFractionStep:
    def test_s_and_1_over_s_match_order_3(self):
        """step=3 and step=Fraction(1,3) give same result on [0, 2]."""
        assert farey_range(0, 2, 3) == farey_range(0, 2, Fraction(1, 3))

    def test_s_and_1_over_s_match_order_5(self):
        """step=5 and step=Fraction(1,5) give same result on [-1, 1]."""
        assert farey_range(-1, 1, 5) == farey_range(-1, 1, Fraction(1, 5))

    def test_s_and_1_over_s_match_order_7(self):
        """step=7 and step=Fraction(1,7) give same result on [0, 3]."""
        assert farey_range(0, 3, 7) == farey_range(0, 3, Fraction(1, 7))

    def test_neg_s_and_neg_1_over_s_match_order_3(self):
        """step=-3 and step=Fraction(-1,3) give same result on [0, 2]."""
        assert farey_range(0, 2, -3) == farey_range(0, 2, Fraction(-1, 3))

    def test_neg_s_and_neg_1_over_s_match_order_5(self):
        """step=-5 and step=Fraction(-1,5) give same result on [-1, 1]."""
        assert farey_range(-1, 1, -5) == farey_range(-1, 1, Fraction(-1, 5))

    def test_neg_s_and_neg_1_over_s_match_order_7(self):
        """step=-7 and step=Fraction(-1,7) give same result on [0, 3]."""
        assert farey_range(0, 3, -7) == farey_range(0, 3, Fraction(-1, 7))

    def test_s_and_1_over_s_match_order_4_shifted(self):
        """step=4 and step=Fraction(1,4) give same result on [5, 10]."""
        assert farey_range(5, 10, 4) == farey_range(5, 10, Fraction(1, 4))

    def test_neg_s_and_neg_1_over_s_match_order_4_shifted(self):
        """step=-4 and step=Fraction(-1,4) give same result on [5, 10]."""
        assert farey_range(5, 10, -4) == farey_range(5, 10, Fraction(-1, 4))

    def test_float_1_over_3(self):
        """step=1/3 (float 0.333...) gives same result as step=3."""
        assert farey_range(0, 2, 1/3) == farey_range(0, 2, 3)

    def test_float_1_over_5(self):
        """step=1/5 (float 0.2) gives same result as step=5."""
        assert farey_range(-1, 1, 1/5) == farey_range(-1, 1, 5)

    def test_float_1_over_7(self):
        """step=1/7 (float) gives same result as step=7."""
        assert farey_range(0, 3, 1/7) == farey_range(0, 3, 7)

    def test_float_neg_1_over_3(self):
        """step=-1/3 (float) gives same result as step=-3."""
        assert farey_range(0, 2, -1/3) == farey_range(0, 2, -3)

    def test_float_neg_1_over_5(self):
        """step=-1/5 (float) gives same result as step=-5."""
        assert farey_range(-1, 1, -1/5) == farey_range(-1, 1, -5)


class TestFareyRangeErrors:
    def test_step_zero_int(self):
        with pytest.raises(FareyError):
            farey_range(0, 1, 0)

    def test_step_zero_fraction(self):
        with pytest.raises(FareyError):
            farey_range(0, 1, Fraction(0))

    def test_step_bad_fraction(self):
        """Fraction like 2/3 is not a valid step."""
        with pytest.raises(FareyError):
            farey_range(0, 1, Fraction(2, 3))


class TestFareyRangeEdgeCases:
    def test_single_point_interval(self):
        """start == end should return a single-element list."""
        result = farey_range(3, 3, 5)
        assert len(result) == 1

    def test_large_order(self):
        """Higher order produces more points."""
        r3 = farey_range(0, 1, 3)
        r7 = farey_range(0, 1, 7)
        assert len(r7) > len(r3)

    def test_float_step(self):
        """Float 4.0 treated as integer order 4."""
        r_float = farey_range(0, 1, 4.0)
        r_int = farey_range(0, 1, 4)
        assert r_float == r_int


# ===========================================================================
# Explicit result tests — FareySequence
# ===========================================================================

class TestFareySequenceExplicit:
    def test_order_4(self):
        """F_4 exact elements."""
        assert farey_sequence(4) == [
            Fraction(0, 1), Fraction(1, 4), Fraction(1, 3),
            Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
            Fraction(1, 1),
        ]

    def test_order_5_explicit(self):
        """F_5 exact elements."""
        assert farey_sequence(5) == [
            Fraction(0, 1), Fraction(1, 5), Fraction(1, 4),
            Fraction(1, 3), Fraction(2, 5), Fraction(1, 2),
            Fraction(3, 5), Fraction(2, 3), Fraction(3, 4),
            Fraction(4, 5), Fraction(1, 1),
        ]

    def test_order_6(self):
        """F_6 exact elements."""
        assert farey_sequence(6) == [
            Fraction(0, 1), Fraction(1, 6), Fraction(1, 5),
            Fraction(1, 4), Fraction(1, 3), Fraction(2, 5),
            Fraction(1, 2), Fraction(3, 5), Fraction(2, 3),
            Fraction(3, 4), Fraction(4, 5), Fraction(5, 6),
            Fraction(1, 1),
        ]

    def test_order_7(self):
        """F_7 exact elements."""
        assert farey_sequence(7) == [
            Fraction(0, 1), Fraction(1, 7), Fraction(1, 6),
            Fraction(1, 5), Fraction(1, 4), Fraction(2, 7),
            Fraction(1, 3), Fraction(2, 5), Fraction(3, 7),
            Fraction(1, 2), Fraction(4, 7), Fraction(3, 5),
            Fraction(2, 3), Fraction(5, 7), Fraction(3, 4),
            Fraction(4, 5), Fraction(5, 6), Fraction(6, 7),
            Fraction(1, 1),
        ]

    def test_order_8_length(self):
        """F_8 has 23 elements."""
        assert len(farey_sequence(8)) == 23

    def test_mediant_property(self):
        """Consecutive Farey fractions a/b, c/d satisfy |bc - ad| = 1."""
        seq = farey_sequence(6)
        for i in range(len(seq) - 1):
            a, b = seq[i].numerator, seq[i].denominator
            c, d = seq[i + 1].numerator, seq[i + 1].denominator
            assert abs(b * c - a * d) == 1


# ===========================================================================
# Explicit result tests — FareyRange
# ===========================================================================

class TestFareyRangeExplicit:
    def test_0_1_order_4(self):
        """farey_range(0, 1, 4) = F_4"""
        assert farey_range(0, 1, 4) == [
            Fraction(0, 1), Fraction(1, 4), Fraction(1, 3),
            Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
            Fraction(1, 1),
        ]

    def test_0_1_order_5(self):
        """farey_range(0, 1, 5) = F_5"""
        assert farey_range(0, 1, 5) == [
            Fraction(0, 1), Fraction(1, 5), Fraction(1, 4),
            Fraction(1, 3), Fraction(2, 5), Fraction(1, 2),
            Fraction(3, 5), Fraction(2, 3), Fraction(3, 4),
            Fraction(4, 5), Fraction(1, 1),
        ]

    def test_0_2_order_4(self):
        """farey_range(0, 2, 4) tiles F_4 over two unit intervals."""
        assert farey_range(0, 2, 4) == [
            Fraction(0, 1), Fraction(1, 4), Fraction(1, 3),
            Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
            Fraction(1, 1), Fraction(5, 4), Fraction(4, 3),
            Fraction(3, 2), Fraction(5, 3), Fraction(7, 4),
            Fraction(2, 1),
        ]

    def test_0_3_order_3(self):
        """farey_range(0, 3, 3) tiles F_3 over three unit intervals."""
        assert farey_range(0, 3, 3) == [
            Fraction(0, 1), Fraction(1, 3), Fraction(1, 2),
            Fraction(2, 3), Fraction(1, 1), Fraction(4, 3),
            Fraction(3, 2), Fraction(5, 3), Fraction(2, 1),
            Fraction(7, 3), Fraction(5, 2), Fraction(8, 3),
            Fraction(3, 1),
        ]

    def test_neg3_3_order_4(self):
        """farey_range(-3, 3, 4) symmetric around zero, 37 elements."""
        result = farey_range(-3, 3, 4)
        assert len(result) == 37
        assert result[0] == Fraction(-3, 1)
        assert result[-1] == Fraction(3, 1)
        # symmetry: midpoint is 0
        mid = len(result) // 2
        assert result[mid] == Fraction(0, 1)
        # each element + its mirror should sum to 0
        for i in range(len(result)):
            assert result[i] + result[-(i + 1)] == 0

    def test_2_5_order_5(self):
        """farey_range(2, 5, 5) shifted interval, order 5."""
        result = farey_range(2, 5, 5)
        assert len(result) == 31
        assert result[0] == Fraction(2, 1)
        assert result[-1] == Fraction(5, 1)
        # first unit interval [2, 3] should contain 2 + F_5
        first_unit = [v for v in result if v <= 3]
        assert first_unit == [
            Fraction(2, 1), Fraction(11, 5), Fraction(9, 4),
            Fraction(7, 3), Fraction(12, 5), Fraction(5, 2),
            Fraction(13, 5), Fraction(8, 3), Fraction(11, 4),
            Fraction(14, 5), Fraction(3, 1),
        ]

    def test_neg1_1_order_5(self):
        """farey_range(-1, 1, 5) explicit."""
        result = farey_range(-1, 1, 5)
        assert len(result) == 21
        assert result == [
            Fraction(-1, 1), Fraction(-4, 5), Fraction(-3, 4),
            Fraction(-2, 3), Fraction(-3, 5), Fraction(-1, 2),
            Fraction(-2, 5), Fraction(-1, 3), Fraction(-1, 4),
            Fraction(-1, 5), Fraction(0, 1), Fraction(1, 5),
            Fraction(1, 4), Fraction(1, 3), Fraction(2, 5),
            Fraction(1, 2), Fraction(3, 5), Fraction(2, 3),
            Fraction(3, 4), Fraction(4, 5), Fraction(1, 1),
        ]

    def test_0_1_order_5_reversed(self):
        """farey_range(0, 1, -5) explicit descending."""
        assert farey_range(0, 1, -5) == [
            Fraction(1, 1), Fraction(4, 5), Fraction(3, 4),
            Fraction(2, 3), Fraction(3, 5), Fraction(1, 2),
            Fraction(2, 5), Fraction(1, 3), Fraction(1, 4),
            Fraction(1, 5), Fraction(0, 1),
        ]

    def test_0_2_order_4_reversed(self):
        """farey_range(0, 2, -4) explicit descending."""
        fwd = farey_range(0, 2, 4)
        rev = farey_range(0, 2, -4)
        assert rev == list(reversed(fwd))
        assert rev[0] == Fraction(2, 1)
        assert rev[-1] == Fraction(0, 1)

    def test_non_integer_span_order_4(self):
        """farey_range(1, 3.5, 4) scales F_4 over [1, 3.5]."""
        result = farey_range(1, 3.5, 4)
        span = 2.5
        expected = sorted(set(1.0 + float(f) * span for f in farey_sequence(4)))
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r == pytest.approx(e)

    def test_non_integer_span_order_5(self):
        """farey_range(0, 0.7, 5) scales F_5 over [0, 0.7]."""
        result = farey_range(0, 0.7, 5)
        span = 0.7
        expected = sorted(set(float(f) * span for f in farey_sequence(5)))
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r == pytest.approx(e)

    def test_10_13_order_6(self):
        """farey_range(10, 13, 6) large shifted interval, order 6."""
        result = farey_range(10, 13, 6)
        assert result[0] == Fraction(10, 1)
        assert result[-1] == Fraction(13, 1)
        # 3 unit intervals × 13 elements in F_6, minus overlaps at boundaries
        # F_6 has 13 elements → 3 * 13 - 2 (shared endpoints) = 37
        assert len(result) == 37

    def test_single_point(self):
        """farey_range(7, 7, 5) returns [7]."""
        result = farey_range(7, 7, 5)
        assert result == [Fraction(7, 1)]
