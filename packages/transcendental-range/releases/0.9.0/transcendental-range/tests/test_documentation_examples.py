"""Deterministic test suite for transcendental_range, suitable for CI.

Transcribes the documentation examples of the Wolfram Language resource
function TranscendentalRange 1.1.0 (exact expected outputs and element
counts) and runs a fixed default-vs-naive comparison for every method —
the deterministic core of the randomized VerificationTests suite in
``test_transcendental_range.py``, which runs locally.
"""

import pytest
from sympy import E, N, Rational, exp, log, pi, sqrt

from transcendental_range import (
    transcendental_range,
    NotAlgebraicError,
    FareyStepError,
)

ALL_METHODS = ['exp', 'log', 'power',
               'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
               'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
               'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
               'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']


def nvals(result):
    return [float(N(e, 15)) for e in result]


def assert_matches(result, expected):
    """Numerical comparison against an expected exact list."""
    rv, ev = nvals(result), nvals(expected)
    assert len(rv) == len(ev), f"{len(rv)} != {len(ev)}: {result}"
    for a, b in zip(rv, ev):
        assert abs(a - b) <= 1e-10 * max(1.0, abs(a))


# ---------------------------------------------------------------------------
# Basic examples
# ---------------------------------------------------------------------------

class TestBasicExamples:

    def test_single_argument(self):
        assert transcendental_range(10) == [E, 2 * E, exp(2), 3 * E]

    def test_two_arguments(self):
        assert_matches(
            transcendental_range(-2, 2),
            [-2 * exp(-1), -exp(-1), -2 * exp(-2), -exp(-2),
             exp(-2), 2 * exp(-2), exp(-1), 2 * exp(-1)])

    def test_step(self):
        assert_matches(
            transcendental_range(0, 4, Rational(1, 2)),
            [exp(Rational(1, 2)) / 2, E / 2, exp(Rational(1, 2)),
             exp(Rational(3, 2)) / 2, 3 * exp(Rational(1, 2)) / 2, E,
             2 * exp(Rational(1, 2)), exp(2) / 2])

    def test_negative_step_descends(self):
        result = transcendental_range(100, 1, -1)
        assert result[-1] == E and len(result) == 54
        vals = nvals(result)
        assert vals == sorted(vals, reverse=True)

    def test_step_lower_bound(self):
        result = transcendental_range(-2, 2, Rational(1, 4), Rational(1, 8))
        assert len(result) == 26
        vals = nvals(result)
        assert all(b - a >= Rational(1, 8) - Rational(1, 10**9)
                   for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# Options examples
# ---------------------------------------------------------------------------

class TestMethodExamples:

    def test_log(self):
        # The WL 1.1.0 documentation shows 41 elements, including both
        # 3 Log[8] and 9 Log[2] — the same number, split by a 1-ulp
        # machine-float difference; the port groups them, giving 40.
        result = transcendental_range(10, method='log')
        assert result[:4] == [log(3), 2 * log(2), log(5), log(6)]
        assert len(result) == 40

    def test_sinh(self):
        from sympy import sinh
        assert_matches(
            transcendental_range(-10, 10, method='sinh')[:4],
            [-8 * sinh(1), -7 * sinh(1), -2 * sinh(2), -6 * sinh(1)])
        assert len(transcendental_range(-10, 10, method='sinh')) == 20

    def test_atan_pi_multiples(self):
        from sympy import atan
        result = transcendental_range(-2, 2, method='atan')
        assert_matches(result, [-pi / 2, -atan(2), -pi / 4,
                                pi / 4, atan(2), pi / 2])

    def test_power_algebraics(self):
        result = transcendental_range(-3, 3, method='power',
                                      generators_domain='algebraics')
        assert len(result) == 54
        assert result[0] == 3 ** (-2 * sqrt(2))

    def test_method_list(self):
        result = transcendental_range(-4, 4, method=['acot', 'exp'])
        assert len(result) == 66
        assert result[0] == -pi

    def test_method_all(self):
        result = transcendental_range(3, method='all')
        assert len(result) == 59


class TestOptionExamples:

    def test_generators_domain(self):
        alg = transcendental_range(8, generators_domain='algebraics')
        rat = transcendental_range(8, generators_domain='rationals')
        assert rat == [E, 2 * E, exp(2)]
        assert len(alg) == 14
        assert set(nvals(rat)) <= set(nvals(alg))

    def test_farey_range(self):
        far = transcendental_range(1, 10, Rational(1, 3), farey_range=True)
        assert len(far) == 30
        plain = transcendental_range(1, 10, Rational(1, 3))
        assert len(plain) == 19
        assert set(nvals(plain)) <= set(nvals(far))

    def test_multiplicity_2(self):
        result = transcendental_range(0, 6, Rational(1, 2), multiplicity=2)
        assert len(result) == 32
        assert result[0] == exp(Rational(1, 2))

    def test_working_precision(self):
        mach = transcendental_range(20, 25, method='tanh')
        assert len(mach) == 5
        high = transcendental_range(20, 25, method='tanh',
                                    working_precision=30)
        assert len(high) == 30


# ---------------------------------------------------------------------------
# Default vs naive baseline, fixed ranges (deterministic testTR core)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", [m for m in ALL_METHODS
                                    if m != 'power'])
def test_default_equals_naive(method):
    """WL testTR on a fixed range covering both signs and a rational
    step (descending variant included)."""
    for x, y, z in [(-3, 8, Rational(1, 2)), (8, -3, Rational(-1, 2))]:
        kw = dict(method=method)
        default = transcendental_range(x, y, z, **kw)
        testing = transcendental_range(x, y, z, test=True, **kw)
        assert len(default) == len(testing), f"{method}[{x},{y},{z}]"
        for d, t in zip(default, testing):
            if d == t:
                continue
            a, b = float(N(d, 15)), float(N(t, 15))
            assert abs(a - b) <= 1e-10 * max(1.0, abs(a)), (
                f"{method}[{x},{y},{z}]: {d} != {t}")


def test_default_equals_naive_power():
    for x, y, z in [(1, 5, Rational(1, 2)), (5, -5, -1)]:
        default = transcendental_range(x, y, z, method='power',
                                       generators_domain='algebraics')
        testing = transcendental_range(x, y, z, method='power',
                                       generators_domain='algebraics',
                                       test=True)
        assert nvals(default) == pytest.approx(nvals(testing))


def test_log_regression_1_0_0():
    """The Log range that failed in WL 1.0.0."""
    default = transcendental_range(-10, 10, Rational(1, 7), method='log')
    testing = transcendental_range(-10, 10, Rational(1, 7), method='log',
                                   test=True)
    assert len(default) == len(testing) == 6262


# ---------------------------------------------------------------------------
# Argument failures
# ---------------------------------------------------------------------------

class TestFailures:

    def test_not_algebraic_bound(self):
        with pytest.raises(NotAlgebraicError):
            transcendental_range(0, E, Rational(1, 3))

    def test_ceiling_bypass(self):
        assert len(transcendental_range(0, 3, Rational(1, 3))) == 17

    def test_farey_step_not_allowed(self):
        with pytest.raises(FareyStepError):
            transcendental_range(1, 10, Rational(2, 3), farey_range=True)

    def test_bad_multiplicity(self):
        with pytest.raises(ValueError):
            transcendental_range(5, multiplicity=0)

    def test_zero_step(self):
        assert transcendental_range(1, 5, 0) == []
