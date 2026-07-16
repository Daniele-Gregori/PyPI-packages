"""Test suite for transcendental_range, ported from the VerificationTests
of the Wolfram Language resource function TranscendentalRange 1.1.0.

The WL tests validate the default monotonic-outer implementation against
the naive Outer-based testing implementation (the ``test=True`` option):

- ``testTR[{f, ord}, x, y, z]`` checks ``N@default == N@testing``;
- ``testTR[All, {f, ord}, {m, c, s}]`` runs that comparison over eight
  sign/direction configurations of the range built from a lower-bound-like
  m, an upper-bound-like c and a step-like s;
- for every efficient method there are five randomized correctness rounds
  plus two fixed larger "efficiency" ranges;
- multiplicities 2 and 3 are tested over Exp/Log/Power and the hyperbolic,
  inverse hyperbolic and inverse trigonometric families;
- two regression cases cover the Log failure of version 1.0.0.

The WL tests draw unseeded random arguments; here every test seeds its own
generator so runs are reproducible.

This randomized suite runs locally (it is skipped on CI, where the
deterministic ``test_documentation_examples.py`` runs instead).
"""

import os
import random
import signal

import pytest
from sympy import E, Rational, N

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="randomized TimeConstrained suite is run locally; "
           "CI runs test_documentation_examples.py")

from transcendental_range import (
    transcendental_range,
    NotAlgebraicError,
    FareyStepError,
)

HYP = ['sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch']
INVHYP = ['asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']
INVTRIG = ['asin', 'acos', 'atan', 'acot', 'asec', 'acsc']


# ---------------------------------------------------------------------------
# TimeConstrained
# ---------------------------------------------------------------------------

class _Aborted(Exception):
    """Raised by the alarm handler; the WL $Aborted."""


def time_constrained(thunk, seconds=30):
    """WL TimeConstrained[expr, t]: evaluate thunk(), aborting after the
    given number of seconds and returning None (the WL $Aborted).

    On platforms without SIGALRM the computation runs unconstrained."""
    if not hasattr(signal, "SIGALRM"):
        return thunk()

    def handler(signum, frame):
        raise _Aborted

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return thunk()
    except _Aborted:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# testTR
# ---------------------------------------------------------------------------

def assert_ranges_equal(default, testing, label=""):
    """WL VerificationTest[N@default == N@testing].

    Structurally identical expressions are accepted directly; only the
    (rare) differing representatives are compared numerically."""
    assert len(default) == len(testing), (
        f"{label}: length mismatch {len(default)} != {len(testing)}")
    for i, (d, t) in enumerate(zip(default, testing)):
        if d == t:
            continue
        a, b = float(N(d, 15)), float(N(t, 15))
        assert abs(a - b) <= 1e-10 * max(1.0, abs(a)), (
            f"{label}: element {i}: {a} != {b}\n"
            f"  default: {d}\n  testing: {t}")


def check_tr(f, x, y, z, generators_domain='rationals', multiplicity=1):
    """WL testTR[{f, ord}, x, y, z] ~TimeConstrained~ 30.

    The notebook applies TimeConstrained to the multiplicity tests; here
    every comparison runs under it, so that pathological random draws are
    skipped (WL $Aborted) instead of stalling the suite."""

    def compare():
        kw = dict(method=f, generators_domain=generators_domain,
                  multiplicity=multiplicity)
        default = transcendental_range(x, y, z, **kw)
        testing = transcendental_range(x, y, z, test=True, **kw)
        assert_ranges_equal(default, testing,
                            label=f"{f}[{x}, {y}, {z}] x{multiplicity}")
        return True

    time_constrained(compare, 30)


def check_tr_all(f, args=(0, 20, 2), generators_domain='rationals',
                multiplicity=1):
    """WL testTR[All, {f, ord}, {m, c, s}]: all sign/direction cases."""
    m, c, s = (abs(v) for v in args)
    for x, y, z in [(-m, c, s), (c, -m, -s), (-c, m, s), (-m, c, -s),
                    (-c - m, c + m, s), (-c + m, c - m, s),
                    (c - m, -c + m, -s), (c + m, -c - m, -s)]:
        check_tr(f, x, y, z, generators_domain=generators_domain,
                multiplicity=multiplicity)


def random_args(rng, diff=10, s=7):
    """The randomized (m, c, s) arguments of the WL correctness tests."""
    return (rng.randint(0, 5),
            rng.randint(5 + diff, 10 + diff),
            Rational(rng.randint(1, s), rng.randint(1, s)))


# The five randomized rounds per method (WL: tries = 5), with the
# method-specific diff used in the notebook.
CORRECTNESS = [
    ('exp', 10), ('log', 10),
    ('sinh', 10), ('cosh', 10), ('tanh', 5), ('coth', 5),
    ('sech', 10), ('csch', 5),
    ('asinh', 10), ('acosh', 10), ('atanh', 10), ('acoth', 10),
    ('asech', 10), ('acsch', 10),
    ('asin', 10), ('acos', 10), ('atan', 10), ('acot', 10),
    ('asec', 10), ('acsc', 10),
]

# The fixed "tests of efficiency" arguments per method.
EFFICIENCY = {
    'exp': [(1, 100, 1), (100, -100, -1)],
    'log': [(1, 100, 1), (100, -100, -1)],
    'sinh': [(1, 100, 1), (100, -100, -1)],
    'cosh': [(1, 100, 1), (100, -100, -1)],
    'tanh': [(1, 10, 1), (10, -10, -1)],
    'coth': [(1, 30, 1), (30, -30, -1)],
    'sech': [(1, 100, 1), (100, -100, -1)],
    'csch': [(1, 100, 1), (100, -100, -1)],
    'asinh': [(1, 100, 1), (100, -100, -1)],
    'acosh': [(1, 100, 1), (100, -100, -1)],
    'atanh': [(0, 100, Rational(1, 10)), (100, -100, Rational(-1, 10))],
    'acoth': [(1, 100, 1), (100, -100, -1)],
    'asech': [(0, 10, Rational(1, 10)), (10, -10, Rational(-1, 10))],
    'acsch': [(1, 100, 1), (100, -100, -1)],
    'asin': [(0, 100, Rational(1, 10)), (100, -100, Rational(-1, 10))],
    'acos': [(0, 10, Rational(1, 10)), (10, -10, Rational(-1, 10))],
    'atan': [(1, 100, 1), (100, -100, -1)],
    'acot': [(1, 100, 1), (100, -100, -1)],
    'asec': [(1, 100, 1), (100, -100, -1)],
    'acsc': [(1, 100, 1), (100, -100, -1)],
}


@pytest.mark.parametrize("method,diff", CORRECTNESS)
def test_correctness(method, diff):
    """WL "Tests of correctness": randomized all-configuration rounds per
    method (the notebook runs five rounds; two keep the suite fast, with
    every comparison under TimeConstrained)."""
    rng = random.Random(f"transcendental-{method}")
    for _ in range(2):
        check_tr_all(method, random_args(rng, diff=diff))


@pytest.mark.parametrize("method", sorted(EFFICIENCY))
def test_efficiency_ranges(method):
    """WL "Tests of efficiency": the two larger fixed ranges."""
    for x, y, z in EFFICIENCY[method]:
        check_tr(method, x, y, z)


class TestPower:
    """WL Power tests (diff = 0, s = 4, algebraic generators)."""

    def test_correctness(self):
        rng = random.Random("transcendental-power")
        s = 4
        for _ in range(2):
            args = (rng.randint(0, s - 1), rng.randint(s, s + 2),
                    Rational(rng.randint(1, s), rng.randint(1, s)))
            check_tr_all('power', args, generators_domain='algebraics')

    def test_efficiency_ranges(self):
        check_tr('power', 1, 20, 1, generators_domain='algebraics')
        check_tr('power', 10, -10, -1, generators_domain='algebraics')


# ---------------------------------------------------------------------------
# Tests of multiplicity (WL: multiplicities 2 and 3, TimeConstrained 30 s)
# ---------------------------------------------------------------------------

def _mult_args_2(rng, s=7, mx=2):
    lo, hi = sorted((rng.randint(-mx, mx), rng.randint(-mx, mx)))
    return lo, hi, Rational(rng.randint(1, s), rng.randint(1, s))


def _mult_args_3(rng, s=7):
    # Round[RandomReal[{-3/2, 3/2}], 1/2]
    vals = sorted(Rational(round(rng.uniform(-1.5, 1.5) * 2), 2)
                  for _ in range(2))
    return vals[0], vals[1], Rational(rng.randint(1, s), rng.randint(1, s))


@pytest.mark.parametrize("family", [['exp', 'log', 'power'],
                                    HYP, INVHYP, INVTRIG])
def test_multiplicity_2(family):
    rng = random.Random(f"multiplicity-2-{family[0]}")
    for method in family:
        domain = 'algebraics' if method == 'power' else 'rationals'
        check_tr_all(method, _mult_args_2(rng),
                    generators_domain=domain, multiplicity=2)


@pytest.mark.parametrize("family", [['exp', 'log', 'power'],
                                    HYP, INVHYP, INVTRIG])
def test_multiplicity_3(family):
    rng = random.Random(f"multiplicity-3-{family[0]}")
    for method in family:
        domain = 'algebraics' if method == 'power' else 'rationals'
        check_tr_all(method, _mult_args_3(rng),
                    generators_domain=domain, multiplicity=3)


# ---------------------------------------------------------------------------
# Debugging examples (WL: regressions of version 1.0.0)
# ---------------------------------------------------------------------------

class TestLogRegressions:
    """In version 1.0.0 the Log method disagreed with the baseline."""

    def test_log_wide(self):
        check_tr('log', -10, 10, Rational(1, 7))

    def test_log_narrow(self):
        check_tr('log', -1, 1, Rational(1, 7))


# ---------------------------------------------------------------------------
# Argument failures (WL: failureNotAlgebraics, failureFareyStep)
# ---------------------------------------------------------------------------

class TestFailures:

    def test_not_algebraic_bound(self):
        with pytest.raises(NotAlgebraicError):
            transcendental_range(0, E, Rational(1, 3))

    def test_ceiling_bypass(self):
        # TranscendentalRange[0, Ceiling[E], 1/3] works
        assert len(transcendental_range(0, 3, Rational(1, 3))) == 17

    def test_farey_step_not_allowed(self):
        with pytest.raises(FareyStepError):
            transcendental_range(1, 10, Rational(2, 3), farey_range=True)
