"""Ground-truth round-trip tests for the closed-form search ranges.

The "Algebraic" and "Transcendental" search ranges (find-closed-form
0.5.0) draw the search arguments from the sibling packages
``algebraic-range`` and ``transcendental-range``, extending the WL
``FindClosedForm`` option set.

Each test invents an exact formula — a function evaluated at an exact
algebraic or transcendental argument — takes its bare float value, and
checks that ``find_closed_form`` recovers the original formula from the
float alone (the reverse direction).

Every search runs under TimeConstrained (WL semantics), twice over:
the test harness bounds the whole call through SIGALRM, and the call
itself is passed a ``search_time_limit``, so a regression can slow the
suite down but never hang it.
"""

import signal
import time
from fractions import Fraction

import pytest
import sympy
from sympy import Rational, exp, sqrt, log, atan, zeta, gamma as spgamma

pytest.importorskip("algebraic_range", reason="algebraic-range not installed")

from algebraic_range import algebraic_range

# transcendental-range needs Python >= 3.10; skip only its tests without it
try:
    from transcendental_range import transcendental_range
    HAVE_TRANSCENDENTAL = True
except ImportError:  # pragma: no cover
    transcendental_range = None
    HAVE_TRANSCENDENTAL = False

needs_transcendental = pytest.mark.skipif(
    not HAVE_TRANSCENDENTAL, reason="transcendental-range not installed")

from find_closed_form import find_closed_form, FindClosedFormError


# ---------------------------------------------------------------------------
# TimeConstrained (as in the transcendental-range test suite)
# ---------------------------------------------------------------------------

TIME_LIMIT = 60  # seconds per search; each takes well under 5 s normally


class _Aborted(Exception):
    """Raised by the alarm handler; the WL $Aborted."""


def time_constrained(thunk, seconds=TIME_LIMIT):
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


def roundtrip(target, **kw):
    """float(target) → find_closed_form → exact comparison with target."""
    y = float(target)  # the ground truth enters as a bare float
    kw.setdefault("search_time_limit", TIME_LIMIT / 2)
    result = time_constrained(lambda: find_closed_form(y, **kw))
    assert result is not None, f"no closed form found for {y} (or timed out)"
    assert sympy.simplify(result - target) == 0, f"{result} != {target}"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Algebraic search range: functions at algebraic arguments
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlgebraicSearchRange:

    def test_exp_at_sqrt2_default_functions(self):
        """exp(sqrt(2)) recovered from 4.1132503787829275 by the defaults."""
        roundtrip(exp(sqrt(2)), search_range="Algebraic")

    def test_gamma_at_algebraic_argument(self):
        """gamma(sqrt(3)) recovered from 0.9151022969730863."""
        roundtrip(spgamma(sqrt(3)), functions=lambda x: spgamma(x),
                  search_range="Algebraic")

    def test_root_order_option_forwarded(self):
        """exp(2**(1/3)) needs cube roots: root_order reaches the generator."""
        roundtrip(exp(2 ** Rational(1, 3)), functions=lambda x: exp(x),
                  search_range="Algebraic",
                  search_range_options={"root_order": 3})

    def test_search_arguments_from_algebraic_range(self):
        """An algebraic_range output passed directly as fixed arguments."""
        args = algebraic_range(0, 2, Fraction(1, 2))
        roundtrip(exp(sqrt(2)), functions=lambda x: exp(x),
                  search_arguments=args)

    def test_callable_search_range(self):
        """WL parity: search_range accepts a function of the round number."""
        roundtrip(exp(sqrt(2)), functions=lambda x: exp(x),
                  search_range=lambda cut:
                      algebraic_range(-cut, cut, Fraction(1, cut)))


# ═══════════════════════════════════════════════════════════════════════════════
# Transcendental search range: functions at transcendental arguments
# ═══════════════════════════════════════════════════════════════════════════════

@needs_transcendental
class TestTranscendentalSearchRange:

    def test_atan_at_log2(self):
        """atan(log(2)) recovered from 0.606111934732855."""
        roundtrip(atan(log(2)), functions=lambda x: atan(x),
                  search_range="Transcendental",
                  search_range_options={"method": "log"})

    def test_zeta_at_transcendental_argument_defaults(self):
        """zeta(exp(1/2)) recovered from 2.1638308208408383 by the defaults."""
        roundtrip(zeta(exp(Rational(1, 2))), search_range="Transcendental")

    def test_identity_included_in_defaults(self):
        """sqrt(2)*log(2): on these ranges the elements themselves are
        candidate formulae, up to an algebraic factor."""
        roundtrip(sqrt(2) * log(2), search_range="Transcendental",
                  search_range_options={"method": "log"})

    def test_power_method_gelfond_schneider(self):
        """2**sqrt(2) (Gelfond–Schneider) is a range element of the 'power'
        method over algebraic generators, recognized by the identity."""
        roundtrip(2 ** sqrt(2), functions=lambda x: x,
                  search_range="Transcendental",
                  search_range_options={"method": "power",
                                        "generators_domain": "algebraics"})

    def test_search_arguments_from_transcendental_range(self):
        """A transcendental_range output passed directly as fixed arguments."""
        args = transcendental_range(-1, 2, Fraction(1, 2), method="log")
        roundtrip(atan(log(2)), functions=lambda x: atan(x),
                  search_arguments=args)

    def test_empty_generator_rounds_are_skipped(self):
        """The log range is empty at round 1; the search must carry on to
        later rounds instead of failing."""
        roundtrip(log(3) / 2, functions=lambda x: x,
                  search_range="Transcendental",
                  search_range_options={"method": "log"},
                  max_search_rounds=4)


# ═══════════════════════════════════════════════════════════════════════════════
# Options and error handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchRangeOptionsAndErrors:

    def test_unknown_search_range_raises(self):
        with pytest.raises(FindClosedFormError, match="Unknown search_range"):
            find_closed_form(1.5, search_range="Bogus")

    def test_options_require_generated_range(self):
        with pytest.raises(FindClosedFormError, match="search_range_options"):
            find_closed_form(1.5, search_range="Farey",
                             search_range_options={"method": "log"})

    def test_search_time_limit_returns_promptly(self):
        """The search's own TimeConstrained: a hopeless search under a small
        time limit must stop near the limit, not run to exhaustion."""
        start = time.monotonic()
        result = find_closed_form(
            1.2345678901234567, functions=lambda x: zeta(x) ** 3,
            search_range="Algebraic", search_time_limit=0.5)
        elapsed = time.monotonic() - start
        assert elapsed < 10.0
        assert result is None or isinstance(result, sympy.Basic)
