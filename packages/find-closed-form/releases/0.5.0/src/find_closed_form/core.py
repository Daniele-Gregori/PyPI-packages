"""
find_closed_form — Find closed-form expressions for numerical values.

A Python translation of the Wolfram Language ResourceFunction ``FindClosedForm``
originally contributed by Daniele Gregori.

The algorithm searches for closed-form mathematical expressions that match a
given numerical value by evaluating candidate functions over Farey-based
argument ranges and testing digit-level agreement.

Implementation notes (differences from the WL original)
-------------------------------------------------------
Ported from the published resource function 1.0.0 plus the bug fixes of
its working kernels 1.0.0.1-1.0.0.4 (``formula_complexity`` weights,
complexity thresholds, the ``erfc`` and ``Ei`` default functions).

Options: 10 of the 15 WL options are ported with matching defaults
(``SignificantDigits``, ``FormulaComplexity``, ``AlgebraicFactor``,
``AlgebraicAdd``, ``RationalSolutions``, ``SearchArguments``,
``SearchRange``, ``MaxSearchRounds``, ``SearchTimeLimit``,
``MonitorSearch``).  Not ported: ``WolframAlphaQueries`` and
``SearchQueries`` (the WolframAlpha integration — this package is
offline by design), ``RootApproximantMethod`` (the lookup-table method
is always used), ``OutputArguments`` and ``SearchComplex`` (deferred).
Python-only additions: ``search_range_options``, ``search_range_fn``
and the ``"Algebraic"``/``"Transcendental"`` search ranges.

Default functions: 31 of the WL 34 — ``Glaisher^#``, ``Khinchin^#`` and
``BarnesG`` are omitted, having no sympy symbolic equivalents.

Search internals: the WL ``functionChamber`` argument chambers (a
per-function argument restriction speeding up the algebraic-combination
steps, noticeable on multi-argument searches) and the
``simplifyRational`` cleanup of rational factors are not yet ported.
Since sympy and WL canonicalize expressions differently, a search may
return an alternative — equally precise — first match.

Deliberate deviations: ``search_time_limit`` is a hard interrupt
(``SIGALRM``) where available; unknown ``search_range`` values raise
``FindClosedFormError`` instead of silently falling back to
``"Farey"``; floats in ``search_arguments`` are exactified via
``Fraction.limit_denominator`` (so 0.1 means 1/10); oversized
multi-argument ranges truncate keeping the smallest-magnitude
arguments.

The full statement-level reduction against the WL kernels is documented
in ``wolfram/REPORT-wl-differences.md`` in the repository.
"""

from __future__ import annotations

import bisect
import inspect
import itertools
import math
import signal
import threading
import time
from contextlib import contextmanager
from fractions import Fraction
from typing import Callable, List, Optional, Tuple, Union

import sympy
from sympy import (
    Integer,
    Rational,
    pi,
    E,
    EulerGamma,
    Catalan,
    GoldenRatio,
    log,
    exp,
    sin,
    cos,
    tan,
    cot,
    asin,
    acos,
    atan,
    acot,
    sinh,
    cosh,
    tanh,
    coth,
    sech,
    csch,
    asinh,
    acosh,
    atanh,
    acoth,
    gamma as spgamma,
    polygamma,
    zeta,
    erf,
    erfinv,
    elliptic_k,
    elliptic_e,
    sqrt,
    airyai,
    airybi,
    erfc,
    Ei,
)
from farey import farey_range as _ext_farey_range

__all__ = [
    "find_closed_form",
    "formula_complexity",
    "farey_range",
    "FindClosedFormError",
]

Number = Union[int, float, Fraction, sympy.Basic]


class FindClosedFormError(Exception):
    """Base exception for find_closed_form errors."""


# ═══════════════════════════════════════════════════════════════════════════════
# Search time limit  (WL TimeConstrained)
# ═══════════════════════════════════════════════════════════════════════════════

class _SearchAborted(Exception):
    """Raised by the alarm handler when the search time limit expires."""


@contextmanager
def _search_time_limit(seconds):
    """
    WL ``TimeConstrained[..., seconds]`` around the whole search block.

    On Unix main threads the limit is enforced through
    ``signal.setitimer``/``SIGALRM``, so even a single long symbolic
    evaluation is interrupted; elsewhere (Windows, non-main threads) the
    cooperative clock checks in the search loop remain the only guard.
    A pre-existing interval timer is re-armed on exit, so nested use
    (e.g. a caller's own ``time_constrained``) stays safe.
    """
    use_signal = (
        seconds is not None
        and math.isfinite(seconds)
        and seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not use_signal:
        yield
        return

    def _handler(signum, frame):
        raise _SearchAborted

    old_handler = signal.signal(signal.SIGALRM, _handler)
    old_delay, old_interval = signal.setitimer(signal.ITIMER_REAL, seconds)
    start = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_delay:
            remaining = old_delay - (time.monotonic() - start)
            signal.setitimer(signal.ITIMER_REAL, max(remaining, 1e-3),
                             old_interval)


# ═══════════════════════════════════════════════════════════════════════════════
# Farey range  (delegate to the ``farey`` package)
# ═══════════════════════════════════════════════════════════════════════════════

def farey_range(
    start: Union[int, float, Fraction],
    end: Union[int, float, Fraction],
    order: Union[int, None] = None,
) -> list:
    """Generate a Farey-based range of rationals, wrapping ``farey.farey_range``."""
    return _ext_farey_range(start, end, order)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula complexity
# ═══════════════════════════════════════════════════════════════════════════════

def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _integer_length(n: int) -> int:
    if n == 0:
        return 1
    return len(str(abs(n)))


def _prime_omega(n: int) -> int:
    """Ω(n): prime factors with multiplicity; Ω(1) = 1 (WL ``FactorInteger[1]``)."""
    if n <= 1:
        return 1
    return sum(sympy.factorint(n).values())


def _collect_integers(expr, mult: int = 1) -> List[int]:
    """Extract all integers from a sympy expression, duplicated for root degrees."""
    if not isinstance(expr, sympy.Basic):
        return []

    known_constants = {pi, E, EulerGamma, Catalan, GoldenRatio}
    if expr in known_constants:
        return [1] * mult

    if expr.is_Integer:
        return [int(expr)] * mult

    if expr.is_Rational and not expr.is_Integer:
        return [int(expr.p)] * mult + [int(expr.q)] * mult

    if expr.func == sympy.Pow and len(expr.args) == 2:
        base, ex = expr.args
        if ex.is_Rational and not ex.is_Integer:
            # WL kernel 1.0.0.4: a root of degree m/n duplicates its base
            # |m| + |n| times (published 1.0.0 used |m*n|)
            deg = abs(int(ex.p)) + abs(int(ex.q))
            return _collect_integers(base, mult * deg)
        if ex.is_Integer and int(ex) != 1:
            # integer powers count the base once plus the exponent itself
            return _collect_integers(base, mult) + [int(ex)] * mult

    result: list[int] = []
    for a in expr.args:
        result.extend(_collect_integers(a, mult))
    return result


def formula_complexity(expr) -> float:
    """
    Heuristic complexity of a sympy expression (WL kernel 1.0.0.4).

    For each integer ``i`` in the expression (non-positive ``j`` mapped to
    ``-j + 1`` first): ``(5*IntegerLength + DigitSum + Ω + sqrt(i)) / 8``,
    where ``Ω`` counts prime factors with multiplicity.  Sum over all
    integers; roots duplicate their base by degree; constants
    (pi, E, …) count as 1.
    """
    if isinstance(expr, (int, float)):
        expr = sympy.sympify(expr)
    if isinstance(expr, Fraction):
        expr = Rational(expr.numerator, expr.denominator)
    if isinstance(expr, list):
        return max((formula_complexity(e) for e in expr), default=0.0)

    ints = _collect_integers(expr)
    if not ints:
        return 0.0
    total = 0.0
    for i in ints:
        if i <= 0:
            i = -i + 1
        total += (
            5 * _integer_length(i) + _digit_sum(i) + _prime_omega(i)
            + math.sqrt(i)
        ) / 8.0
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Precision helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_digits(num: float) -> int:
    """Count significant digits using Python's shortest-repr heuristic."""
    if num == 0:
        return 1
    s = repr(abs(num))
    if "e" in s or "E" in s:
        s = s.split("e")[0].split("E")[0]
    s = s.replace(".", "").lstrip("0")
    return max(len(s), 2)


def _precision_match(num: float, candidate: float, digits: int) -> bool:
    """True when *candidate* reproduces *num* to *digits* − 1 significant digits."""
    if num == 0:
        return abs(candidate) <= 10 ** (-digits + 1)
    return abs(1.0 - candidate / num) <= 10 ** (-digits + 1)


def _safe_neval(expr, digits: int = 18) -> Optional[float]:
    """Numerically evaluate a sympy expression; None on failure or non-real."""
    try:
        val = complex(expr.evalf(n=digits))
        if abs(val.imag) > 1e-15 * max(abs(val.real), 1):
            return None
        return val.real
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Argument range generation
# ═══════════════════════════════════════════════════════════════════════════════

_SEARCH_RANGE_VALUES = ("Farey", "Plain", "Integer", "Algebraic",
                        "Transcendental")


def _algebraic_range_gen() -> Callable:
    """Import ``algebraic_range`` lazily (optional dependency)."""
    try:
        from algebraic_range import algebraic_range
    except ImportError:
        raise ImportError(
            'search_range="Algebraic" requires the algebraic-range package. '
            "Install it with: pip install algebraic-range"
        ) from None
    return algebraic_range


def _transcendental_range_gen() -> Callable:
    """Import ``transcendental_range`` lazily (optional dependency)."""
    try:
        from transcendental_range import transcendental_range
    except ImportError:
        raise ImportError(
            'search_range="Transcendental" requires the '
            "transcendental-range package. "
            "Install it with: pip install transcendental-range"
        ) from None
    return transcendental_range


def _generate_range(
    cut: int,
    search_range: Union[str, Callable],
    custom_fn: Optional[Callable] = None,
    range_options: Optional[dict] = None,
) -> list:
    if custom_fn is not None:
        return custom_fn(cut)
    if callable(search_range):  # WL: Head[OptionValue["SearchRange"]] === Function
        return search_range(cut)
    opts = dict(range_options) if range_options else {}
    if search_range == "Farey":
        return _ext_farey_range(-cut, cut, cut)
    if search_range == "Plain":
        step = Fraction(1, cut)
        result, val, end = [], Fraction(-cut), Fraction(cut)
        while val <= end:
            result.append(val)
            val += step
        return result
    if search_range == "Integer":
        return [Fraction(i) for i in range(-cut, cut + 1)]
    if search_range == "Algebraic":
        return _algebraic_range_gen()(-cut, cut, Fraction(1, cut), **opts)
    if search_range == "Transcendental":
        return _transcendental_range_gen()(-cut, cut, Fraction(1, cut), **opts)
    raise FindClosedFormError(
        f"Unknown search_range {search_range!r}: expected one of "
        + ", ".join(repr(v) for v in _SEARCH_RANGE_VALUES)
        + ", or a callable."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Algebraic number lookup table  (modifiedRootApproximant)
#
# Mirrors the Wolfram ``absLookupBase``.  Positive reals only; sign handled
# separately.  Built lazily on first use.
# ═══════════════════════════════════════════════════════════════════════════════

_ABS_LOOKUP_CACHE: Optional[tuple] = None


def _build_abs_lookup() -> tuple:
    """
    Build a sorted lookup table of positive algebraic-number candidates,
    mirroring the WL ``absLookupBase``.

    Returns four parallel packed arrays (sorted by float value):
      values  – float approximations
      nums    – base numerators
      dens    – base denominators
      orders  – root order (1 = plain fraction, 2 = sqrt, …)
    """
    from array import array

    entries: list[Tuple[float, int, int, int]] = []

    def _add(fv: float, num: int, den: int, order: int) -> None:
        if math.isfinite(fv) and fv >= 0:
            entries.append((fv, num, den, order))

    # 1) Range[0, 10000, 1/10]  – tenths
    for n in range(0, 100_001):
        q, r = divmod(n, 10)
        if r == 0:
            _add(float(q), q, 1, 1)
        else:
            _add(n / 10.0, n, 10, 1)

    # 2) integer ranges: Range[10^4, 10^5], Range[10^5, 10^6, 10],
    #    Range[10^6, 10^8, 1000], Range[10^8, 10^9, 10^5]
    for lo, hi, step in (
        (10_000, 100_000, 1),
        (100_000, 1_000_000, 10),
        (1_000_000, 100_000_000, 1_000),
        (100_000_000, 1_000_000_000, 100_000),
    ):
        for n in range(lo, hi + 1, step):
            _add(float(n), n, 1, 1)

    # 3) FareyRange[0, 1000, 10] and FareyRange[0, 100, 100]
    for f in _ext_farey_range(0, 1000, 10):
        _add(float(f), f.numerator, f.denominator, 1)
    for f in _ext_farey_range(0, 100, 100):
        _add(float(f), f.numerator, f.denominator, 1)

    # 4) Sqrt of FareyRange[0, 100, 50]
    for f in _ext_farey_range(0, 100, 50):
        if f >= 0:
            _add(math.sqrt(float(f)), f.numerator, f.denominator, 2)

    # 5) Higher roots of FareyRange[0, 100, 10]
    for f in _ext_farey_range(0, 100, 10):
        if f > 0:
            fv = float(f)
            for order in (3, 4, 5, 6):
                _add(fv ** (1.0 / order), f.numerator, f.denominator, order)

    entries.sort(key=lambda e: (e[0], e[3]))
    # drop duplicates (same value and root order)
    prev_key = None
    values, nums, dens, orders = (
        array("d"), array("q"), array("q"), array("b"),
    )
    for fv, num, den, order in entries:
        key = (round(fv, 14), order)
        if key == prev_key:
            continue
        prev_key = key
        values.append(fv)
        nums.append(num)
        dens.append(den)
        orders.append(order)
    return values, nums, dens, orders


def _get_abs_lookup() -> tuple:
    global _ABS_LOOKUP_CACHE
    if _ABS_LOOKUP_CACHE is None:
        _ABS_LOOKUP_CACHE = _build_abs_lookup()
    return _ABS_LOOKUP_CACHE


def _entry_to_sympy(num: int, den: int, order: int) -> sympy.Basic:
    r = Rational(num, den)
    if order == 1:
        return r
    return r ** Rational(1, order)


def _find_nearest_algebraic(target: float, digits: int) -> List[sympy.Basic]:
    """Binary-search the lookup table for algebraic numbers ≈ *target*."""
    if not math.isfinite(target):
        return []

    values, nums, dens, orders = _get_abs_lookup()
    if not values:
        return []

    abs_target = abs(target)
    sign = -1 if target < 0 else 1
    pos = bisect.bisect_left(values, abs_target)
    results: list[sympy.Basic] = []

    for i in range(max(0, pos - 1), min(len(values), pos + 2)):
        cval = values[i]
        if cval == 0 and abs_target > 0.01:
            continue
        if _precision_match(abs_target, cval, max(digits - 2, 2)):
            sym = _entry_to_sympy(nums[i], dens[i], orders[i])
            results.append(-sym if sign == -1 else sym)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Function evaluation over argument ranges
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_COMBOS = 80_000


def _to_sympy(a) -> sympy.Basic:
    if isinstance(a, Fraction):
        return Rational(a.numerator, a.denominator)
    if isinstance(a, int):
        return Integer(a)
    return sympy.sympify(a)


def _coerce_argument(a):
    """Exactify a user-supplied search argument; exact types pass through."""
    if isinstance(a, bool):
        raise FindClosedFormError(f"Invalid search argument {a!r}.")
    if isinstance(a, int):
        return Fraction(a)
    if isinstance(a, float):
        return Fraction(a).limit_denominator(10 ** 12)
    return a


def _slots_from_dict(d: dict, arity: int) -> List[list]:
    """
    Resolve a WL-style slots association (``<|#1 -> list1, #2 -> list2|>``)
    into per-slot argument lists.  Keys may be 1-based integers or ``"#1"``
    strings.
    """
    slots = []
    for i in range(1, arity + 1):
        for key in (i, f"#{i}", f"#{i}&"):
            if key in d:
                slots.append(list(d[key]))
                break
        else:
            raise FindClosedFormError(
                f"search_arguments dict is missing slot {i}: expected key "
                f"{i} or '#{i}'."
            )
    return slots


def _detect_arity(func: Callable) -> int:
    try:
        sig = inspect.signature(func)
        return sum(
            1
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except (ValueError, TypeError):
        return 1


def _is_identity_func(func: Callable) -> bool:
    try:
        x = sympy.Symbol("_x")
        return func(x) == x
    except Exception:
        return False


def _eval_func(func: Callable, args_sym: tuple, num_digits: int = 18):
    """Evaluate *func* on sympy args; return ``(symbolic, float)`` or ``None``."""
    try:
        val_sym = func(*args_sym)
    except Exception:
        return None
    val_f = _safe_neval(val_sym, num_digits)
    if val_f is None or not math.isfinite(val_f):
        return None
    return val_sym, val_f


def _eval_function_over_range(
    func: Callable,
    arg_ranges,
    arity: int,
    prev_args: set,
) -> List[Tuple[tuple, sympy.Basic, float]]:
    """
    Evaluate *func* over Cartesian-product argument combinations.

    *arg_ranges* is either a flat list (arity-1) or a list-of-lists (arity>1).
    """
    results: list[Tuple[tuple, sympy.Basic, float]] = []

    if arity == 1:
        for a in arg_ranges:
            key = (a,)
            if key in prev_args:
                continue
            r = _eval_func(func, (_to_sympy(a),))
            if r is not None:
                results.append((key, r[0], r[1]))
    else:
        if arg_ranges and isinstance(arg_ranges[0], (list, tuple)):
            per_slot = arg_ranges
        else:
            per_slot = [arg_ranges] * arity

        max_per = max(2, int(_MAX_COMBOS ** (1.0 / arity)))
        truncated = [
            r if len(r) <= max_per
            else sorted(r, key=lambda a: abs(float(a)))[:max_per]
            for r in per_slot
        ]

        for combo in itertools.product(*truncated):
            if combo in prev_args:
                continue
            combo_sym = tuple(_to_sympy(a) for a in combo)
            r = _eval_func(func, combo_sym)
            if r is not None:
                results.append((combo, r[0], r[1]))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-search rounds  (None / Times / Plus)
# ═══════════════════════════════════════════════════════════════════════════════

def _sub_search_none(
    num: float,
    evals: list,
    digits: int,
    compl: float,
    rational_solutions: bool,
    is_identity: bool,
) -> List[sympy.Basic]:
    results: list[sympy.Basic] = []
    for _, val_sym, val_f in evals:
        if not _precision_match(num, val_f, digits):
            continue
        if not rational_solutions and not is_identity and val_sym.is_Rational:
            continue
        if formula_complexity(val_sym) <= compl:
            results.append(val_sym)
    return results


def _sub_search_times(
    num: float,
    evals: list,
    digits: int,
    compl: float,
    rational_solutions: bool,
    is_identity: bool,
) -> List[sympy.Basic]:
    results: list[sympy.Basic] = []
    for _, val_sym, val_f in evals:
        if val_f == 0:
            continue
        ratio = num / val_f
        for alg in _find_nearest_algebraic(ratio, digits):
            candidate = alg * val_sym
            cand_f = _safe_neval(candidate)
            if cand_f is None:
                continue
            if not _precision_match(num, cand_f, digits):
                continue
            if not rational_solutions and not is_identity and candidate.is_Rational:
                continue
            if formula_complexity(candidate) <= compl:
                results.append(candidate)
    return results


def _sub_search_plus(
    num: float,
    evals: list,
    digits: int,
    compl: float,
    rational_solutions: bool,
    is_identity: bool,
) -> List[sympy.Basic]:
    results: list[sympy.Basic] = []
    for _, val_sym, val_f in evals:
        diff = num - val_f
        for alg in _find_nearest_algebraic(diff, digits):
            candidate = alg + val_sym
            cand_f = _safe_neval(candidate)
            if cand_f is None:
                continue
            if not _precision_match(num, cand_f, digits):
                continue
            if not rational_solutions and not is_identity and candidate.is_Rational:
                continue
            if formula_complexity(candidate) <= compl:
                results.append(candidate)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Default function list
# ═══════════════════════════════════════════════════════════════════════════════

def _make_default_functions() -> List[Tuple[str, Callable, int]]:
    """
    Default functional forms searched when no function is specified.

    Each entry is ``(name, callable, arity)``.
    Mirrors the WL default list: constants as powers, trig, hyp, special.
    """
    return [
        ("pi^#", lambda x: pi ** x, 1),
        ("EulerGamma^#", lambda x: EulerGamma ** x, 1),
        ("Catalan^#", lambda x: Catalan ** x, 1),
        ("GoldenRatio^#", lambda x: GoldenRatio ** x, 1),
        ("sin(pi*#)", lambda x: sin(pi * x), 1),
        ("cos(pi*#)", lambda x: cos(pi * x), 1),
        ("tan(pi*#)", lambda x: tan(pi * x), 1),
        ("asin(#)", lambda x: asin(x), 1),
        ("acos(#)", lambda x: acos(x), 1),
        ("atan(#)", lambda x: atan(x), 1),
        ("acot(#)", lambda x: acot(x), 1),
        ("log(#)", lambda x: log(x), 1),
        ("exp(#)", lambda x: exp(x), 1),
        ("sinh(#)", lambda x: sinh(x), 1),
        ("cosh(#)", lambda x: cosh(x), 1),
        ("tanh(#)", lambda x: tanh(x), 1),
        ("asinh(#)", lambda x: asinh(x), 1),
        ("acosh(#)", lambda x: acosh(x), 1),
        ("atanh(#)", lambda x: atanh(x), 1),
        ("acoth(#)", lambda x: acoth(x), 1),
        ("zeta(#)", lambda x: zeta(x), 1),
        ("gamma(#)", lambda x: spgamma(x), 1),
        ("polygamma(#)", lambda x: polygamma(0, x), 1),
        ("elliptic_k(#)", lambda x: elliptic_k(x), 1),
        ("elliptic_e(#)", lambda x: elliptic_e(x), 1),
        ("erf(#)", lambda x: erf(x), 1),
        ("erfc(#)", lambda x: erfc(x), 1),
        ("erfinv(#)", lambda x: erfinv(x), 1),
        ("airyai(#)", lambda x: airyai(x), 1),
        ("airybi(#)", lambda x: airybi(x), 1),
        ("Ei(#)", lambda x: Ei(x), 1),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication and sorting
# ═══════════════════════════════════════════════════════════════════════════════

def _dedup_results(results: List[sympy.Basic], digits: int) -> List[sympy.Basic]:
    """Remove structurally identical expressions (like WL ``Union``)."""
    seen: set = set()
    out: list = []
    for r in results:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main public API
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_functions(functions) -> List[Tuple[str, Callable, int]]:
    """Coerce *functions* to ``[(name, callable, arity), ...]``."""
    if functions is None:
        return _make_default_functions()

    if callable(functions) and not isinstance(functions, (list, tuple)):
        arity = _detect_arity(functions)
        return [("custom", functions, arity)]

    if isinstance(functions, (list, tuple)):
        result: list = []
        for item in functions:
            if callable(item):
                arity = _detect_arity(item)
                name = getattr(item, "__name__", "f")
                result.append((name, item, arity))
            elif isinstance(item, tuple):
                name = item[0] if isinstance(item[0], str) else "f"
                func = item[1] if len(item) > 1 else item[0]
                arity = _detect_arity(func)
                result.append((name, func, arity))
        return result

    return [("custom", functions, _detect_arity(functions))]


def find_closed_form(
    y: Number,
    functions=None,
    max_results: Optional[int] = None,
    *,
    significant_digits: Optional[int] = None,
    formula_complexity_threshold: Optional[float] = None,
    algebraic_factor: bool = True,
    algebraic_add: bool = True,
    rational_solutions: bool = False,
    max_search_rounds: int = 50,
    search_range: Union[str, Callable] = "Farey",
    search_range_fn: Optional[Callable] = None,
    search_range_options: Optional[dict] = None,
    search_arguments: Optional[Union[list, dict]] = None,
    search_time_limit: float = 3600,
    monitor_search: bool = False,
) -> Union[List[sympy.Basic], sympy.Basic, None]:
    """
    Search for closed-form expressions matching a numerical value.

    Positional forms mirror the WL resource function::

        find_closed_form(y)              # FindClosedForm[y]
        find_closed_form(y, n)           # FindClosedForm[y, n]
        find_closed_form(y, f)           # FindClosedForm[y, f]
        find_closed_form(y, [f1, f2])    # FindClosedForm[y, {f1, f2, ...}]
        find_closed_form(y, f, n)        # FindClosedForm[y, f, n]

    where an integer second argument is the number of results *n* and a
    callable (or list of callables) is the functional forms *f*.

    Parameters
    ----------
    y : number
        The target numerical value.
    functions : callable, list, or None
        Functional forms to search.  ``None`` uses ~31 common functions
        (plus the identity for the ``"Algebraic"`` and ``"Transcendental"``
        search ranges, whose elements are closed forms themselves).
        An integer here is dispatched to ``max_results`` (WL
        ``FindClosedForm[y, n]``).
    max_results : int
        Maximum number of results (default 1).
    significant_digits : int or None
        Precision target.  Auto-detected if None.
    formula_complexity_threshold : float or None
        Maximum allowed complexity.  Auto-scaled per round if None.
    algebraic_factor : bool
        Enable multiplicative algebraic combinations.
    algebraic_add : bool
        Enable additive algebraic combinations.
    rational_solutions : bool
        Allow purely rational results.
    max_search_rounds : int
        Maximum argument-range expansion rounds.
    search_range : str or callable
        Argument range per search round ``cut``:

        - ``"Farey"``: ``farey_range(-cut, cut, cut)``;
        - ``"Plain"``: rationals ``-cut, ..., cut`` with step ``1/cut``;
        - ``"Integer"``: the integers ``-cut, ..., cut``;
        - ``"Algebraic"``: ``algebraic_range(-cut, cut, 1/cut)`` — exact
          roots, from the algebraic-range package;
        - ``"Transcendental"``: ``transcendental_range(-cut, cut, 1/cut)``
          — exact transcendental numbers, from the transcendental-range
          package;
        - a callable ``f(cut) → list`` (same as ``search_range_fn``).
    search_range_fn : callable or None
        Custom ``f(cut) → list`` for argument generation
        (overrides ``search_range``).
    search_range_options : dict or None
        Extra keyword options forwarded to the range generator when
        ``search_range`` is ``"Algebraic"`` or ``"Transcendental"``,
        e.g. ``{"root_order": 3}`` or ``{"method": "log"}``.
    search_arguments : list, dict, or None
        Fixed argument values instead of auto-generated ranges; exact
        sympy expressions (e.g. an ``algebraic_range`` or
        ``transcendental_range`` output) are used as given.  A dict maps
        slots to per-slot lists (WL ``<|#1 -> list1, #2 -> list2|>``),
        with 1-based integer or ``"#1"`` keys.
    search_time_limit : float
        Maximum seconds for the search (WL ``TimeConstrained``: enforced
        by ``SIGALRM`` where available, else checked between functions).
    monitor_search : bool
        Print each result as it is found (WL ``"MonitorSearch"``).

    Returns
    -------
    Single sympy expression, list, or None.
    """
    # ── WL-style positional dispatch ─────────────────────────────────────
    # FindClosedForm[y, n]: an integer second argument is the number of
    # results, not the functions.
    if isinstance(functions, (int, sympy.Integer)) and not isinstance(functions, bool):
        if max_results is not None:
            raise FindClosedFormError(
                "with an integer second argument (the number of results, "
                "WL FindClosedForm[y, n]) no further positional argument "
                "is allowed."
            )
        functions, max_results = None, int(functions)
    if max_results is None:
        max_results = 1

    # ── normalise input ──────────────────────────────────────────────────
    exact_input = isinstance(y, (int, Fraction)) and not isinstance(y, bool)
    if isinstance(y, sympy.Basic):
        exact_input = bool(y.is_Rational)
        num = float(y.evalf(n=18))
    elif isinstance(y, Fraction):
        num = float(y)
    else:
        num = float(y)

    if not math.isfinite(num):
        raise FindClosedFormError(f"Input must be a finite number, got {y}")

    if search_range_options and search_range not in (
            "Algebraic", "Transcendental"):
        raise FindClosedFormError(
            "search_range_options only applies to the \"Algebraic\" and "
            f"\"Transcendental\" search ranges, not {search_range!r}."
        )

    if significant_digits is not None:
        digits = significant_digits
    elif exact_input and num != 0:
        digits = 16  # WL autoDigits: exact Integer/Rational input
    else:
        digits = _auto_digits(num)
    func_list = _normalize_functions(functions)
    # On the generated closed-form ranges the elements themselves are
    # candidate formulae: include the identity among the default functions.
    if functions is None and search_range in ("Algebraic", "Transcendental"):
        func_list = [("#", lambda x: x, 1)] + func_list

    eff_rational = rational_solutions
    if not algebraic_factor and not algebraic_add:
        eff_rational = True

    all_results: List[sympy.Basic] = []
    start_time = time.time()
    prev_args: dict[int, set] = {i: set() for i in range(len(func_list))}

    try:
        with _search_time_limit(search_time_limit):
            if search_arguments is not None:
                # ── fixed search-arguments mode (single pass, no rounds) ──
                if isinstance(search_arguments, dict):
                    flat_args = [a for sl in search_arguments.values() for a in sl]
                elif search_arguments and isinstance(
                        search_arguments[0], (list, tuple)):
                    flat_args = [a for sl in search_arguments for a in sl]
                else:
                    flat_args = list(search_arguments)
                args_compl = max(
                    (formula_complexity(_to_sympy(_coerce_argument(a)))
                     for a in flat_args),
                    default=0.0,
                )
                for fi, (name, func, arity) in enumerate(func_list):
                    if time.time() - start_time > search_time_limit:
                        break
                    is_identity = _is_identity_func(func)
                    if formula_complexity_threshold is not None:
                        compl = formula_complexity_threshold
                    else:
                        # WL complexityF over the custom argument list
                        compl = 0.5 * (1 + math.sqrt(arity)) * (5 + args_compl)

                    if arity == 1:
                        if isinstance(search_arguments, dict):
                            raw = _slots_from_dict(search_arguments, 1)[0]
                        else:
                            raw = search_arguments
                            if raw and isinstance(raw[0], (list, tuple)):
                                raw = raw[0]
                        frac_args = [_coerce_argument(a) for a in raw]
                        evals = _eval_function_over_range(func, frac_args, arity, set())
                    else:
                        if isinstance(search_arguments, dict):
                            slot_ranges = [
                                [_coerce_argument(a) for a in sl]
                                for sl in _slots_from_dict(search_arguments, arity)
                            ]
                        elif search_arguments and isinstance(search_arguments[0], (list, tuple)):
                            slot_ranges = [
                                [_coerce_argument(a) for a in sl]
                                for sl in search_arguments
                            ]
                        else:
                            slot_ranges = [
                                [_coerce_argument(a) for a in search_arguments]
                            ] * arity
                        evals = _eval_function_over_range(func, slot_ranges, arity, set())

                    # WL searchRound: skip remaining operations once enough
                    # results were already found
                    rr = _sub_search_none(num, evals, digits, compl, eff_rational, is_identity)
                    if algebraic_factor and len(rr) < max_results:
                        rr.extend(_sub_search_times(num, evals, digits, compl, eff_rational, is_identity))
                    if algebraic_add and len(rr) < max_results:
                        rr.extend(_sub_search_plus(num, evals, digits, compl, eff_rational, is_identity))
                    prev_count = len(all_results)
                    all_results.extend(rr)
                    all_results = _dedup_results(all_results, digits)
                    if monitor_search:
                        for k, r in enumerate(all_results[prev_count:],
                                              start=prev_count + 1):
                            print(f"Search result {k}: {r}")

            else:
                # ── main search loop over progressively larger ranges ──
                for cut in range(1, max_search_rounds + 1):
                    if time.time() - start_time > search_time_limit:
                        break

                    arg_range = _generate_range(
                        cut, search_range, search_range_fn,
                        search_range_options)
                    if not arg_range:
                        continue
                    range_compl = None  # computed lazily, once per round

                    for fi, (name, func, arity) in enumerate(func_list):
                        if time.time() - start_time > search_time_limit:
                            break

                        if formula_complexity_threshold is not None:
                            compl = formula_complexity_threshold
                        else:
                            # WL: formulaComplexity[range] = Max over elements;
                            # constant 15 lowered to 5 in kernel 1.0.0.4
                            if range_compl is None:
                                range_compl = max(
                                    formula_complexity(_to_sympy(a))
                                    for a in arg_range
                                )
                            compl = 0.5 * (1 + math.sqrt(arity)) * (5 + range_compl)

                        is_identity = _is_identity_func(func)

                        if arity == 1:
                            evals = _eval_function_over_range(func, arg_range, arity, prev_args[fi])
                        else:
                            evals = _eval_function_over_range(
                                func, [arg_range] * arity, arity, prev_args[fi],
                            )

                        for args_tuple, _, _ in evals:
                            prev_args[fi].add(args_tuple)

                        # WL searchRound: skip remaining operations once
                        # enough results were already found in this round
                        rr = _sub_search_none(num, evals, digits, compl, eff_rational, is_identity)
                        if algebraic_factor and len(rr) < max_results:
                            rr.extend(_sub_search_times(num, evals, digits, compl, eff_rational, is_identity))
                        if algebraic_add and len(rr) < max_results:
                            rr.extend(_sub_search_plus(num, evals, digits, compl, eff_rational, is_identity))

                        prev_count = len(all_results)
                        all_results.extend(rr)
                        all_results = _dedup_results(all_results, digits)
                        if monitor_search:
                            for k, r in enumerate(all_results[prev_count:],
                                                  start=prev_count + 1):
                                print(f"Search result {k}: {r}")
                        if len(all_results) >= max_results:
                            break

                    if len(all_results) >= max_results:
                        break
    except _SearchAborted:
        pass  # time limit hit: return what was found so far (WL CheckAbort)

    all_results = _dedup_results(all_results, digits)
    all_results.sort(key=formula_complexity)
    all_results = all_results[:max_results]

    if max_results == 1:
        return all_results[0] if all_results else None
    return all_results
