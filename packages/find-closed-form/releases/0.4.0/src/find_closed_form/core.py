"""
find_closed_form — Find closed-form expressions for numerical values.

A Python translation of the Wolfram Language ResourceFunction ``FindClosedForm``
originally contributed by Daniele Gregori.

The algorithm searches for closed-form mathematical expressions that match a
given numerical value by evaluating candidate functions over Farey-based
argument ranges and testing digit-level agreement.
"""

from __future__ import annotations

import bisect
import inspect
import itertools
import math
import time
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
            deg = max(abs(int(ex.p)), abs(int(ex.q)))
            return _collect_integers(base, mult * max(deg, 1))
        if ex.is_Integer and abs(int(ex)) > 1:
            return _collect_integers(base, mult * abs(int(ex)))

    result: list[int] = []
    for a in expr.args:
        result.extend(_collect_integers(a, mult))
    return result


def formula_complexity(expr) -> float:
    """
    Heuristic complexity of a sympy expression.

    For each integer in the expression: ``mean(5*IntegerLength, DigitSum, sqrt(|n|))``.
    Sum over all integers.  Constants (pi, E, …) count as 1.
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
        ai = abs(i) if i != 0 else 1
        total += (5 * _integer_length(ai) + _digit_sum(ai) + math.sqrt(ai)) / 3.0
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

def _generate_range(
    cut: int, search_range: str, custom_fn: Optional[Callable] = None,
) -> list:
    if custom_fn is not None:
        return custom_fn(cut)
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
    return _ext_farey_range(-cut, cut, cut)


# ═══════════════════════════════════════════════════════════════════════════════
# Algebraic number lookup table  (modifiedRootApproximant)
#
# Mirrors the Wolfram ``absLookupBase``.  Positive reals only; sign handled
# separately.  Built lazily on first use.
# ═══════════════════════════════════════════════════════════════════════════════

_ABS_LOOKUP_CACHE: Optional[Tuple[list, list, list]] = None


def _build_abs_lookup() -> Tuple[List[float], List[Fraction], List[int]]:
    """
    Build a sorted lookup table of positive algebraic-number candidates.

    Returns three parallel lists (sorted by float value):
      values  – float approximations
      bases   – Fraction bases (p/q)
      orders  – root order (1 = plain fraction, 2 = sqrt, …)
    """
    seen: set[Tuple[float, int]] = set()
    entries: list[Tuple[float, Fraction, int]] = []

    def _add(fv: float, base: Fraction, order: int) -> None:
        key = (round(fv, 14), order)
        if key not in seen and math.isfinite(fv) and fv >= 0:
            seen.add(key)
            entries.append((fv, base, order))

    # 1) Range[0, 10000, 1/10]  – tenths
    for n in range(0, 100_001):
        f = Fraction(n, 10)
        _add(float(f), f, 1)

    # 2) FareyRange[0, 1000, 10]  – simple fractions up to 1000
    for f in _ext_farey_range(0, 1000, 10):
        _add(float(f), f, 1)

    # 3) Sqrt of FareyRange[0, 100, 50]
    for f in _ext_farey_range(0, 100, 50):
        if f >= 0:
            _add(math.sqrt(float(f)), f, 2)

    # 4) Higher roots of FareyRange[0, 100, 10]
    for f in _ext_farey_range(0, 100, 10):
        if f > 0:
            fv = float(f)
            for order in (3, 4, 5, 6):
                _add(fv ** (1.0 / order), f, order)

    entries.sort(key=lambda e: e[0])
    return (
        [e[0] for e in entries],
        [e[1] for e in entries],
        [e[2] for e in entries],
    )


def _get_abs_lookup() -> Tuple[List[float], List[Fraction], List[int]]:
    global _ABS_LOOKUP_CACHE
    if _ABS_LOOKUP_CACHE is None:
        _ABS_LOOKUP_CACHE = _build_abs_lookup()
    return _ABS_LOOKUP_CACHE


def _entry_to_sympy(base: Fraction, order: int) -> sympy.Basic:
    r = Rational(base.numerator, base.denominator)
    if order == 1:
        return r
    return r ** Rational(1, order)


def _find_nearest_algebraic(target: float, digits: int) -> List[sympy.Basic]:
    """Binary-search the lookup table for algebraic numbers ≈ *target*."""
    if not math.isfinite(target):
        return []

    values, bases, orders = _get_abs_lookup()
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
            sym = _entry_to_sympy(bases[i], orders[i])
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
        truncated = [r[:max_per] for r in per_slot]

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
        ("erfinv(#)", lambda x: erfinv(x), 1),
        ("airyai(#)", lambda x: airyai(x), 1),
        ("airybi(#)", lambda x: airybi(x), 1),
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
    max_results: int = 1,
    *,
    significant_digits: Optional[int] = None,
    formula_complexity_threshold: Optional[float] = None,
    algebraic_factor: bool = True,
    algebraic_add: bool = True,
    rational_solutions: bool = False,
    max_search_rounds: int = 50,
    search_range: str = "Farey",
    search_range_fn: Optional[Callable] = None,
    search_arguments: Optional[Union[list, dict]] = None,
    search_time_limit: float = 3600,
) -> Union[List[sympy.Basic], sympy.Basic, None]:
    """
    Search for closed-form expressions matching a numerical value.

    Parameters
    ----------
    y : number
        The target numerical value.
    functions : callable, list, or None
        Functional forms to search.  ``None`` uses ~29 common functions.
    max_results : int
        Maximum number of results.
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
    search_range : str
        ``"Farey"``, ``"Plain"``, or ``"Integer"``.
    search_range_fn : callable or None
        Custom ``f(cut) → list`` for argument generation.
    search_arguments : list, dict, or None
        Fixed argument values instead of auto-generated ranges.
    search_time_limit : float
        Maximum seconds for the search.

    Returns
    -------
    Single sympy expression, list, or None.
    """
    # ── normalise input ──────────────────────────────────────────────────
    if isinstance(y, sympy.Basic):
        num = float(y.evalf(n=18))
    elif isinstance(y, Fraction):
        num = float(y)
    else:
        num = float(y)

    if not math.isfinite(num):
        raise FindClosedFormError(f"Input must be a finite number, got {y}")

    digits = significant_digits if significant_digits is not None else _auto_digits(num)
    func_list = _normalize_functions(functions)

    eff_rational = rational_solutions
    if not algebraic_factor and not algebraic_add:
        eff_rational = True

    all_results: List[sympy.Basic] = []
    start_time = time.time()
    prev_args: dict[int, set] = {i: set() for i in range(len(func_list))}

    # ── fixed search-arguments mode (single pass, no rounds) ─────────
    if search_arguments is not None:
        for fi, (name, func, arity) in enumerate(func_list):
            if time.time() - start_time > search_time_limit:
                break
            is_identity = _is_identity_func(func)
            compl = formula_complexity_threshold if formula_complexity_threshold else 50.0

            if arity == 1:
                raw = search_arguments
                if raw and isinstance(raw[0], (list, tuple)):
                    raw = raw[0]
                frac_args = [Fraction(a) if isinstance(a, (int, float)) else a for a in raw]
                evals = _eval_function_over_range(func, frac_args, arity, set())
            else:
                if isinstance(search_arguments, (list, tuple)):
                    if search_arguments and isinstance(search_arguments[0], (list, tuple)):
                        slot_ranges = [
                            [Fraction(a) if isinstance(a, (int, float)) else a for a in sl]
                            for sl in search_arguments
                        ]
                    else:
                        slot_ranges = [
                            [Fraction(a) if isinstance(a, (int, float)) else a for a in search_arguments]
                        ] * arity
                else:
                    slot_ranges = [search_arguments] * arity
                evals = _eval_function_over_range(func, slot_ranges, arity, set())

            rr = _sub_search_none(num, evals, digits, compl, eff_rational, is_identity)
            if algebraic_factor:
                rr.extend(_sub_search_times(num, evals, digits, compl, eff_rational, is_identity))
            if algebraic_add:
                rr.extend(_sub_search_plus(num, evals, digits, compl, eff_rational, is_identity))
            all_results.extend(rr)

        all_results = _dedup_results(all_results, digits)
        all_results.sort(key=formula_complexity)
        all_results = all_results[:max_results]
        if max_results == 1:
            return all_results[0] if all_results else None
        return all_results

    # ── main search loop over progressively larger argument ranges ────
    for cut in range(1, max_search_rounds + 1):
        if time.time() - start_time > search_time_limit:
            break

        arg_range = _generate_range(cut, search_range, search_range_fn)

        for fi, (name, func, arity) in enumerate(func_list):
            if time.time() - start_time > search_time_limit:
                break

            if formula_complexity_threshold is not None:
                compl = formula_complexity_threshold
            else:
                max_elem = max(arg_range, key=lambda a: abs(float(a)))
                range_compl = formula_complexity(_to_sympy(max_elem))
                compl = 0.5 * (1 + math.sqrt(arity)) * (15 + range_compl)

            is_identity = _is_identity_func(func)

            if arity == 1:
                evals = _eval_function_over_range(func, arg_range, arity, prev_args[fi])
            else:
                evals = _eval_function_over_range(
                    func, [arg_range] * arity, arity, prev_args[fi],
                )

            for args_tuple, _, _ in evals:
                prev_args[fi].add(args_tuple)

            rr = _sub_search_none(num, evals, digits, compl, eff_rational, is_identity)
            if algebraic_factor:
                rr.extend(_sub_search_times(num, evals, digits, compl, eff_rational, is_identity))
            if algebraic_add:
                rr.extend(_sub_search_plus(num, evals, digits, compl, eff_rational, is_identity))

            all_results.extend(rr)

        all_results = _dedup_results(all_results, digits)
        if len(all_results) >= max_results:
            break

    all_results.sort(key=formula_complexity)
    all_results = all_results[:max_results]

    if max_results == 1:
        return all_results[0] if all_results else None
    return all_results
