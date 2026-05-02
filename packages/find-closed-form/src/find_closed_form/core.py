"""
find_closed_form — Find closed-form expressions for numerical values.

A Python translation of the Wolfram Language ResourceFunction ``FindClosedForm``
originally contributed by Daniele Gregori.

The algorithm searches for closed-form mathematical expressions that match a
given numerical value by evaluating candidate functions over Farey-based
argument ranges and testing digit-level agreement. Complexity is scored using
the same heuristic as the WL ``AlgebraicRange`` resource function.
"""

from __future__ import annotations

import bisect
import inspect
import itertools
import math
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import sympy
from sympy import (
    Integer,
    Rational,
    Pow,
    S,
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
    asin,
    acos,
    atan,
    acot,
    sinh,
    cosh,
    tanh,
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
    nsimplify,
    factorint,
)

__all__ = ["find_closed_form", "formula_complexity", "farey_range",
           "FindClosedFormError"]

Number = Union[int, float, Fraction, sympy.Basic]


# ── Exceptions ──────────────────────────────────────────────────────────────

class FindClosedFormError(Exception):
    """Base exception for find_closed_form errors."""


# ═══════════════════════════════════════════════════════════════════════════
# Farey range (self-contained, no dependency on the ``farey`` package)
# ═══════════════════════════════════════════════════════════════════════════

def _farey_sequence(n: int) -> List[Fraction]:
    """Farey sequence F_n (fractions in [0, 1] with denominator <= n)."""
    result: set[Fraction] = set()
    for d in range(1, n + 1):
        for k in range(0, d + 1):
            result.add(Fraction(k, d))
    return sorted(result)


def farey_range(
    start: Union[int, float, Fraction],
    end: Union[int, float, Fraction],
    step: Union[int, float, Fraction, None] = None,
) -> List[Union[Fraction, float]]:
    """
    Generate a Farey range over the interval [start, end].

    Parameters
    ----------
    start, end : number
        Interval endpoints.
    step : int, float, Fraction, or None
        Positive integer *n* or ``Fraction(1, n)`` → Farey order *n*.
        Negative → descending. ``None`` → order 1.
    """
    if step is None:
        order, reverse = 1, False
    else:
        if isinstance(step, float):
            step = Fraction(step).limit_denominator(10 ** 9)
        if isinstance(step, Fraction):
            if step == 0:
                raise ValueError("step must be nonzero")
            reverse = step < 0
            astep = abs(step)
            if astep.denominator == 1:
                order = int(astep)
            elif astep.numerator == 1:
                order = astep.denominator
            else:
                order = max(int(astep.numerator), int(astep.denominator))
        elif isinstance(step, int):
            if step == 0:
                raise ValueError("step must be nonzero")
            order, reverse = abs(step), step < 0
        else:
            order, reverse = int(abs(step)), float(step) < 0

    farey = _farey_sequence(order)
    mn, mx = min(start, end), max(start, end)
    span = float(mx) - float(mn)
    num_units = int(span)

    if abs(span - num_units) > 1e-15:
        result = sorted(set(float(mn) + float(f) * span for f in farey))
        return list(reversed(result)) if reverse else result

    all_values: set[float] = set()
    for unit in range(num_units):
        base = float(mn) + unit
        for f in farey:
            all_values.add(base + float(f))
    all_values.add(float(mx))
    result_f = sorted(all_values)

    if all(isinstance(x, (int, Fraction)) or float(x) == int(float(x))
           for x in [start, end]):
        frac_result: List[Union[Fraction, float]] = []
        for val in result_f:
            frac = Fraction(val).limit_denominator(1_000_000)
            frac_result.append(frac if abs(float(frac) - val) < 1e-14 else val)
        result_out = frac_result
    else:
        result_out = result_f  # type: ignore[assignment]

    return list(reversed(result_out)) if reverse else result_out


# ═══════════════════════════════════════════════════════════════════════════
# Formula complexity (same heuristic as algebraic-range)
# ═══════════════════════════════════════════════════════════════════════════

def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _int_complexity(n: int) -> float:
    """Per-integer complexity: 0.5 * mean(DigitSum, 5*Len, #PrimeFactors, sqrt)."""
    if n <= 0:
        n = abs(n) + 1
    ds = _digit_sum(n)
    il = len(str(n))
    pf = sum(e for _, e in factorint(n).items()) if n > 1 else 0
    sq = math.sqrt(n)
    return 0.5 * (ds + 5 * il + pf + sq) / 4.0


def _collect_integers(expr: sympy.Basic, acc: list, mult: int = 1):
    if expr.is_Integer:
        acc.extend([int(expr)] * mult)
        return
    if expr.is_Rational and not expr.is_Integer:
        acc.extend([int(expr.p)] * mult)
        acc.extend([int(expr.q)] * mult)
        return
    if expr.func == Pow and len(expr.args) == 2:
        base, ex = expr.args
        if ex.is_Rational and not ex.is_Integer:
            deg = max(int(abs(ex.p)), int(abs(ex.q)))
            _collect_integers(base, acc, mult * deg)
            return
        if ex.is_Integer:
            _collect_integers(base, acc, mult * int(abs(ex)))
            return
    for a in expr.args:
        _collect_integers(a, acc, mult)


def formula_complexity(expr: Number) -> float:
    """
    Compute the heuristic formula complexity of an algebraic expression.

    Complexity is the sum of per-integer complexities for every integer
    appearing in the expression (accounting for root orders as multipliers).
    """
    if isinstance(expr, (int, float, Fraction)):
        expr = sympy.sympify(expr)
    ints: list[int] = []
    _collect_integers(expr, ints)
    positives = [abs(n) + 1 if n <= 0 else n for n in ints]
    return sum(_int_complexity(p) for p in positives) if positives else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Core find_closed_form algorithm
# ═══════════════════════════════════════════════════════════════════════════

# ── Arity detection ─────────────────────────────────────────────────────────

def _func_arity(func: Callable) -> int:
    """Detect the number of positional parameters of a callable."""
    try:
        sig = inspect.signature(func)
        return sum(
            1 for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except (ValueError, TypeError):
        return 1


# ── Default functional forms ────────────────────────────────────────────────

def _make_default_functions() -> List[Tuple[str, Callable, Optional[Callable]]]:
    """
    Return default functional forms as (name, sympy_func, domain_filter).

    domain_filter receives the same arguments as the function (one value
    per slot) and returns True if the argument combination is valid.
    """
    _0to1 = lambda x: 0 <= float(x) <= 1
    _0toHalf = lambda x: 0 <= float(x) <= 0.5
    _ge0 = lambda x: float(x) >= 0
    _ge1 = lambda x: float(x) >= 1
    _01open = lambda x: 0 < float(x) < 1

    return [
        # Constants as power bases
        ("pi^#", lambda x: pi ** x, None),
        ("E^#", lambda x: E ** x, None),
        ("EulerGamma^#", lambda x: EulerGamma ** x, None),
        ("Catalan^#", lambda x: Catalan ** x, None),
        ("GoldenRatio^#", lambda x: GoldenRatio ** x, None),
        # Trig (argument multiplied by pi)
        ("sin(pi*#)", lambda x: sin(pi * x), _0toHalf),
        ("cos(pi*#)", lambda x: cos(pi * x), _0toHalf),
        ("tan(pi*#)", lambda x: tan(pi * x), _0toHalf),
        # Inverse trig
        ("asin(#)", lambda x: asin(x), _0to1),
        ("acos(#)", lambda x: acos(x), _0to1),
        ("atan(#)", lambda x: atan(x), _ge0),
        ("acot(#)", lambda x: acot(x), _ge0),
        # Exp / Log
        ("log(#)", lambda x: log(x), _ge1),
        ("exp(#)", lambda x: exp(x), None),
        # Hyperbolic
        ("sinh(#)", lambda x: sinh(x), _ge0),
        ("cosh(#)", lambda x: cosh(x), _ge0),
        ("tanh(#)", lambda x: tanh(x), _ge0),
        # Inverse hyperbolic
        ("asinh(#)", lambda x: asinh(x), _ge0),
        ("acosh(#)", lambda x: acosh(x), _ge1),
        ("atanh(#)", lambda x: atanh(x), _01open),
        # Special functions
        ("zeta(#)", lambda x: zeta(x), None),
        ("gamma(#)", lambda x: spgamma(x), _0to1),
        ("polygamma(#)", lambda x: polygamma(0, x), _0to1),
        ("elliptic_k(#)", lambda x: elliptic_k(x), None),
        ("elliptic_e(#)", lambda x: elliptic_e(x), None),
        ("erf(#)", lambda x: erf(x), _ge0),
        ("erfinv(#)", lambda x: erfinv(x), _01open),
    ]


# ── Precision helpers ───────────────────────────────────────────────────────

def _significant_digits(num: float) -> int:
    """Auto-detect significant digits of a float, at least 2."""
    if num == 0:
        return 1
    s = f"{num:.17g}"
    s = s.lstrip("-").lstrip("0").replace(".", "")
    if "." in f"{num:.17g}":
        s = s.rstrip("0")
    return max(len(s), 2)


def _precision_match(num: float, candidate: float, digits: int) -> bool:
    """Check whether candidate matches num to the required precision."""
    if num == 0:
        return abs(candidate) <= 10 ** (-digits + 1)
    return abs(1 - candidate / num) <= 10 ** (-digits + 1)


def _safe_neval(expr, digits: int = 18) -> Optional[float]:
    """Numerically evaluate a sympy expression, returning None on failure."""
    try:
        val = complex(expr.evalf(n=digits))
        if abs(val.imag) > 1e-15 * max(abs(val.real), 1):
            return None
        return val.real
    except Exception:
        return None


def _to_sympy_arg(a) -> sympy.Basic:
    """Convert a single argument value to a sympy expression."""
    if isinstance(a, Fraction):
        return Rational(a.numerator, a.denominator)
    if isinstance(a, (int, float)):
        return nsimplify(a, rational=True)
    return a


# ── Algebraic lookup table ──────────────────────────────────────────────────

def _build_abs_lookup(max_int: int = 200, farey_order: int = 10,
                      root_orders: Sequence[int] = (2, 3, 4, 5, 6),
                      ) -> Tuple[List[sympy.Basic], List[float]]:
    """
    Build a sorted lookup table of positive algebraic numbers and their
    float approximations, for use as algebraic factors/addends.
    """
    candidates: set = set()

    for n in range(0, max_int + 1):
        candidates.add(Integer(n))

    fr = farey_range(0, min(max_int, 100), min(farey_order, 10))
    for f in fr:
        candidates.add(Rational(f.numerator, f.denominator) if isinstance(f, Fraction)
                        else nsimplify(f, rational=True))

    fr_small = farey_range(0, min(max_int, 50), min(farey_order, 8))
    for f in fr_small:
        val = Rational(f.numerator, f.denominator) if isinstance(f, Fraction) else nsimplify(f, rational=True)
        if val < 0:
            continue
        for order in root_orders:
            candidates.add(val ** Rational(1, order))

    table_sym: List[sympy.Basic] = []
    table_num: List[float] = []
    for c in candidates:
        fv = _safe_neval(c)
        if fv is not None and fv >= 0 and math.isfinite(fv):
            table_sym.append(c)
            table_num.append(fv)

    pairs = sorted(zip(table_num, table_sym))
    return [p[1] for p in pairs], [p[0] for p in pairs]


_ABS_LOOKUP: Optional[Tuple[List[sympy.Basic], List[float]]] = None


def _get_abs_lookup() -> Tuple[List[sympy.Basic], List[float]]:
    global _ABS_LOOKUP
    if _ABS_LOOKUP is None:
        _ABS_LOOKUP = _build_abs_lookup()
    return _ABS_LOOKUP


def _nearest_algebraic(target: float) -> List[sympy.Basic]:
    """Find algebraic numbers in the lookup table closest to target."""
    if not math.isfinite(target):
        return []
    sym_table, num_table = _get_abs_lookup()
    if not num_table:
        return []

    abs_target = abs(target)
    sign = 1 if target >= 0 else -1

    pos = bisect.bisect_left(num_table, abs_target)
    candidates = []
    for i in range(max(0, pos - 1), min(len(num_table), pos + 2)):
        candidates.append(sym_table[i] * sign if sign == -1 else sym_table[i])
    return candidates


def _modified_root_approximant(ratio: float, digits: int) -> List[sympy.Basic]:
    """Approximate a real number by a simple algebraic number."""
    if not math.isfinite(ratio):
        return []
    candidates = _nearest_algebraic(ratio)
    result = []
    for c in candidates:
        cv = _safe_neval(c)
        if cv is not None and _precision_match(ratio, cv, max(digits - 2, 2)):
            result.append(c)
    return result


# ── Argument range generation ───────────────────────────────────────────────

def _generate_arg_range(cutoff: int) -> List[Fraction]:
    """Generate argument search range for a given cutoff (round number)."""
    return farey_range(-cutoff, cutoff, cutoff)


# ── Function evaluation (supports any arity) ───────────────────────────────

_MAX_MULTI_ARG_COMBOS = 50_000  # safety cap on cartesian product size


def _eval_function_over_args(
    func: Callable,
    args: List,
    domain_filter: Optional[Callable],
    arity: int = 1,
) -> List[Tuple[tuple, float, sympy.Basic]]:
    """
    Evaluate *func* over argument combinations, filtering by domain.

    For arity == 1, iterates over *args* directly.
    For arity > 1, iterates over the cartesian product args^arity.

    Returns list of (arg_tuple, value_float, value_sympy) where arg_tuple
    is a tuple of sympy expressions (length == arity).
    """
    if arity == 1:
        return _eval_single_arg(func, args, domain_filter)
    return _eval_multi_arg(func, args, domain_filter, arity)


def _eval_single_arg(func, args, domain_filter):
    """Fast path for single-argument functions."""
    results = []
    for a in args:
        try:
            af = float(a)
        except (ValueError, TypeError):
            continue
        if domain_filter is not None:
            try:
                if not domain_filter(af):
                    continue
            except Exception:
                continue

        arg_sym = _to_sympy_arg(a)
        try:
            val_sym = func(arg_sym)
        except Exception:
            continue

        val_f = _safe_neval(val_sym)
        if val_f is None or not math.isfinite(val_f) or val_f == 0:
            continue

        results.append(((arg_sym,), val_f, val_sym))
    return results


def _eval_multi_arg(func, args, domain_filter, arity):
    """Evaluate a multi-argument function over the cartesian product."""
    # Limit combinatorial explosion
    max_per_dim = max(1, int(_MAX_MULTI_ARG_COMBOS ** (1.0 / arity)))
    truncated = args[:max_per_dim]

    results = []
    for combo in itertools.product(truncated, repeat=arity):
        # Convert to floats for domain filter
        try:
            combo_f = tuple(float(a) for a in combo)
        except (ValueError, TypeError):
            continue

        if domain_filter is not None:
            try:
                if not domain_filter(*combo_f):
                    continue
            except Exception:
                continue

        combo_sym = tuple(_to_sympy_arg(a) for a in combo)
        try:
            val_sym = func(*combo_sym)
        except Exception:
            continue

        val_f = _safe_neval(val_sym)
        if val_f is None or not math.isfinite(val_f) or val_f == 0:
            continue

        results.append((combo_sym, val_f, val_sym))
    return results


# ── Sub-search rounds ──────────────────────────────────────────────────────

def _sub_search_none(num, func, args, domain_filter, digits,
                     complexity_threshold, arity):
    """Direct matching: find args where func(args...) == num."""
    evals = _eval_function_over_args(func, args, domain_filter, arity)
    results = []
    for _, val_f, val_sym in evals:
        if _precision_match(num, val_f, digits):
            fc = formula_complexity(val_sym)
            if fc <= complexity_threshold:
                results.append(val_sym)
    return results


def _sub_search_times(num, func, args, domain_filter, digits,
                      complexity_threshold, arity):
    """Multiplicative search: find a * func(args...) == num."""
    evals = _eval_function_over_args(func, args, domain_filter, arity)
    results = []
    for _, val_f, val_sym in evals:
        if val_f == 0:
            continue
        ratio = num / val_f
        algebraics = _modified_root_approximant(ratio, digits)
        for alg in algebraics:
            candidate = alg * val_sym
            cand_f = _safe_neval(candidate)
            if cand_f is not None and _precision_match(num, cand_f, digits):
                fc = formula_complexity(candidate)
                if fc <= complexity_threshold:
                    results.append(candidate)
    return results


def _sub_search_plus(num, func, args, domain_filter, digits,
                     complexity_threshold, arity):
    """Additive search: find a + func(args...) == num."""
    evals = _eval_function_over_args(func, args, domain_filter, arity)
    results = []
    for _, val_f, val_sym in evals:
        diff = num - val_f
        algebraics = _modified_root_approximant(diff, digits)
        for alg in algebraics:
            candidate = alg + val_sym
            cand_f = _safe_neval(candidate)
            if cand_f is not None and _precision_match(num, cand_f, digits):
                fc = formula_complexity(candidate)
                if fc <= complexity_threshold:
                    results.append(candidate)
    return results


# ── Search round ────────────────────────────────────────────────────────────

def _search_round(num, func, domain_filter, args, digits,
                  complexity_threshold, algebraic_factor, algebraic_add,
                  max_results, arity):
    """Execute one search round: None, then Times, then Plus."""
    results = _sub_search_none(
        num, func, args, domain_filter, digits, complexity_threshold, arity)
    if len(results) >= max_results:
        return results

    if algebraic_factor:
        res_t = _sub_search_times(
            num, func, args, domain_filter, digits, complexity_threshold, arity)
        results.extend(res_t)
        if len(results) >= max_results:
            return results

    if algebraic_add:
        res_p = _sub_search_plus(
            num, func, args, domain_filter, digits, complexity_threshold, arity)
        results.extend(res_p)

    return results


# ── Deduplication ───────────────────────────────────────────────────────────

def _dedup_results(results: List[sympy.Basic], digits: int) -> List[sympy.Basic]:
    """Remove results that evaluate to the same number."""
    seen: Dict[float, sympy.Basic] = {}
    for r in results:
        fv = _safe_neval(r)
        if fv is None:
            continue
        key = round(fv, digits - 1)
        if key not in seen or formula_complexity(r) < formula_complexity(seen[key]):
            seen[key] = r
    return list(seen.values())


# ── Main public API ─────────────────────────────────────────────────────────

def find_closed_form(
    y: Number,
    functions: Optional[Union[
        Callable,
        List[Callable],
        List[Tuple[str, Callable]],
        List[Tuple[str, Callable, Optional[Callable]]],
    ]] = None,
    max_results: int = 1,
    *,
    significant_digits: Optional[int] = None,
    formula_complexity_threshold: Optional[float] = None,
    algebraic_factor: bool = True,
    algebraic_add: bool = True,
    max_search_rounds: int = 50,
) -> List[sympy.Basic]:
    """
    Search for closed-form expressions matching a numerical value.

    Parameters
    ----------
    y : number
        The target numerical value.
    functions : callable, list, or None
        Functional forms to search over. Each element can be:

        - A bare callable (arity auto-detected via ``inspect``)::

            lambda x: sin(pi * x)
            lambda x, y: x * log(y)

        - A tuple ``(name, callable)``
        - A tuple ``(name, callable, domain_filter)``

        Multi-argument functions are supported: the search evaluates
        the cartesian product of the argument range for each slot.

        If None, searches over ~27 common single-argument functions.

    max_results : int, default 1
        Maximum number of results to return.
    significant_digits : int or None
        Precision target for matching. Auto-detected from y if None.
    formula_complexity_threshold : float or None
        Maximum allowed complexity. Auto-scaled if None.
    algebraic_factor : bool, default True
        Enable multiplicative algebraic combinations ``a * f(b, ...)``.
    algebraic_add : bool, default True
        Enable additive algebraic combinations ``a + f(b, ...)``.
    max_search_rounds : int, default 50
        Maximum number of argument-range expansion rounds.

    Returns
    -------
    list[sympy.Expr]
        Closed-form expressions sorted by complexity.

    Examples
    --------
    >>> from find_closed_form import find_closed_form
    >>> find_closed_form(0.7071067811865476)  # sqrt(2)/2
    [sqrt(2)/2]

    >>> from sympy import sin, pi
    >>> find_closed_form(0.5, functions=lambda x: sin(pi * x))
    [sin(pi/6)]

    Multi-argument:

    >>> from sympy import log
    >>> find_closed_form(1.6094379124341003,
    ...                  functions=lambda x, y: x * log(y))
    [log(5)]
    """
    if isinstance(y, sympy.Basic):
        num = float(y.evalf(n=18))
    elif isinstance(y, Fraction):
        num = float(y)
    else:
        num = float(y)

    if not math.isfinite(num):
        raise FindClosedFormError(f"Input must be a finite number, got {y}")

    if significant_digits is not None:
        digits = significant_digits
    else:
        digits = _significant_digits(num)

    # ── Normalise function list with arity ─────────────────────────────
    # Each entry: (name, callable, domain_filter, arity)
    func_list: List[Tuple[str, Callable, Optional[Callable], int]] = []
    if functions is None:
        for name, fn, dom in _make_default_functions():
            func_list.append((name, fn, dom, 1))
    elif callable(functions) and not isinstance(functions, list):
        arity = _func_arity(functions)
        func_list.append(("custom", functions, None, arity))
    else:
        for item in functions:
            if callable(item):
                arity = _func_arity(item)
                func_list.append(
                    (getattr(item, "__name__", "f"), item, None, arity))
            elif isinstance(item, tuple):
                fn = item[1]
                arity = _func_arity(fn)
                dom = item[2] if len(item) >= 3 else None
                func_list.append((item[0], fn, dom, arity))

    all_results: List[sympy.Basic] = []

    for cutoff in range(1, max_search_rounds + 1):
        arg_range = _generate_arg_range(cutoff)

        for name, func, domain, arity in func_list:
            # Dynamic complexity threshold for this round
            if formula_complexity_threshold is not None:
                compl = formula_complexity_threshold
            else:
                range_compl = formula_complexity(Integer(cutoff))
                # Scale with sqrt(arity), matching WL:
                #   0.5 * (1 + sqrt(n_slots)) * (15 + range_complexity)
                compl = 0.5 * (1 + math.sqrt(arity)) * (15 + range_compl)

            round_results = _search_round(
                num, func, domain, arg_range, digits, compl,
                algebraic_factor, algebraic_add, max_results, arity,
            )
            all_results.extend(round_results)

        all_results = _dedup_results(all_results, digits)
        if len(all_results) >= max_results:
            break

    all_results.sort(key=formula_complexity)
    return all_results[:max_results]
