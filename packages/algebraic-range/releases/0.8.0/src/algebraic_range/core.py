"""
algebraic_range — Generate ranges of algebraic numbers.

A Python translation of the Wolfram Language ResourceFunction ``AlgebraicRange``
originally contributed by Daniele Gregori.

This module extends the basic concept of ``range`` to include, besides rational
numbers, also roots — always restricted to the real domain.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional, Union

import sympy
from sympy import (
    Abs as spAbs,
    Integer,
    Rational,
    Pow,
    S,
    nsimplify,
    factorint,
    ask,
    Q,
)

__all__ = ["algebraic_range", "formula_complexity", "AlgebraicRangeError"]
__version__ = "0.8.0"

Number = Union[int, float, Fraction, sympy.Basic]


# ── Exceptions ──────────────────────────────────────────────────────────────

class AlgebraicRangeError(Exception):
    """Base exception for algebraic_range errors."""


class NotRealError(AlgebraicRangeError):
    """Input parameters are not real numbers."""


class NotAlgebraicError(AlgebraicRangeError):
    """Input parameters are not algebraic numbers."""


class StepBoundError(AlgebraicRangeError):
    """The lower bound *d* exceeds the step upper bound |s|."""


# ── Internal helpers ────────────────────────────────────────────────────────

def _to_sympy(x: Number) -> sympy.Basic:
    """Coerce a numeric input to a sympy expression."""
    if isinstance(x, sympy.Basic):
        return x
    if isinstance(x, int):
        return Integer(x)
    if isinstance(x, Fraction):
        return Rational(x.numerator, x.denominator)
    if isinstance(x, float):
        return nsimplify(x, rational=False)
    return sympy.sympify(x)


def _nv(x: sympy.Basic) -> float:
    """Return a float approximation for sorting / comparisons."""
    try:
        return float(x.evalf(n=18))
    except Exception:
        return float(sympy.re(x).evalf(n=18))


def _is_real(x: sympy.Basic) -> bool:
    if x.is_real is True:
        return True
    if x.is_real is False:
        return False
    r = ask(Q.real(x))
    if r is not None:
        return r
    try:
        return abs(complex(x).imag) < 1e-30
    except Exception:
        return False


def _is_algebraic(x: sympy.Basic) -> bool:
    if x.is_algebraic is True:
        return True
    if x.is_algebraic is False:
        return False
    r = ask(Q.algebraic(x))
    if r is not None:
        return r
    return True  # conservative default for radical expressions


def _clean_sort(lst: List[sympy.Basic]) -> List[sympy.Basic]:
    """Sort numerically and remove duplicates (by numerical value)."""
    if not lst:
        return []
    seen: dict[float, sympy.Basic] = {}
    for item in lst:
        key = round(_nv(item), 12)
        if key not in seen:
            seen[key] = item
    result = list(seen.values())
    result.sort(key=_nv)
    return result


def _step_select(lst: List[sympy.Basic], d) -> List[sympy.Basic]:
    """Greedy filter: keep elements at least *d* apart."""
    if not lst:
        return []
    d_f = float(spAbs(_to_sympy(d)).evalf())
    out = [lst[0]]
    last = _nv(lst[0])
    for item in lst[1:]:
        nv = _nv(item)
        if abs(nv - last) >= d_f - 1e-15:
            out.append(item)
            last = nv
    return out


# ── Formula‑complexity heuristic ────────────────────────────────────────────

def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _int_complexity(n: int) -> float:
    """Per‑integer complexity: 0.5 * mean(DigitSum, 5*Len, #PrimeFactors, sqrt)."""
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
        base, exp = expr.args
        if exp.is_Rational and not exp.is_Integer:
            deg = max(int(abs(exp.p)), int(abs(exp.q)))
            _collect_integers(base, acc, mult * deg)
            return
        if exp.is_Integer:
            _collect_integers(base, acc, mult * int(abs(exp)))
            return
    for a in expr.args:
        _collect_integers(a, acc, mult)


def formula_complexity(expr: Number) -> float:
    """
    Compute the heuristic formula complexity of an algebraic expression.

    See the original WL documentation for the full prescription.
    """
    expr = _to_sympy(expr)
    ints: list[int] = []
    _collect_integers(expr, ints)
    positives = [abs(n) + 1 if n <= 0 else n for n in ints]
    return sum(_int_complexity(p) for p in positives) if positives else 0.0


# ── Elementary algebraic range ──────────────────────────────────────────────

def _elem_range_pos(order: int, lo: sympy.Basic, hi: sympy.Basic):
    """Range[lo^order .. hi^order]^(1/order)  for  0 <= lo <= hi."""
    lo_p = _safe_int_pow(lo, order, math.floor)
    hi_p = _safe_int_pow(hi, order, math.ceil)
    nlo, nhi = _nv(lo), _nv(hi)
    out = []
    for n in range(lo_p, hi_p + 1):
        val = Integer(n) ** Rational(1, order) if order > 1 else Integer(n)
        vnv = _nv(val)
        if nlo - 1e-12 <= vnv <= nhi + 1e-12:
            out.append(sympy.powsimp(val))
    return out


def _safe_int_pow(x, order, rounding):
    """x^order → nearest integer, using exact arithmetic when possible."""
    xp = x ** order
    try:
        if xp.is_Integer:
            return int(xp)
    except Exception:
        pass
    return rounding(float(xp.evalf()))


def _elem_range(order: int, r1, r2):
    """Elementary algebraic range for a single root order, any sign combo."""
    n1, n2 = _nv(r1), _nv(r2)
    if n1 >= 0 and n2 >= n1:
        return _elem_range_pos(order, r1, r2)
    if n1 < 0 and n2 >= 0:
        neg = [-v for v in _elem_range_pos(order, S.Zero, -r1)]
        pos = _elem_range_pos(order, S.Zero, r2)
        return _clean_sort(neg + pos)
    if n2 < 0 and n1 <= n2:
        return _clean_sort([-v for v in _elem_range_pos(order, -r2, -r1)])
    if n2 < n1:
        return []
    return []


def _elem_range_root_step(order, r1, r2, s):
    """'Root' step method: roots of Range[r1^ord, r2^ord, s^ord]."""
    n1, n2, ns = _nv(r1), _nv(r2), _nv(s)
    # Positive ascending case
    if ns > 0 and n1 >= 0 and n2 >= n1:
        lo_p_exact = spAbs(r1) ** order
        hi_p_exact = spAbs(r2) ** order
        s_p_exact = spAbs(s) ** order
        # Use rational arithmetic when possible
        lo_p_r = sympy.nsimplify(lo_p_exact)
        hi_p_r = sympy.nsimplify(hi_p_exact)
        s_p_r = sympy.nsimplify(s_p_exact)
        out = []
        current = lo_p_r
        while _nv(current) <= _nv(hi_p_r) + 1e-12:
            val = current ** Rational(1, order) if order > 1 else current
            val = sympy.powsimp(val)
            vnv = _nv(val)
            if n1 - 1e-12 <= vnv <= n2 + 1e-12:
                out.append(val)
            current = current + s_p_r
            if len(out) > 50000:
                break
        return out
    # Negative step → reverse the positive version
    if ns < 0 and n1 >= n2:
        pos = _elem_range_root_step(order, r2, r1, -s)
        return list(reversed(pos)) if pos else []
    # Mixed signs with negative step
    if ns < 0 and n1 >= 0 and n2 < 0:
        neg = [-v for v in _elem_range_root_step(order, S.Zero, -r2, -s)]
        pos = _elem_range_root_step(order, S.Zero, r1, -s)
        combined = _clean_sort(neg + pos)
        return list(reversed(combined))
    return []


# ── Factor range (multiplier set for the Outer product) ─────────────────────

def _arange_sympy(start, stop, step):
    """Generate a list analogous to Range[start, stop, step] in WL."""
    out = []
    cur = start
    ns = _nv(step)
    if ns > 0:
        while _nv(cur) <= _nv(stop) + 1e-12:
            out.append(cur)
            cur = cur + step
            if len(out) > 50000:
                break
    elif ns < 0:
        while _nv(cur) >= _nv(stop) - 1e-12:
            out.append(cur)
            cur = cur + step
            if len(out) > 50000:
                break
    return out


def _factor_range(r1, r2, s, farey: bool):
    """Build the multiplier list for the outer product construction."""
    nmax = max(abs(_nv(r1)), abs(_nv(r2)))
    abs_s = spAbs(s)
    one = sympy.Max(S.One, abs_s)

    if not farey:
        down = [v for v in _arange_sympy(one, S.Zero, -abs_s) if _nv(v) != 1.0 or abs(_nv(v) - 1.0) > 1e-15]
        # Cleaner: exclude exact 1
        down = [v for v in _arange_sympy(one, S.Zero, -abs_s) if abs(_nv(v) - 1.0) > 1e-15]
        up = _arange_sympy(one, _to_sympy(nmax), abs_s)
        return down + up
    else:
        return _farey_factor_range(one, nmax, abs_s)


def _farey_sequence(n: int) -> List[sympy.Basic]:
    """Farey sequence F_n."""
    result = [Rational(0, 1)]
    a, b, c, d = 0, 1, 1, n
    while c <= n:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        result.append(Rational(a, b))
    return result


def _farey_factor_range(one, nmax, abs_s):
    nas = _nv(abs_s)
    if nas >= 1:
        order = int(nas)
    elif nas > 0:
        order = int(round(1.0 / nas))
    else:
        return [S.One]
    seq = [f for f in _farey_sequence(order) if _nv(f) > 0]
    factors = set()
    for step in seq:
        for v in _arange_sympy(sympy.Max(S.One, step), S.Zero, -step):
            if abs(_nv(v) - 1.0) > 1e-15:
                factors.add(v)
        for v in _arange_sympy(sympy.Max(S.One, step), _to_sympy(nmax), step):
            factors.add(v)
    return list(factors)


# ── Outer range ─────────────────────────────────────────────────────────────

def _outer_range(order, r1, r2, s, farey=False):
    """Outer product step construction."""
    n1, n2, ns = _nv(r1), _nv(r2), _nv(s)

    if ns > 0 and n1 <= n2:
        if ns <= 1:
            elem_lo = r1
        else:
            elem_lo = r1 / s
        elem = _elem_range(order, elem_lo, r2)
        factors = _factor_range(r1, r2, s, farey)
        products = [e * f for e in elem for f in factors]
        result = _clean_sort(products)
        return [v for v in result if n1 - 1e-12 <= _nv(v) <= n2 + 1e-12]

    if ns < 0 and n2 <= n1:
        abs_s = -s
        if ns >= -1:
            elem_lo = r1
        else:
            elem_lo = r1 / abs_s
        # Build the elem range for the reversed (positive) direction
        lo_v = min(abs(n1), abs(n2))
        hi_v = max(abs(n1), abs(n2))
        # For mixed-sign ranges: compute elem over [0, max_abs]
        if n2 < 0 and n1 >= 0:
            elem = _elem_range(order, S.Zero, _to_sympy(hi_v))
        elif n1 >= 0 and n2 >= 0:
            elem = _elem_range(order, r2, r1)
        else:
            elem = _elem_range(order, _to_sympy(lo_v), _to_sympy(hi_v))
        factors = _factor_range(r1, r2, s, farey)
        products = [e * f for e in elem for f in factors]
        # For negative step with mixed signs, include negatives
        if n2 < 0 and n1 >= 0:
            neg_elem = _elem_range(order, S.Zero, _to_sympy(abs(n2)))
            neg_prods = [-e * f for e in neg_elem for f in factors]
            products.extend(neg_prods)
        result = _clean_sort(products)
        mn, mx = min(n1, n2), max(n1, n2)
        filtered = [v for v in result if mn - 1e-12 <= _nv(v) <= mx + 1e-12]
        return list(reversed(filtered))

    return []


# ── Public API ──────────────────────────────────────────────────────────────

def algebraic_range(
    r1: Number,
    r2: Optional[Number] = None,
    s: Optional[Number] = None,
    d: Number = 0,
    *,
    root_order: Union[int, List[int]] = 2,
    step_method: str = "Outer",
    farey_range: bool = False,
    formula_complexity_threshold: Number = math.inf,
    algebraics_only: bool = True,
) -> List[sympy.Basic]:
    """
    Generate a range of algebraic numbers (real roots).

    Parameters
    ----------
    r1 : number
        Single-argument form: ``algebraic_range(x)`` returns
        ``[n**(1/2) for n in 1..x**2]``.
        Multi-argument: starting bound.
    r2 : number, optional
        Ending bound.
    s : number, optional
        Step upper bound.  Negative *s* → descending range.
    d : number, default 0
        Minimum absolute difference between successive elements
        (``0 <= d <= |s|``).
    root_order : int or list[int], default 2
        ``int r`` — include roots of orders 2 through *r*.
        ``[r]`` — only roots of order *r*.
        ``[r1, r2, …]`` — roots of all listed orders.
    step_method : ``"Outer"`` | ``"Root"``, default ``"Outer"``
        Algorithm for the step construction.
    farey_range : bool, default False
        Use Farey-sequence–based rational multipliers.
    formula_complexity_threshold : number, default inf
        Discard elements whose formula complexity exceeds this.
    algebraics_only : bool, default True
        Reject transcendental inputs with ``NotAlgebraicError``.

    Returns
    -------
    list[sympy.Expr]
        Sorted list of algebraic numbers.

    Raises
    ------
    NotRealError
        If any input is not real.
    NotAlgebraicError
        If ``algebraics_only`` and any input is transcendental.
    StepBoundError
        If ``d > |s|``.
    """
    # ── Convert to sympy ────────────────────────────────────────────────
    r1_s = _to_sympy(r1)

    if r2 is None and s is None:
        # Single-argument form: algebraic_range(x) → Sqrt[Range[1,x^2]]
        r2_s = r1_s
        r1_s = S.One
        s_s = S.One
    elif s is None:
        r2_s = _to_sympy(r2)
        s_s = S.One
    else:
        r2_s = _to_sympy(r2)
        s_s = _to_sympy(s)

    d_s = _to_sympy(d)

    # ── Validate ────────────────────────────────────────────────────────
    for p in (r1_s, r2_s, s_s):
        if not _is_real(p):
            raise NotRealError(
                f"The input parameter {p} is not a real number."
            )
    if algebraics_only:
        for p in (r1_s, r2_s, s_s):
            if not _is_algebraic(p):
                raise NotAlgebraicError(
                    f"The input parameter {p} is not an algebraic number."
                )

    d_f = float(spAbs(d_s).evalf())
    s_f = float(spAbs(s_s).evalf())
    if d_f > s_f + 1e-15:
        raise StepBoundError(
            f"Lower bound d={d} exceeds step upper bound |s|={s_f}."
        )

    # ── Normalise root_order ────────────────────────────────────────────
    if isinstance(root_order, int):
        orders = list(range(2, root_order + 1))
    elif isinstance(root_order, (list, tuple)):
        orders = list(root_order)
    else:
        orders = [int(root_order)]

    step_neg = _nv(s_s) < 0

    # ── Compute ─────────────────────────────────────────────────────────
    if len(orders) == 1:
        combined = _single_order(
            orders[0], r1_s, r2_s, s_s, step_method, farey_range
        )
    else:
        parts = []
        for o in orders:
            parts.extend(
                _single_order(o, r1_s, r2_s, s_s, step_method, farey_range)
            )
        combined = _clean_sort(parts)
        if step_neg:
            combined = list(reversed(combined))

    # ── Formula complexity filter ───────────────────────────────────────
    if formula_complexity_threshold < math.inf:
        combined = [
            v for v in combined
            if formula_complexity(v) <= formula_complexity_threshold
        ]

    # ── Minimum-step filter ─────────────────────────────────────────────
    if d_f > 0:
        combined = _step_select(combined, d_s)

    return combined


def _single_order(order, r1, r2, s, method, farey):
    """Dispatch for one root order."""
    if s == S.One:
        return _elem_range(order, r1, r2)
    if method == "Root":
        return _elem_range_root_step(order, r1, r2, s)
    return _outer_range(order, r1, r2, s, farey)
