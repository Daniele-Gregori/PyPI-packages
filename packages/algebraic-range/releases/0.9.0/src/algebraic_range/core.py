"""
algebraic_range — Generate ranges of algebraic numbers.

A Python translation of the Wolfram Language ResourceFunction ``AlgebraicRange``
version 2.0, originally contributed by Daniele Gregori.

This module extends the basic concept of ``range`` to include, besides rational
numbers, also roots — always restricted to the real domain.

The implementation follows the Wolfram Language 2.0 source closely,
section by section:

* Formula complexity   (``formulaComplexity``)
* Utilities            (``realReplace``, ``cleanSort``, ``stepSelect``)
* Error management     (``Failure`` objects → Python exceptions)
* Farey range          (delegated to the ``farey`` package)
* Elementary range     (``elemRange``)
* Step range           (``factorRange``, ``outerRange``, ``stepRange``)
* Restricted range     (``complexitySelect``, ``restrictRange``)
* Main definition      (``iAlgebraicRange``, ``AlgebraicRange``)
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_left, bisect_right
from fractions import Fraction
from typing import List, Optional, Sequence, Union

import sympy
from sympy import Integer, Rational, S

__all__ = [
    "algebraic_range",
    "formula_complexity",
    "AlgebraicRangeError",
    "NotRealError",
    "NotAlgebraicError",
    "FareyStepError",
    "LowerBoundError",
    "StepBoundError",
]
__version__ = "0.9.0"

Number = Union[int, float, Fraction, sympy.Basic]

_HALF_TIE = Rational(1, 2)


# ── Error management ────────────────────────────────────────────────────────
# WL 2.0 returns Failure objects; the Python port raises exceptions with the
# same message templates.

class AlgebraicRangeError(Exception):
    """Base exception for algebraic_range errors."""


class NotRealError(AlgebraicRangeError):
    """Failure["NotReal"]: input parameters are not real numbers."""


class NotAlgebraicError(AlgebraicRangeError):
    """Failure["NotAlgebraic"]: input parameters are not algebraic numbers."""


class FareyStepError(AlgebraicRangeError):
    """Failure["FareyStep"]: step not allowed with the Farey range option."""


class LowerBoundError(AlgebraicRangeError):
    """Failure["LowerBound"]: the steps' lower bound d cannot be negative."""


class StepBoundError(AlgebraicRangeError):
    """Failure["UpperLowerBound"]: |s| cannot be lower than d."""


def _failure_not_real(params) -> NotRealError:
    return NotRealError(
        f"The input parameters {params} are not real numbers."
    )


def _failure_not_alg(params) -> NotAlgebraicError:
    return NotAlgebraicError(
        f"The input parameters {params} are not algebraic numbers."
    )


def _failure_farey_step(s) -> FareyStepError:
    return FareyStepError(
        f"The step parameter {s} is not allowed with the Farey range option."
    )


def _failure_lower_bound(d) -> LowerBoundError:
    return LowerBoundError(
        f"The steps' lower bound parameter {d} cannot be negative."
    )


def _failure_upper_bound(s, d) -> StepBoundError:
    return StepBoundError(
        f"The steps' upper bound {s} cannot be lower than the lower bound"
        f" {d} in absolute value."
    )


# ── Numeric context ─────────────────────────────────────────────────────────
# WL threads WorkingPrecision through cleanSort and all numeric decisions.
# A context object carries the precision and a per-call numeric-value cache.

class _Ctx:
    __slots__ = ("wp", "cache")

    def __init__(self, wp: Optional[int] = None):
        self.wp = wp  # None → MachinePrecision (float64), int → decimal digits
        self.cache: dict = {}

    def nval(self, x):
        """N[x, wp] — the numeric key used for sorting and deduplication."""
        key = self.cache.get(x)
        if key is not None:
            return key
        key = _nval_raw(x, self.wp)
        self.cache[x] = key
        return key

    def seed(self, x, key):
        """Record a numeric key computed alongside expression construction."""
        self.cache[x] = key


def _nval_raw(x, wp):
    if wp is None:
        if isinstance(x, (int, Fraction)):
            return float(x)
        try:
            return float(x)
        except TypeError:
            return float(sympy.re(x).evalf())
    v = sympy.Float(x, wp) if isinstance(x, (int, Fraction)) else x.evalf(wp)
    return v


def _to_sympy(x: Number) -> sympy.Basic:
    if isinstance(x, sympy.Basic):
        return x
    if isinstance(x, bool):
        raise TypeError("Boolean input is not a number.")
    if isinstance(x, int):
        return Integer(x)
    if isinstance(x, Fraction):
        return Rational(x.numerator, x.denominator)
    if isinstance(x, float):
        return sympy.Float(x)
    return sympy.sympify(x)


def _exact_sign(expr, ctx: _Ctx) -> int:
    """Sign of an exact real expression, resolving numeric ties exactly."""
    if isinstance(expr, (int, Fraction)):
        return (expr > 0) - (expr < 0)
    if expr.is_Rational:
        return bool(expr > 0) - bool(expr < 0)
    v = float(ctx.nval(expr))
    if abs(v) > 1e-9 * (1.0 + abs(v)):
        return 1 if v > 0 else -1
    hv = expr.evalf(50)
    if abs(hv) < sympy.Float(10) ** (-45):
        return 0
    return 1 if hv > 0 else -1


def _cmp(a, b, ctx: _Ctx) -> int:
    """Compare two exact reals: -1, 0, 1 (numeric fast path, exact ties)."""
    if isinstance(a, (int, Fraction)) and isinstance(b, (int, Fraction)):
        return (a > b) - (a < b)
    return _exact_sign(_to_sympy(a) - _to_sympy(b), ctx)


# ── Formula complexity ──────────────────────────────────────────────────────
# Transliteration of formulaComplexityHeuristic: a single outside-in
# replacement pass turns the expression into nested lists of integers, which
# are then scored individually.

def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _big_omega(n: int) -> int:
    # WL FactorInteger[1] = {{1, 1}}, so Omega(1) = 1.
    if n == 1:
        return 1
    return sum(sympy.factorint(n).values())


def _int_score(i: int) -> float:
    # Mean[(1/2) {5 IntegerLength, DigitSum, Omega, Sqrt}]
    if i <= 0:
        i = -i + 1
    return 0.5 * (
        5 * len(str(i)) + _digit_sum(i) + _big_omega(i) + math.sqrt(i)
    ) / 4.0


def _fc_structure(expr):
    """Passes 1–3: constants → 1, Complex → [re, im], Rational → [p, q]."""
    if isinstance(expr, sympy.NumberSymbol):
        return 1
    if expr.is_Integer:
        return int(expr)
    if expr.is_Rational:
        return [int(expr.p), int(expr.q)]
    if expr.is_Float:
        return []  # reals carry no integer content
    if expr.is_number and expr.is_real is False:
        re, im = expr.as_real_imag()
        return [_fc_structure(re), _fc_structure(im)]
    if isinstance(expr, sympy.exp):
        # WL sees Sqrt[E] as Power[E, 1/2]; sympy stores it as exp(1/2)
        return [sympy.Pow, 1, _fc_structure(expr.args[0])]
    return [expr.func] + [_fc_structure(a) for a in expr.args] \
        if expr.args else [expr]


def _fc_power_pass(node, collected):
    """Pass 5 (outermost only): Power[b, [m, n]] → (|m|+|n|) copies of b."""
    if isinstance(node, int):
        collected.append(node)
        return
    if not isinstance(node, list):
        return
    if node and node[0] is sympy.Pow and len(node) == 3:
        base, expo = node[1], node[2]
        if isinstance(expo, list) and len(expo) == 2 \
                and all(isinstance(e, int) for e in expo):
            m, n = expo
            copies = abs(n) + abs(m)
            for _ in range(copies):
                _fc_collect_raw(base, collected)
            return
    for child in (node[1:] if node and not isinstance(node[0], (int, list))
                  else node):
        _fc_power_pass(child, collected)


def _fc_collect_raw(node, collected):
    """Collect every integer in a subtree without further Power replacement."""
    if isinstance(node, int):
        collected.append(node)
    elif isinstance(node, list):
        for child in (node[1:] if node and not isinstance(node[0], (int, list))
                      else node):
            _fc_collect_raw(child, collected)


def formula_complexity(form: Number) -> float:
    """
    Heuristic complexity of a numeric expression (WL ``formulaComplexity``).

    Every integer appearing in the expression — with radicals ``b^(m/n)``
    contributing ``|m| + |n|`` copies of their base, rationals contributing
    numerator and denominator, and built-in constants counting as 1 —
    is scored by ``(5 IntegerLength + DigitSum + Omega + Sqrt)/8``,
    and the scores are summed.
    """
    expr = _to_sympy(form)
    if not expr.is_number:
        raise TypeError(f"{form} is not a numeric expression.")
    structure = _fc_structure(expr)
    ints: List[int] = []
    _fc_power_pass(structure, ints)
    return float(sum(_int_score(i) for i in ints))


# ── Utilities ───────────────────────────────────────────────────────────────

def _real_replace(x):
    """
    WL ``realReplace``: exactify approximate reals (RootApproximant).

    Decimal-looking floats become exact rationals; other floats are
    recognized as algebraic numbers of small degree when possible.
    """
    if isinstance(x, float):
        x = sympy.Float(x)
    if not isinstance(x, sympy.Basic) or not x.is_Float:
        return x
    f = float(x)
    frac = Fraction(f).limit_denominator(10 ** 12)
    if frac != 0 and abs(float(frac) - f) <= 1e-14 * max(1.0, abs(f)):
        return Rational(frac.numerator, frac.denominator)
    if f == 0.0:
        return S.Zero
    root = _root_approximant(f)
    if root is not None:
        return root
    return Rational(frac.numerator, frac.denominator)


def _root_approximant(f: float):
    """Small-degree analogue of RootApproximant, via mpmath's findpoly."""
    try:
        from mpmath import findpoly, mp, mpf
    except ImportError:  # pragma: no cover
        return None
    old = mp.dps
    try:
        mp.dps = 17
        for deg in (2, 3, 4, 5, 6):
            coeffs = findpoly(mpf(f), deg, maxcoeff=10 ** 8)
            if coeffs is None:
                continue
            poly = sympy.Poly(coeffs, sympy.Symbol("x"))
            try:
                roots = poly.real_roots()
            except Exception:
                continue
            best = None
            for r in roots:
                err = abs(float(r.evalf(20)) - f)
                if best is None or err < best[0]:
                    best = (err, r)
            if best is not None and best[0] <= 1e-12 * max(1.0, abs(f)):
                return best[1]
    except Exception:
        return None
    finally:
        mp.dps = old
    return None


def _clean_sort(lst: Sequence, ctx: _Ctx) -> list:
    """WL ``cleanSort``: SortBy[DeleteDuplicatesBy[list, N[#, wp]], N[#, wp]]."""
    seen = {}
    for item in lst:
        key = ctx.nval(item)
        if key not in seen:
            seen[key] = item
    return [seen[k] for k in sorted(seen)]


def _step_select(lst: Sequence, d, ctx: _Ctx) -> list:
    """WL ``stepSelect``: greedy filter keeping elements at least d apart."""
    if not len(lst):
        return []
    d_s = _to_sympy(d)
    d_f = float(abs(d_s.evalf()))
    sel = [lst[0]]
    eln = lst[0]
    eln_f = float(ctx.nval(eln))
    for item in lst[1:]:
        it_f = float(ctx.nval(item))
        gap = abs(it_f - eln_f)
        if gap >= d_f:
            accept = True
        elif gap > d_f - 1e-9 * (1.0 + d_f):
            # numeric tie — resolve exactly, as WL's symbolic Abs[x-y] >= d
            accept = bool(
                (sympy.Abs(_to_sympy(item) - _to_sympy(eln)) - d_s)
                .evalf(40) >= -sympy.Float(10) ** (-35)
            )
        else:
            accept = False
        if accept:
            sel.append(item)
            eln = item
            eln_f = it_f
    return sel


# ── Farey range ─────────────────────────────────────────────────────────────
# Transliteration of the improved FareyRange resource function:
# FareyRange[xmin, xmax, n] gives the sorted rationals with denominator <= n
# inside [xmin, xmax], generated by the Farey next-term recurrence from a
# bracketing pair (fareyBracket / fareyPartial).

def _as_fraction(x) -> Optional[Fraction]:
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    if isinstance(x, sympy.Basic) and x.is_Rational:
        return Fraction(int(x.p), int(x.q))
    return None


def _farey_range(r1, r2, r3) -> List[Fraction]:
    """Delegate to the ``farey`` package (optional dependency)."""
    try:
        from farey import farey_range as _fr
    except ImportError:
        raise ImportError(
            "The 'farey' package is required for farey_range=True. "
            "Install it with: pip install farey"
        ) from None
    f1, f2, f3 = _as_fraction(r1), _as_fraction(r2), _as_fraction(r3)
    if f3 is None or f1 is None or f2 is None:
        raise _failure_farey_step(r3)
    try:
        return _fr(f1, f2, f3)
    except Exception:
        raise _failure_farey_step(r3) from None


# ── Range building blocks ───────────────────────────────────────────────────

def _wl_range(a, b, s=1, ctx: Optional[_Ctx] = None) -> list:
    """
    WL ``Range[a, b, s]`` over exact numbers.

    Uses Fraction arithmetic when possible, sympy otherwise; the element
    count is Floor[(b - a)/s] resolved exactly on numeric ties.
    """
    ctx = ctx or _Ctx()
    fa, fb, fs = _as_fraction(a), _as_fraction(b), _as_fraction(s)
    if fa is not None and fb is not None and fs is not None:
        if fs == 0:
            return []
        k = (fb - fa) / fs
        kmax = math.floor(k)
        return [fa + i * fs for i in range(kmax + 1)]
    a_s, b_s, s_s = _to_sympy(a), _to_sympy(b), _to_sympy(s)
    ratio = (b_s - a_s) / s_s
    fr = _as_fraction(ratio)
    if fr is not None:
        kmax = math.floor(fr)
    else:
        rv = float(ratio.evalf())
        kmax = math.floor(rv)
        # exact tie resolution around integer boundaries
        if abs(rv - round(rv)) < 1e-9:
            kr = round(rv)
            sign = _exact_sign(ratio - Integer(kr), ctx)
            kmax = kr if sign >= 0 else kr - 1
    if kmax < 0:
        return []
    return [sympy.expand(a_s + i * s_s) for i in range(kmax + 1)]


_SQF_CACHE: dict = {}
_SPF: list = [0, 1]  # smallest-prime-factor sieve, grown on demand


def _ensure_spf(n: int):
    global _SPF
    if n < len(_SPF):
        return
    size = max(n + 1, 2 * len(_SPF))
    spf = list(range(size))
    i = 2
    while i * i < size:
        if spf[i] == i:
            for j in range(i * i, size, i):
                if spf[j] == j:
                    spf[j] = i
        i += 1
    _SPF = spf


def _sqf_decompose(n: int):
    """n = a²·b with b squarefree."""
    dec = _SQF_CACHE.get(n)
    if dec is not None:
        return dec
    a, b = 1, 1
    if n < len(_SPF):
        m = n
        while m > 1:
            p = _SPF[m]
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            a *= p ** (e // 2)
            b *= p ** (e % 2)
    else:
        for p, e in sympy.factorint(n).items():
            a *= p ** (e // 2)
            b *= p ** (e % 2)
    _SQF_CACHE[n] = (a, b)
    return a, b


def _sqrt_int(n: int, ctx: _Ctx):
    """Fast canonical sqrt of a nonnegative int: n = a²·b → a·√b."""
    if n == 0:
        return S.Zero
    if n == 1:
        return S.One
    a, b = _sqf_decompose(n)
    if b == 1:
        expr = Integer(a)
    elif a == 1:
        expr = sympy.Pow(Integer(b), S.Half, evaluate=False)
    else:
        expr = sympy.Mul(Integer(a), sympy.Pow(Integer(b), S.Half,
                                               evaluate=False), evaluate=False)
    if ctx.wp is None:
        ctx.seed(expr, math.sqrt(n))
    return expr


def _nth_root(g, order, ctx: _Ctx):
    """g^(1/order) in canonical sympy form, with numeric key seeded."""
    if isinstance(g, int) and order == 2 and g >= 0:
        return _sqrt_int(g, ctx)
    if isinstance(g, Fraction):
        if order == 2 and g >= 0:
            num = _sqrt_int(g.numerator, ctx)
            den = _sqrt_int(g.denominator, ctx)
            expr = num / den
            if ctx.wp is None:
                ctx.seed(expr, math.sqrt(g.numerator) / math.sqrt(g.denominator))
            return expr
        g = Rational(g.numerator, g.denominator)
    g_s = _to_sympy(g)
    if order == 1:
        return g_s
    return sympy.Pow(g_s, Rational(1, order))


def _roots_of(grid: Sequence, order, ctx: _Ctx) -> list:
    if order == 2 and len(grid) > 512 and isinstance(grid[-1], int):
        ints = [g for g in grid if isinstance(g, int)]
        if len(ints) == len(grid):
            _ensure_spf(max(grid[0], grid[-1]))
    return [_nth_root(g, order, ctx) for g in grid]


def _neg(values: Sequence, ctx: Optional[_Ctx] = None) -> list:
    out = []
    for v in values:
        if isinstance(v, (int, Fraction)):
            out.append(-v)
            continue
        nv = -v
        if ctx is not None and ctx.wp is None:
            key = ctx.cache.get(v)
            if key is not None:
                ctx.seed(nv, -key)
        out.append(nv)
    return out


def _pow_exact(x, order):
    """x^order keeping int/Fraction types when possible."""
    if isinstance(x, (int, Fraction)):
        r = x ** order if isinstance(order, int) else None
        if r is not None:
            return r
    xs = _to_sympy(x) ** order
    fr = _as_fraction(xs)
    return fr if fr is not None else xs


def _select_between(values, lo, hi, ctx: _Ctx) -> list:
    """Select lo <= v <= hi — float fast path, exact ties at the borders."""
    lo_f = float(_nval_raw(lo, None) if not isinstance(lo, (int, Fraction))
                 else lo)
    hi_f = float(_nval_raw(hi, None) if not isinstance(hi, (int, Fraction))
                 else hi)
    band = 1e-9 * (1.0 + max(abs(lo_f), abs(hi_f)))
    out = []
    for v in values:
        v_f = float(ctx.nval(v)) if ctx.wp is None else float(v)
        if lo_f + band < v_f < hi_f - band:
            out.append(v)
        elif v_f < lo_f - band or v_f > hi_f + band:
            continue
        elif _cmp(lo, v, ctx) <= 0 and _cmp(v, hi, ctx) <= 0:
            out.append(v)
    return out


# ── Elementary range ────────────────────────────────────────────────────────

def _elem_range(order, args, opts, ctx: _Ctx) -> list:
    """WL ``elemRange`` — the elementary algebraic range, all sign cases."""
    ord_ = order
    if len(args) == 1:
        r1 = args[0]
        if _cmp(r1, 1, ctx) >= 0:
            return _roots_of(_wl_range(1, _pow_exact(r1, ord_), 1, ctx),
                             ord_, ctx)
        return []

    if len(args) == 2:
        r1, r2 = args
        s1, s2 = _exact_sign(_to_sympy(r1), ctx), _exact_sign(_to_sympy(r2), ctx)
        if s1 >= 0 and _cmp(r1, r2, ctx) <= 0:
            return _roots_of(
                _wl_range(_pow_exact(r1, ord_), _pow_exact(r2, ord_), 1, ctx),
                ord_, ctx)
        if s2 <= 0 and _cmp(r1, r2, ctx) <= 0:
            vals = _neg(_roots_of(
                _wl_range(_pow_exact(-_to_sympy(r1), ord_),
                          _pow_exact(-_to_sympy(r2), ord_), -1, ctx),
                ord_, ctx), ctx)
            return _select_between(vals, r1, r2, ctx)
        if s1 < 0 and s2 >= 0:
            elrg1 = _elem_range(ord_, [r1, 0], opts, ctx)
            elrg2 = _elem_range(ord_, [0, r2], opts, ctx)
            p1 = _pow_exact(-_to_sympy(r1), ord_)
            p2 = _pow_exact(r2, ord_)
            if _is_integerq(p1) and _is_integerq(p2) and elrg1 and elrg2:
                return elrg1 + elrg2[1:]
            return elrg1 + elrg2
        return []

    r1, r2, r3 = args

    # descending marker -1 (internal)
    if _is_exact_one(r3, -1):
        s1 = _exact_sign(_to_sympy(r1), ctx)
        s2 = _exact_sign(_to_sympy(r2), ctx)
        if s2 >= 0 and _cmp(r2, r1, ctx) <= 0:
            return _roots_of(
                _wl_range(_pow_exact(r1, ord_), _pow_exact(r2, ord_), -1, ctx),
                ord_, ctx)
        if s1 <= 0 and _cmp(r2, r1, ctx) <= 0:
            # -Range[(-r1)^ord, (-r2)^ord]^(1/ord) is already descending
            vals = _neg(_roots_of(
                _wl_range(_pow_exact(-_to_sympy(r1), ord_),
                          _pow_exact(-_to_sympy(r2), ord_), 1, ctx),
                ord_, ctx), ctx)
            return _select_between(vals, r2, r1, ctx)
        if s2 < 0 and s1 >= 0:
            elrg1 = _elem_range(ord_, [r1, 0, Integer(-1)], opts, ctx)
            elrg2 = _elem_range(ord_, [0, r2, Integer(-1)], opts, ctx)
            p1 = _pow_exact(r1, ord_)
            p2 = _pow_exact(-_to_sympy(r2), ord_)
            if _is_integerq(p1) and _is_integerq(p2) and elrg1 and elrg2:
                return elrg1 + elrg2[1:]
            return elrg1 + elrg2
        return []

    s3 = _exact_sign(_to_sympy(r3), ctx)
    s1 = _exact_sign(_to_sympy(r1), ctx)
    s2 = _exact_sign(_to_sympy(r2), ctx)
    farey = opts["farey_range"]

    if s3 > 0 and _cmp(r1, r2, ctx) <= 0:
        if s1 >= 0:
            if _is_exact_one(r3, 1) and opts["step_method"] != "Root":
                return _elem_range(ord_, [r1, r2], opts, ctx)
            grid_args = (_pow_exact(r1, ord_), _pow_exact(r2, ord_),
                         _pow_exact(r3, ord_))
            if farey:
                grid = _farey_range(*grid_args)
            else:
                grid = _wl_range(*grid_args, ctx)
            return _roots_of(grid, ord_, ctx)
        if s2 <= 0:
            if _is_exact_one(r3, 1) and opts["step_method"] != "Root":
                return _elem_range(ord_, [r1, r2], opts, ctx)
            grid_args = (_pow_exact(-_to_sympy(r1), ord_),
                         _pow_exact(-_to_sympy(r2), ord_),
                         -_pow_exact(r3, ord_))
            if farey:
                grid = _farey_range(*grid_args)
            else:
                grid = _wl_range(*grid_args, ctx)
            vals = _neg(_roots_of(grid, ord_, ctx), ctx)
            return _select_between(vals, r1, r2, ctx)
        # r1 < 0 <= r2
        if _cmp(r3, _to_sympy(r2) - _to_sympy(r1), ctx) <= 0:
            elrg1 = _elem_range(ord_, [r1, 0, r3], opts, ctx)
            elrg2 = _elem_range(ord_, [0, r2, r3], opts, ctx)
            if elrg1 and elrg2 and _exact_sign(_to_sympy(elrg1[-1]), ctx) == 0:
                return elrg1 + elrg2[1:]
            return elrg1 + elrg2
        return [_to_sympy(r1)]

    if s3 < 0 and _cmp(r2, r1, ctx) <= 0:
        if s2 >= 0:
            return _roots_of(
                _wl_range(_pow_exact(r1, ord_), _pow_exact(r2, ord_),
                          -_pow_exact(-_to_sympy(r3), ord_), ctx),
                ord_, ctx)
        if s1 <= 0:
            vals = _neg(_roots_of(
                _wl_range(_pow_exact(-_to_sympy(r1), ord_),
                          _pow_exact(-_to_sympy(r2), ord_),
                          _pow_exact(-_to_sympy(r3), ord_), ctx),
                ord_, ctx), ctx)
            return _select_between(vals, r2, r1, ctx)
        # r2 < 0 <= r1
        if _cmp(-_to_sympy(r3), _to_sympy(r1) - _to_sympy(r2), ctx) <= 0:
            elrg1 = _elem_range(ord_, [r1, 0, r3], opts, ctx)
            elrg2 = _elem_range(ord_, [0, r2, r3], opts, ctx)
            if elrg1 and elrg2 and _exact_sign(_to_sympy(elrg1[-1]), ctx) == 0:
                return elrg1 + elrg2[1:]
            return elrg1 + elrg2
        return [_to_sympy(r1)]

    return []


def _is_integerq(x) -> bool:
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator == 1
    return isinstance(x, sympy.Basic) and x.is_Integer is True


def _is_exact_one(x, target: int) -> bool:
    """WL SameQ with 1 or -1: exact numerals only (floats excluded)."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return x == target
    if isinstance(x, Fraction):
        return x == target
    if isinstance(x, sympy.Basic):
        return (x.is_Integer or x.is_Rational) and x == target
    return False


# ── Step range ──────────────────────────────────────────────────────────────

def _wl_quotient(m, n, ctx: _Ctx):
    """WL Quotient[m, n] = Floor[m/n] for real numbers."""
    fm, fn = _as_fraction(m), _as_fraction(n)
    if fm is not None and fn is not None:
        return Fraction(math.floor(fm / fn))
    ratio = _to_sympy(m) / _to_sympy(n)
    rv = float(ratio.evalf())
    k = math.floor(rv)
    if abs(rv - round(rv)) < 1e-9:
        kr = round(rv)
        k = kr if _exact_sign(ratio - Integer(kr), ctx) >= 0 else kr - 1
    return k


def _abs_exact(x, ctx: _Ctx):
    return x if _exact_sign(_to_sympy(x), ctx) >= 0 else (
        -x if isinstance(x, (int, Fraction)) else -_to_sympy(x))


def _min_exact(a, b, ctx: _Ctx):
    return a if _cmp(a, b, ctx) <= 0 else b


def _max_exact(a, b, ctx: _Ctx):
    return a if _cmp(a, b, ctx) >= 0 else b


def _factor_range(rL, opts, ctx: _Ctx) -> list:
    """WL ``factorRange``: the multiplier grid for the outer construction."""
    r1, r2, r3 = rL
    s3 = _exact_sign(_to_sympy(r3), ctx)
    s1 = _exact_sign(_to_sympy(r1), ctx)
    s2 = _exact_sign(_to_sympy(r2), ctx)
    a_s = _abs_exact(r3, ctx)
    one = _max_exact(1, a_s, ctx)
    ar1, ar2 = _abs_exact(r1, ctx), _abs_exact(r2, ctx)
    min_a = _min_exact(ar1, ar2, ctx)
    max_a = _max_exact(ar1, ar2, ctx)

    if s3 > 0 and s1 > 0:
        max_ = _mul(one, _div(ar2, min_a))
        zero = _mul(_wl_quotient(_div(_div(r1, max_a), one), a_s, ctx), a_s)
    elif s3 > 0:
        max_ = max_a
        zero = 0
    elif s3 < 0 and s2 > 0:
        max_ = _mul(one, _div(ar1, min_a))
        zero = _mul(_wl_quotient(_div(_div(r2, max_a), one), a_s, ctx), a_s)
    else:
        max_ = max_a
        zero = 0

    if not opts["farey_range"]:
        rg1 = _wl_range(zero, one, a_s, ctx)
        rg2 = _wl_range(one, max_, a_s, ctx)
    else:
        if _is_integerq(a_s):
            a_s = _div(1, a_s)
        rg1 = _farey_range(_frac_or_fail(zero, r3), _frac_or_fail(one, r3),
                           _frac_or_fail(a_s, r3))
        rg2 = _farey_range(_frac_or_fail(one, r3), _frac_or_fail(max_, r3),
                           _frac_or_fail(a_s, r3))
    if rg1 and rg2 and _cmp(rg1[-1], one, ctx) == 0 \
            and _cmp(rg2[0], one, ctx) == 0:
        rg1 = rg1[:-1]
    return rg1 + rg2


def _frac_or_fail(x, s):
    f = _as_fraction(x)
    if f is None:
        raise _failure_farey_step(s)
    return f


def _mul(a, b):
    if isinstance(a, (int, Fraction)) and isinstance(b, (int, Fraction)):
        return a * b
    return _to_sympy(a) * _to_sympy(b)


def _div(a, b):
    if isinstance(a, (int, Fraction)) and isinstance(b, (int, Fraction)):
        return Fraction(a) / Fraction(b)
    return _to_sympy(a) / _to_sympy(b)


def _outer_range(order, rL, opts, ctx: _Ctx) -> list:
    """
    WL ``outerRange``: output equivalent to the Outer-product construction,
    computed with a per-element factor window (linear, not quadratic).
    """
    r1, r2, r3 = rL
    s3 = _exact_sign(_to_sympy(r3), ctx)
    span_cmp_asc = _cmp(r1, r2, ctx)

    if s3 > 0 and span_cmp_asc <= 0:
        if _cmp(r3, _to_sympy(r2) - _to_sympy(r1), ctx) > 0:
            return [_to_sympy(r1)]
        elem_lo = (_min_exact(r1, _div(r1, r3), ctx)
                   if _exact_sign(_to_sympy(r1), ctx) > 0
                   else _max_exact(r1, _div(r1, r3), ctx))
        elrg = _elem_range(order, [elem_lo, r2], opts, ctx)
        fcrg = _factor_range(rL, opts, ctx)
        return _clean_sort(
            _window_products(elrg, fcrg, r1, r2, False, ctx,
                             _seed_ok(r3)), ctx)

    if s3 < 0 and span_cmp_asc >= 0:
        if _cmp(-_to_sympy(r3), _to_sympy(r1) - _to_sympy(r2), ctx) > 0:
            return [_to_sympy(r1)]
        elem_lo = (_min_exact(r1, _div(-_to_sympy(r1), r3), ctx)
                   if _exact_sign(_to_sympy(r1), ctx) > 0
                   else _max_exact(r1, _div(-_to_sympy(r1), r3), ctx))
        elrg = _elem_range(order, [elem_lo, r2, Integer(-1)], opts, ctx)
        fcrg = _factor_range(rL, opts, ctx)
        return list(reversed(_clean_sort(
            _window_products(elrg, fcrg, r1, r2, True, ctx,
                             _seed_ok(r3)), ctx)))

    return []


def _seed_ok(r3) -> bool:
    """Fast float keys are safe unless the factor grid is finer than ~1e-12
    (then products must be numericized from their exact form, as WL's N)."""
    try:
        return abs(float(_to_sympy(r3).evalf())) > 1e-12
    except Exception:
        return False


def _window_products(elrg, fcrg, r1, r2, descending, ctx: _Ctx,
                     seed_keys: bool = True) -> list:
    """For each element, multiply by the factors within its exact window."""
    fc_keys = [float(_nval_raw(f, None)) for f in fcrg]
    out = []
    r1_s, r2_s = _to_sympy(r1), _to_sympy(r2)
    r1_f, r2_f = float(_nval_raw(r1_s, None)), float(_nval_raw(r2_s, None))
    for curr in elrg:
        sign = _exact_sign(_to_sympy(curr), ctx)
        if sign == 0:
            out.append(S.Zero)
            continue
        curr_s = _to_sympy(curr)
        curr_f = float(ctx.nval(curr)) if ctx.wp is None \
            else float(_nval_raw(curr_s, None))
        # window bounds as floats; the exact bound is built only for the
        # factors falling within the numeric tie band (WL's exact While)
        if (sign > 0) != descending:
            flo_f, fhi_f = r1_f / curr_f, r2_f / curr_f
            flo_of, fhi_of = r1_s, r2_s
        else:
            flo_f, fhi_f = r2_f / curr_f, r1_f / curr_f
            flo_of, fhi_of = r2_s, r1_s
        band_lo = 1e-9 * (1.0 + abs(flo_f))
        band_hi = 1e-9 * (1.0 + abs(fhi_f))
        first = bisect_left(fc_keys, flo_f - band_lo)
        while first < len(fcrg) and fc_keys[first] < flo_f + band_lo \
                and _cmp(fcrg[first], flo_of / curr_s, ctx) < 0:
            first += 1
        last = bisect_right(fc_keys, fhi_f + band_hi) - 1
        while last >= 0 and fc_keys[last] > fhi_f - band_hi \
                and _cmp(fcrg[last], fhi_of / curr_s, ctx) > 0:
            last -= 1
        if ctx.wp is None and seed_keys:
            curr_key = ctx.nval(curr)
            for j in range(first, last + 1):
                prod = _mul_expr(curr_s, fcrg[j])
                ctx.seed(prod, curr_key * fc_keys[j])
                out.append(prod)
        elif ctx.wp is None:
            # Factor grid finer than double resolution: neighbours collide in
            # machine precision (the WorkingPrecision note of the WL doc).
            # Round each exact value once, at high precision, so collision
            # classes are well defined; WL's machine N may split them at
            # boundaries differing by a grid unit or two.
            for j in range(first, last + 1):
                prod = _mul_expr(curr_s, fcrg[j])
                ctx.seed(prod, float(prod.evalf(40)))
                out.append(prod)
        else:
            out.extend(_mul_expr(curr_s, fcrg[j])
                       for j in range(first, last + 1))
    return out


def _mul_expr(a: sympy.Basic, f):
    if isinstance(f, Fraction):
        if f == 1:
            return a
        f = Rational(f.numerator, f.denominator)
    elif isinstance(f, int):
        if f == 1:
            return a
        f = Integer(f)
    return a * f


def _step_range(order, rL, opts, ctx: _Ctx) -> list:
    """WL ``stepRange``: dispatch on the "StepMethod" option."""
    method = opts["step_method"]
    if method == "Outer":
        return _outer_range(order, rL, opts, ctx)
    if method == "Root":
        return _elem_range(order, rL, opts, ctx)
    raise ValueError(
        f'step_method must be "Outer" or "Root", got {method!r}.')


# ── Restricted range ────────────────────────────────────────────────────────

def _complexity_select(lst, c) -> list:
    return [v for v in lst if formula_complexity(v) <= c]


def _restrict_range(main, compl, d, ctx: _Ctx) -> list:
    """WL ``restrictRange``: complexity threshold, then minimum step."""
    d_pos = False
    if not (isinstance(d, (int, Fraction)) and d == 0):
        d_pos = _exact_sign(_to_sympy(d), ctx) > 0
    if compl == math.inf or compl is sympy.oo:
        return _step_select(main, d, ctx) if d_pos else main
    selected = _complexity_select(main, compl)
    return _step_select(selected, d, ctx) if d_pos else selected


# ── Main definition ─────────────────────────────────────────────────────────

_DEFAULT_OPTS = {
    "root_order": 2,
    "farey_range": False,
    "working_precision": None,
    "formula_complexity": math.inf,
    "step_method": "Outer",
    "algebraics_only": True,
}


def _is_real_number(x) -> bool:
    if isinstance(x, (int, Fraction)):
        return True
    r = x.is_real
    if r is not None:
        return bool(r)
    try:
        return abs(complex(x.evalf()).imag) < 1e-25
    except Exception:
        return False


def _is_algebraic_number(x) -> bool:
    if isinstance(x, (int, Fraction)):
        return True
    r = x.is_algebraic
    if r is not None:
        return bool(r)
    return True  # conservative default (radical combinations)


def _i_algebraic_range(orders, rL, d, opts) -> list:
    """WL ``iAlgebraicRange``."""
    if isinstance(orders, (int, Integer)) and not isinstance(orders, bool):
        orders = list(range(2, int(orders) + 1))
        if not orders:
            raise ValueError(
                "root_order as an integer r must satisfy r >= 2.")
    orders = list(orders)
    if len(orders) == 0:
        raise ValueError("root_order gave no root orders.")

    if len(orders) >= 2:
        step_neg = len(rL) == 3 and _exact_sign(
            _to_sympy(rL[2]), _Ctx(_wp_digits(opts))) < 0
        ctx = _Ctx(_wp_digits(opts))
        join: list = []
        for o in orders:
            join.extend(_i_algebraic_range([o], rL, d, opts))
        sort = _clean_sort(join, ctx)
        if step_neg:
            sort = list(reversed(sort))
        if not (isinstance(d, (int, Fraction)) and d == 0) \
                and _exact_sign(_to_sympy(d), ctx) != 0:
            sort = _step_select(sort, d, ctx)
        return sort

    # single root order
    ord_in = orders[0]
    ctx = _Ctx(_wp_digits(opts))
    r1, r2 = rL[0], rL[1]
    r3 = rL[2] if len(rL) == 3 else 1

    o = _real_replace(_to_sympy(ord_in))
    x = _real_replace(_to_sympy(r1))
    y = _real_replace(_to_sympy(r2))
    s = _real_replace(_to_sympy(r3))

    if float(sympy.re(o).evalf()) < 1:
        raise ValueError("root orders must be >= 1.")

    not_real = [p for p in (o, x, y, s) if not _is_real_number(p)]
    if not_real:
        raise _failure_not_real(not_real)
    if opts["algebraics_only"]:
        not_alg = [p for p in (o, x, y, s) if not _is_algebraic_number(p)]
        if not_alg:
            raise _failure_not_alg(not_alg)

    d_s = _to_sympy(d)
    if _cmp(d_s, sympy.Abs(_to_sympy(r3)), ctx) > 0:
        raise _failure_upper_bound(r3, d)

    if opts["farey_range"] and _is_integerq(s):
        s = _div(1, _as_fraction(s))

    o_int = int(o) if o.is_Integer else o
    x_c = _canon_number(x)
    y_c = _canon_number(y)
    s_c = _canon_number(s)

    if _is_exact_one(r3, 1) or _is_exact_one(r3, -1):
        marker = Integer(1) if _is_exact_one(r3, 1) else Integer(-1)
        main = _elem_range(o_int, [x_c, y_c, marker], opts, ctx)
    else:
        main = _step_range(o_int, [x_c, y_c, s_c], opts, ctx)

    return _restrict_range(main, opts["formula_complexity"], d, ctx)


def _canon_number(x):
    """Prefer Fraction/int for rational values (fast exact arithmetic)."""
    f = _as_fraction(x)
    if f is None:
        return x
    return int(f) if f.denominator == 1 else f


def _wp_digits(opts) -> Optional[int]:
    wp = opts["working_precision"]
    if wp is None:
        return None
    return int(round(float(wp)))


def algebraic_range(
    r1: Number,
    r2: Optional[Number] = None,
    s: Optional[Number] = None,
    d: Number = 0,
    *,
    root_order: Union[int, Sequence] = 2,
    step_method: str = "Outer",
    farey_range: bool = False,
    formula_complexity: Number = math.inf,
    algebraics_only: bool = True,
    working_precision: Optional[Union[int, float]] = None,
    formula_complexity_threshold: Optional[Number] = None,
) -> List[sympy.Basic]:
    """
    Generate a range of algebraic numbers (real roots).

    Python port of the Wolfram Language ``AlgebraicRange`` 2.0.

    Parameters
    ----------
    r1 : number
        ``algebraic_range(x)`` gives the square roots ``Sqrt[Range[1, x^2]]``
        for ``x >= 1``; with more arguments, the start of the range.
    r2 : number, optional
        End of the range: ``algebraic_range(x, y)`` gives
        ``Sqrt[Range[x^2, y^2]]`` for ``0 <= x <= y``, extended to negative
        bounds by reflection.
    s : number, optional
        Upper bound for the steps, with ``0 < s <= y - x``.
        A negative *s* produces a descending range.
    d : number, default 0
        Lower bound for the steps, with ``0 <= d <= |s|``.
    root_order : int or sequence, default 2
        ``r`` includes roots up to order *r*; ``[r]`` only order *r*;
        ``[r1, r2, ...]`` all listed orders.
    step_method : "Outer" | "Root", default "Outer"
        "Outer" takes the outer product of an elementary range with rational
        multipliers; "Root" uses ``(Range[x^n, y^n, s^n])^(1/n)``.
    farey_range : bool, default False
        Steps given by the Farey sequence (generalizes ``FareyRange``).
    formula_complexity : number, default inf
        Discard elements whose formula complexity exceeds this threshold.
    algebraics_only : bool, default True
        Reject transcendental input parameters.
    working_precision : int, optional
        Decimal precision for all internal numerical comparisons
        (default: machine precision).

    Returns
    -------
    list of sympy expressions, sorted (descending for negative *s*).

    Raises
    ------
    NotRealError, NotAlgebraicError, FareyStepError, LowerBoundError,
    StepBoundError
    """
    if formula_complexity_threshold is not None:
        warnings.warn(
            "formula_complexity_threshold is deprecated; "
            "use formula_complexity.",
            DeprecationWarning, stacklevel=2)
        formula_complexity = formula_complexity_threshold

    opts = {
        "root_order": root_order,
        "farey_range": bool(farey_range),
        "working_precision": working_precision,
        "formula_complexity": formula_complexity,
        "step_method": step_method,
        "algebraics_only": bool(algebraics_only),
    }

    if r2 is None and s is not None:
        raise TypeError("r2 must be given when s is given.")

    if r2 is None:
        return _i_algebraic_range(root_order, [1, r1, 1], 0, opts)

    if s is None:
        s = 1

    d_sign = None
    try:
        d_sign = _exact_sign(_to_sympy(d), _Ctx(None))
    except Exception:
        pass
    if d_sign is not None and d_sign < 0:
        raise _failure_lower_bound(d)

    return _i_algebraic_range(root_order, [r1, r2, s], d, opts)
