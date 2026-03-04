"""Core implementation of transcendental_range.

Translates the Wolfram Language resource function TranscendentalRange
into Python, using sympy for symbolic computation and algebraic-range
for generating algebraic numbers.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Union, Optional

import sympy
from sympy import (
    Rational, Integer, S, pi, E, oo,
    exp, log, sin, cos, tan, cot, sec, csc,
    sinh, cosh, tanh, coth, sech, csch,
    asin, acos, atan, acot, asec, acsc,
    asinh, acosh, atanh, acoth, asech, acsch,
    ask, Q, Abs, sqrt, ceiling,
)

from algebraic_range import algebraic_range, formula_complexity

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TranscendentalRangeError(Exception):
    """Base exception for transcendental_range errors."""

class NotAlgebraicError(TranscendentalRangeError):
    """Input parameters are not algebraic numbers."""

# ---------------------------------------------------------------------------
# Numerical helper
# ---------------------------------------------------------------------------

Number = Union[int, float, Fraction, sympy.Basic]

def _nv(expr, prec: int = 15) -> float:
    """Numerical value of a sympy expression (float precision)."""
    try:
        return float(sympy.N(expr, prec))
    except (TypeError, ValueError):
        return float("nan")


def _nv_hp(expr, prec: int = 15) -> sympy.Float:
    """High-precision numerical value as sympy Float."""
    try:
        return sympy.N(expr, prec)
    except (TypeError, ValueError):
        return sympy.Float("nan")


def _dedup_key(expr, prec: int = 15) -> str:
    """Dedup key: string of numerical value rounded to prec digits."""
    try:
        v = sympy.N(expr, prec)
        # Use string representation at requested precision for grouping
        return str(v)
    except (TypeError, ValueError):
        return str(id(expr))

def _to_sympy(x) -> sympy.Basic:
    """Coerce input to a sympy expression."""
    if isinstance(x, sympy.Basic):
        return x
    if isinstance(x, Fraction):
        return Rational(x.numerator, x.denominator)
    if isinstance(x, int):
        return Integer(x)
    if isinstance(x, float):
        return Rational(x).limit_denominator(10**12)
    return sympy.sympify(x)

def _is_algebraic(expr: sympy.Basic) -> bool:
    """Check if a sympy expression is algebraic.

    Returns True if definitely algebraic, False otherwise.
    Conservative: returns False when undecidable (keeps the element).
    """
    if expr.is_number:
        result = ask(Q.algebraic(expr))
        if result is True:
            return True
        if result is False:
            return False
        # Undecidable: try numerical heuristic for known cases
        # Zero and rationals are algebraic
        if expr.is_rational:
            return True
        if expr.is_zero:
            return True
    return False

# ---------------------------------------------------------------------------
# Method registry – 27 transcendental functions
# ---------------------------------------------------------------------------

TRIG = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']
HYP = ['sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch']
INVTRIG = ['asin', 'acos', 'atan', 'acot', 'asec', 'acsc']
INVHYP = ['asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']
ALL_METHODS = ['exp', 'log', 'power'] + TRIG + HYP + INVTRIG + INVHYP

_FUNC_MAP = {
    'exp': exp, 'log': log,
    'sin': sin, 'cos': cos, 'tan': tan,
    'cot': cot, 'sec': sec, 'csc': csc,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'coth': coth, 'sech': sech, 'csch': csch,
    'asin': asin, 'acos': acos, 'atan': atan,
    'acot': acot, 'asec': asec, 'acsc': acsc,
    'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
    'acoth': acoth, 'asech': asech, 'acsch': acsch,
}

# ---------------------------------------------------------------------------
# Turning point constants (for monotonic range splitting)
# ---------------------------------------------------------------------------

def _compute_turning_points():
    """Compute critical-point constants for monotonicOuter splitting."""
    import mpmath as _mp
    old_dps = _mp.mp.dps
    _mp.mp.dps = 50
    try:
        t_sech = _mp.findroot(lambda x: x - _mp.coth(x), 1.2)
        t_arcsech = _mp.findroot(
            lambda x: _mp.asech(x) - 1 / _mp.sqrt(1 - x**2), 0.55)
        t_arccos = _mp.findroot(
            lambda x: _mp.acos(x) - x / _mp.sqrt(1 - x**2), 0.65)
        t_arcsec = _mp.re(_mp.findroot(
            lambda x: _mp.asec(x) - 1 / _mp.sqrt(x**2 - 1),
            _mp.mpf('-1.065')))
        t_power = -1 / _mp.e
    finally:
        _mp.mp.dps = old_dps
    return (sympy.Float(str(t_sech), 30),
            sympy.Float(str(t_arcsech), 30),
            sympy.Float(str(t_arccos), 30),
            sympy.Float(str(t_arcsec), 30),
            -S.One / E)

_TURN_SECH, _TURN_ARCSECH, _TURN_ARCCOS, _TURN_ARCSEC, _TURN_POWER = (
    _compute_turning_points())

EFFICIENT_METHODS = frozenset(
    ['exp', 'log', 'power'] + HYP + INVTRIG + INVHYP
)

# ---------------------------------------------------------------------------
# Domain restrictions (kernel lines 188-207)
# ---------------------------------------------------------------------------

def _domain_check(method: str, a: sympy.Basic) -> bool:
    """Return True if *a* is in the domain of *method*."""
    try:
        av = float(a)
    except (TypeError, ValueError):
        return False
    if method == 'log' or method == 'power':
        return av > 0
    if method == 'coth' or method == 'csch' or method == 'acsch' or method == 'acot':
        return av != 0
    if method == 'acosh':
        return av >= 1
    if method == 'atanh':
        return -1 < av < 1
    if method == 'acoth':
        return abs(av) > 1
    if method == 'asech':
        return 0 < av <= 1
    if method == 'asin' or method == 'acos':
        return abs(av) <= 1
    if method == 'asec' or method == 'acsc':
        return abs(av) >= 1
    # tan/sec: exclude odd multiples of pi/2 — not reachable with algebraic args
    # cot/csc: exclude multiples of pi — not reachable with algebraic args
    return True

# ---------------------------------------------------------------------------
# Generator ranges (kernel lines 134-143)
# ---------------------------------------------------------------------------

def _rational_range(x: sympy.Basic, y: sympy.Basic, z: sympy.Basic) -> list:
    """Arithmetic sequence from x to y with step z (like Wolfram Range)."""
    if z == 0:
        return []
    result = []
    xn, yn, zn = _nv(x), _nv(y), _nv(z)
    if zn > 0:
        cur = x
        while _nv(cur) <= yn + 1e-15:
            result.append(cur)
            cur = cur + z
    elif zn < 0:
        cur = x
        while _nv(cur) >= yn - 1e-15:
            result.append(cur)
            cur = cur + z
    return result


def _farey_sequence(n: int) -> list:
    """Generate Farey sequence F_n as a list of Rationals in [0, 1]."""
    if n <= 0:
        return [Rational(0), Rational(1)]
    a, b, c, d = 0, 1, 1, n
    seq = [Rational(a, b)]
    while c <= n:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        seq.append(Rational(a, b))
    return seq


def _farey_range(x: sympy.Basic, y: sympy.Basic, z: sympy.Basic) -> list:
    """Farey-sequence-based rational range in [x, y].

    Mirrors the WL FareyRange resource function behaviour:
    - if z >= 1: use Farey order z
    - if z == 1/n: use Farey order n
    - if z == -1/n: reverse
    """
    zn = _nv(z)
    if zn >= 1:
        order = int(zn)
    elif isinstance(z, Rational) and z.p == 1:
        order = int(z.q)
    elif isinstance(z, Rational) and z.p == -1:
        order = int(z.q)
    else:
        # fallback
        order = max(1, int(round(1 / abs(zn))))

    farey = _farey_sequence(order)
    xn, yn = _nv(x), _nv(y)
    mn, mx = min(xn, yn), max(xn, yn)

    # Scale Farey fractions to cover [floor(mn), ceil(mx)]
    lo = int(math.floor(mn))
    hi = int(math.ceil(mx))
    result = set()
    for i in range(lo, hi + 1):
        for f in farey:
            val = Rational(i) + f
            vn = float(val)
            if mn - 1e-15 <= vn <= mx + 1e-15:
                result.add(val)
    result = sorted(result, key=float)
    if zn < 0:
        result = list(reversed(result))
    return result


def _gen_range(x, y, z, generators_domain: str, farey: bool) -> list:
    """Build the generator range analogous to WL range[x,y,z,opt]."""
    if generators_domain == 'rationals':
        if farey:
            return _farey_range(x, y, z)
        else:
            return _rational_range(x, y, z)
    else:  # 'algebraics'
        return algebraic_range(x, y, z, farey_range=farey, algebraics_only=False)

# ---------------------------------------------------------------------------
# Naive Outer algorithm (kernel lines 225-265)
# ---------------------------------------------------------------------------

def _elem_naive_range(
    x: sympy.Basic, y: sympy.Basic, z: sympy.Basic,
    method: str, gen_range: list,
) -> list:
    """Single-method naive outer product.

    Computes all coeff * f(arg) where coeff and arg come from gen_range
    (with arg domain-restricted), keeps results in [min, max] that are
    not algebraic.
    """
    # Use sympy min/max for exact range bounds
    mn_s = sympy.Min(x, y)
    mx_s = sympy.Max(x, y)

    is_power = (method == 'power')

    # Coefficient range
    if is_power:
        # For Power: exponents must be non-rational (algebraic irrationals)
        coeff_range = [b for b in gen_range if not b.is_rational]
    else:
        coeff_range = gen_range

    # Argument range: restricted to function domain
    arg_range = [a for a in gen_range if _domain_check(method, a)]

    results = []
    for coeff in coeff_range:
        for arg in arg_range:
            # Compute the transcendental value
            if is_power:
                val = arg ** coeff  # base^exponent (base=arg, exp=coeff)
            else:
                f = _FUNC_MAP[method]
                val = coeff * f(arg)

            # Numerical value for quick checks
            vn = _nv(val)
            if math.isnan(vn) or math.isinf(vn):
                continue

            # In range? Use sympy exact comparison.
            try:
                in_range = bool(val >= mn_s) and bool(val <= mx_s)
            except TypeError:
                in_range = False
            if not in_range:
                continue

            # Not algebraic?
            if _is_algebraic(val):
                continue

            results.append(val)

    return results

# ---------------------------------------------------------------------------
# Extract argument for dedup tiebreaking
# ---------------------------------------------------------------------------

def _extract_arg(expr: sympy.Basic, method: str, prec: int) -> float:
    """Extract |argument to the transcendental function| for dedup tiebreaking."""
    try:
        if method == 'power':
            if expr.is_Pow:
                return abs(float(expr.exp))
            return float('inf')

        f = _FUNC_MAP.get(method)
        if f is None:
            return float('inf')

        if expr.func == f:
            return abs(float(expr.args[0]))

        if expr.is_Mul:
            for factor in expr.args:
                if hasattr(factor, 'func') and factor.func == f:
                    return abs(float(factor.args[0]))
    except (TypeError, ValueError):
        pass
    return float('inf')

# ---------------------------------------------------------------------------
# Combined range (kernel lines 269-287)
# ---------------------------------------------------------------------------

def _combined_range(
    x, y, z, methods: list, gen_range: list, prec: int,
) -> list:
    """Apply efficient/naive outer for each method, merge, dedup.

    For each group of numerically-equal results, keep the one with the
    smallest |argument to the transcendental function|.
    """
    all_results = []  # (expr, float_val, method)
    for m in methods:
        if m in EFFICIENT_METHODS:
            raw, make_expr = _core_range(m, x, y, z, gen_range)
            for ls1e, ls2e, fv in raw:
                all_results.append((make_expr(ls1e, ls2e), fv, m))
        else:
            for e in _elem_naive_range(x, y, z, m, gen_range):
                try:
                    fv = float(e)
                except (TypeError, ValueError):
                    fv = float('nan')
                all_results.append((e, fv, m))

    if not all_results:
        return []

    # Group by float value at requested precision
    groups: dict[str, list] = {}
    for val, fv, m in all_results:
        key = f"{fv:.{prec}g}"
        if key not in groups:
            groups[key] = []
        groups[key].append((val, m))

    # Pick element with smallest |argument| in each group
    deduped = []
    for key, items in groups.items():
        best = min(items, key=lambda vm: _extract_arg(vm[0], vm[1], prec))
        deduped.append(best[0])

    return deduped


def _sort_range(results: list, ascending: bool, prec: int) -> list:
    """Sort results by numerical value."""
    def _key(e):
        try:
            return float(e)
        except (TypeError, ValueError):
            return float('nan')
    return sorted(results, key=_key, reverse=not ascending)

# ---------------------------------------------------------------------------
# Monotonic outer algorithm (kernel lines 295-351)
# ---------------------------------------------------------------------------

def _monotonic_outer(mn_f, mx_f, ls1, ls2, dirs, sign, fval):
    """Optimized double loop — pure float, no sympy in the inner loop.

    Parameters
    ----------
    mn_f, mx_f : float
        Range bounds as floats.
    ls1, ls2 : list
        Generator lists (segments).
    dirs : tuple of (int, int)
        Iteration directions (+1 or -1) for outer and inner loops.
    sign : int
        +1 if fun is increasing toward max, -1 if decreasing toward min.
    fval : callable
        (ls1[i], ls2[j]) -> float.  Fast numerical evaluation.

    Returns
    -------
    list of (ls1_elem, ls2_elem, float_val)
        Lightweight tuples; sympy expression creation is deferred.
    """
    if not ls1 or not ls2:
        return []

    di, dj = dirs
    outer = []

    i_iter = range(len(ls1)) if di == 1 else range(len(ls1) - 1, -1, -1)

    for i in i_iter:
        begin = True
        end = False

        j_iter = (range(len(ls2)) if dj == 1
                  else range(len(ls2) - 1, -1, -1))

        for j in j_iter:
            try:
                vn = fval(ls1[i], ls2[j])
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                if begin and not end:
                    continue
                else:
                    break

            if math.isnan(vn) or math.isinf(vn):
                if begin and not end:
                    continue
                else:
                    break

            in_rg = mn_f <= vn <= mx_f

            if in_rg:
                begin = False
                outer.append((ls1[i], ls2[j], vn))
            else:
                if sign == 1 and vn > mx_f:
                    end = True
                elif sign == -1 and vn < mn_f:
                    end = True
                if begin and not end:
                    continue
                else:
                    break

    return outer

# ---------------------------------------------------------------------------
# Range splitting (kernel lines 360-366)
# ---------------------------------------------------------------------------

def _split_range(lst, split_pts, ascending):
    """Split a sorted list into segments at critical points.

    Returns (segments_dict, n_segments).  Dict is 1-indexed.
    Uses float comparisons for speed (exact splitting is unnecessary here).
    """
    pts_f = sorted(float(p) for p in split_pts)
    bounds_f = [float('-inf')] + pts_f + [float('inf')]
    n = len(bounds_f) - 1
    segments = {}
    for k in range(n):
        lo_f, hi_f = bounds_f[k], bounds_f[k + 1]
        seg = []
        for elem in lst:
            try:
                ef = float(elem)
            except (TypeError, ValueError):
                continue
            if lo_f <= ef <= hi_f:
                seg.append(elem)
        if not ascending:
            seg = list(reversed(seg))
        segments[k + 1] = seg
    return segments, n

# ---------------------------------------------------------------------------
# Range preprocessing (kernel lines 376-397)
# ---------------------------------------------------------------------------

def _prepr_range(gen_range, method, c_alg, c_sing, c_split,
                 a_alg, a_sing, a_split, ascending):
    """Prepare coefficient and argument ranges for monotonicOuter."""
    base = list(gen_range)

    rc = base[:]
    if c_alg is not None:
        rc = [x for x in rc if x != c_alg]
    if c_sing:
        rc = [x for x in rc if x not in c_sing]

    ra = [a for a in base if _domain_check(method, a)]
    if a_alg is not None:
        ra = [a for a in ra if a != a_alg]
    if a_sing:
        ra = [a for a in ra if a not in a_sing]

    cs, n_cs = _split_range(rc, c_split, ascending)
    ags, n_as = _split_range(ra, a_split, ascending)

    return rc, ra, cs, ags, n_cs, n_as

# ---------------------------------------------------------------------------
# Method specifications for efficient path (kernel lines 422-1261)
# ---------------------------------------------------------------------------

_METHOD_SPECS = {
    'exp': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [-S.One],
        'outp': [(2, 1, (1, -1), 1), (2, 2, (1, 1), 1)],
        'outn': [(1, 1, (-1, -1), -1), (1, 2, (-1, 1), -1)],
    },
    'log': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.One, 'a_sing': [], 'a_split': [S.One / E, S.One],
        'outp': [(1, 1, (-1, 1), 1), (1, 2, (-1, -1), 1),
                 (2, 3, (1, 1), 1)],
        'outn': [(2, 1, (1, -1), -1), (2, 2, (1, 1), -1),
                 (1, 3, (-1, 1), -1)],
    },
    'sinh': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [S.Zero],
        'outp': [(1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)],
        'outn': [(2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)],
    },
    'cosh': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [S.Zero],
        'outp': [(2, 2, (1, 1), 1), (2, 1, (1, -1), 1)],
        'outn': [(1, 2, (-1, 1), -1), (1, 1, (-1, -1), -1)],
    },
    'tanh': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [S.Zero],
        'outp': [(2, 2, (1, 1), 1), (1, 1, (-1, -1), 1)],
        'outn': [(1, 2, (-1, 1), -1), (2, 1, (1, -1), -1)],
    },
    'coth': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [S.Zero], 'a_split': [S.Zero],
        'outp': [(2, 2, (1, -1), 1), (1, 1, (-1, 1), 1)],
        'outn': [(1, 2, (-1, -1), -1), (2, 1, (1, 1), -1)],
    },
    'sech': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [],
        'a_split': [-_TURN_SECH, S.Zero, _TURN_SECH],
        'outp': [(2, 4, (1, 1), 1), (2, 3, (1, 1), 1),
                 (2, 2, (1, -1), 1), (2, 1, (1, -1), 1)],
        'outn': [(1, 4, (-1, 1), -1), (1, 3, (-1, 1), -1),
                 (1, 2, (-1, -1), -1), (1, 1, (-1, -1), -1)],
    },
    'csch': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [S.Zero], 'a_split': [S.Zero],
        'outp': [(2, 2, (1, -1), 1), (1, 1, (-1, 1), 1)],
        'outn': [(1, 2, (-1, -1), -1), (2, 1, (1, 1), -1)],
    },
    'asinh': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [S.Zero],
        'outp': [(1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)],
        'outn': [(2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)],
    },
    'acosh': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.One, 'a_sing': [], 'a_split': [],
        'outp': [(2, 1, (1, 1), 1)],
        'outn': [(1, 1, (-1, 1), -1)],
    },
    'atanh': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [S.Zero],
        'outp': [(1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)],
        'outn': [(2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)],
    },
    'acoth': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [-S.One, S.One],
        'a_split': [-S.One, S.One],
        'outp': [(2, 3, (1, -1), 1), (1, 1, (-1, 1), 1)],
        'outn': [(1, 3, (-1, -1), -1), (2, 1, (1, 1), -1)],
    },
    'asech': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.One, 'a_sing': [S.Zero],
        'a_split': [_TURN_ARCSECH],
        'outp': [(2, 1, (1, -1), 1), (2, 2, (1, -1), 1)],
        'outn': [(1, 1, (-1, -1), -1), (1, 2, (-1, -1), -1)],
    },
    'acsch': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [S.Zero], 'a_split': [S.Zero],
        'outp': [(2, 2, (1, -1), 1), (1, 1, (-1, 1), 1)],
        'outn': [(1, 2, (-1, -1), -1), (2, 1, (1, 1), -1)],
    },
    'asin': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [S.Zero],
        'outp': [(1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)],
        'outn': [(2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)],
    },
    'acos': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.One, 'a_sing': [], 'a_split': [_TURN_ARCCOS],
        'outp': [(2, 2, (1, -1), 1), (2, 1, (1, -1), 1)],
        'outn': [(1, 2, (-1, -1), -1), (1, 1, (-1, -1), -1)],
    },
    'atan': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [S.Zero],
        'outp': [(1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)],
        'outn': [(2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)],
    },
    'acot': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [S.Zero], 'a_split': [S.Zero],
        'outp': [(2, 2, (1, -1), 1), (1, 1, (-1, 1), 1)],
        'outn': [(1, 2, (-1, -1), -1), (2, 1, (1, 1), -1)],
    },
    'asec': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.One, 'a_sing': [],
        'a_split': [_TURN_ARCSEC, -S.One, S.One],
        'outp': [(2, 4, (1, 1), 1), (2, 2, (1, 1), 1),
                 (2, 1, (1, 1), 1)],
        'outn': [(1, 4, (-1, 1), -1), (1, 2, (-1, 1), -1),
                 (1, 1, (-1, 1), -1)],
    },
    'acsc': {
        'c_alg': S.Zero, 'c_sing': [], 'c_split': [S.Zero],
        'a_alg': S.Zero, 'a_sing': [], 'a_split': [-S.One, S.One],
        'outp': [(2, 3, (1, -1), 1), (1, 1, (-1, 1), 1)],
        'outn': [(1, 3, (-1, -1), -1), (2, 1, (1, 1), -1)],
    },
}

# ---------------------------------------------------------------------------
# Exact boundary filter (post-filter for float pre-screening)
# ---------------------------------------------------------------------------

_BOUNDARY_EPS = 1e-10

def _exact_boundary_filter(results, mn, mx, mn_f, mx_f):
    """Exact sympy range check for values near float boundaries.

    Values clearly inside the range (by float) are kept.
    Values near mn_f or mx_f are rechecked with exact sympy comparison.
    """
    eps = _BOUNDARY_EPS
    filtered = []
    for expr, fv in results:
        if mn_f + eps < fv < mx_f - eps:
            # Clearly inside
            filtered.append((expr, fv))
        else:
            # Near boundary — exact check
            try:
                if bool(expr >= mn) and bool(expr <= mx):
                    filtered.append((expr, fv))
            except TypeError:
                pass
    return filtered

# ---------------------------------------------------------------------------
# Efficient range computation (kernel lines 1231-1330)
# ---------------------------------------------------------------------------

def _core_range_power(x, y, z, mn_f, mx_f, gen_range, ascending):
    """Efficient range for Power method (base^exponent).

    Returns (raw_tuples, make_expr) where raw_tuples is
    [(base, exponent, float_val), ...] and make_expr creates sympy exprs.
    """
    rc, ra, cs, ags, n_cs, n_as = _prepr_range(
        gen_range, 'power',
        S.Zero, [], [_TURN_POWER, S.Zero],
        S.One, [], [S.One],
        ascending)

    for k in list(cs.keys()):
        cs[k] = [c for c in cs[k] if not c.is_rational]

    # Precompute float caches
    a_fv = {}
    for a in ra:
        try:
            a_fv[a] = float(a)
        except (TypeError, ValueError):
            a_fv[a] = float('nan')

    all_irr = set()
    for seg in cs.values():
        all_irr.update(seg)
    b_fv = {}
    for b in all_irr:
        try:
            b_fv[b] = float(b)
        except (TypeError, ValueError):
            b_fv[b] = float('nan')

    def fval(base, exponent):
        bf = a_fv.get(base, float('nan'))
        ef = b_fv.get(exponent, float('nan'))
        try:
            return bf ** ef
        except (OverflowError, ZeroDivisionError, ValueError):
            return float('nan')

    outp = []
    for as_idx, cs_idx, dirs, sign in [
        (1, 3, (1, 1), 1), (2, 3, (1, 1), 1),
        (1, 2, (1, -1), 1), (2, 2, (1, -1), 1),
        (1, 1, (1, -1), 1), (2, 1, (1, -1), 1),
    ]:
        outp.extend(_monotonic_outer(
            mn_f, mx_f,
            ags.get(as_idx, []), cs.get(cs_idx, []),
            dirs, sign, fval))

    outn = []

    xn, yn = float(x), float(y)
    if min(xn, yn) >= 0:
        result = outp
    elif max(xn, yn) <= 0:
        result = outn
    elif xn <= 0 and yn >= 0:
        result = outn + outp
    else:
        result = outp + outn

    make_expr = lambda base, exponent: base ** exponent
    return result, make_expr


def _core_range(method, x, y, z, gen_range):
    """Dispatch to efficient monotonicOuter-based range computation.

    Returns (raw_tuples, make_expr) where raw_tuples is
    [(ls1_elem, ls2_elem, float_val), ...] and make_expr(ls1, ls2) -> sympy.
    No sympy expressions are created here; creation is deferred to caller.
    """
    ascending = float(z) > 0
    mn = x if ascending else y
    mx = y if ascending else x
    mn_f, mx_f = float(mn), float(mx)

    if method == 'power':
        return _core_range_power(x, y, z, mn_f, mx_f, gen_range, ascending)

    spec = _METHOD_SPECS[method]
    f = _FUNC_MAP[method]

    rc, ra, cs, ags, n_cs, n_as = _prepr_range(
        gen_range, method,
        spec['c_alg'], spec['c_sing'], spec['c_split'],
        spec['a_alg'], spec['a_sing'], spec['a_split'],
        ascending)

    # Cache sympy f(a) evaluations — computed once per argument, not per pair
    fa_sym = {}
    for a in ra:
        fa_sym[a] = f(a)

    # Precompute float caches
    c_fv = {}
    for c in rc:
        try:
            c_fv[c] = float(c)
        except (TypeError, ValueError):
            c_fv[c] = float('nan')

    fa_fv = {}
    for a in ra:
        try:
            fa_fv[a] = float(fa_sym[a])
        except (TypeError, ValueError, OverflowError):
            fa_fv[a] = float('nan')

    def fval(c, a):
        return c_fv.get(c, float('nan')) * fa_fv.get(a, float('nan'))

    def _compute_out(key):
        out = []
        for cs_idx, as_idx, dirs, sign in spec[key]:
            out.extend(_monotonic_outer(
                mn_f, mx_f,
                cs.get(cs_idx, []), ags.get(as_idx, []),
                dirs, sign, fval))
        return out

    # Lazy evaluation: only compute outp/outn when needed (mirrors WL SetDelayed :=)
    xn, yn = float(x), float(y)
    if min(xn, yn) >= 0:
        result = _compute_out('outp')
    elif max(xn, yn) <= 0:
        result = _compute_out('outn')
    elif xn <= 0 and yn >= 0:
        result = _compute_out('outn') + _compute_out('outp')
    else:
        result = _compute_out('outp') + _compute_out('outn')

    # Expression factory with cached f(a) values
    _fa = fa_sym
    make_expr = lambda c, a: c * _fa[a]
    return result, make_expr


def _semifinal_range(method, x, y, z, gen_range, prec):
    """Efficient single-method range: deferred expr creation + dedup + sort.

    Expression creation is deferred until after float-based dedup,
    so only surviving pairs incur sympy overhead.
    """
    ascending = float(z) > 0

    raw, make_expr = _core_range(method, x, y, z, gen_range)
    if not raw:
        return []

    mn = x if ascending else y
    mx = y if ascending else x
    mn_f, mx_f = float(mn), float(mx)
    eps = _BOUNDARY_EPS

    if prec > 15:
        # High precision: need sympy expressions for accurate dedup key
        groups: dict[str, list] = {}
        for ls1e, ls2e, fv in raw:
            expr = make_expr(ls1e, ls2e)
            key = str(sympy.N(expr, prec))
            if key not in groups:
                groups[key] = []
            groups[key].append((expr, fv, ls2e))

        deduped = []
        for items in groups.values():
            best = min(items, key=lambda t: abs(float(t[2])))
            deduped.append((best[0], best[1]))

        # Exact boundary filter
        deduped = _exact_boundary_filter(deduped, mn, mx, mn_f, mx_f)
        deduped.sort(key=lambda vf: vf[1], reverse=not ascending)
        return [expr for expr, fv in deduped]

    # Fast path (prec <= 15): float-based dedup, then create expressions
    groups_fast: dict[str, list] = {}
    for ls1e, ls2e, fv in raw:
        key = f"{fv:.{prec}g}"
        if key not in groups_fast:
            groups_fast[key] = []
        groups_fast[key].append((ls1e, ls2e, fv))

    # Pick pair with smallest |ls2_elem| (= smallest |argument/exponent|)
    deduped_pairs = []
    for items in groups_fast.values():
        best = min(items, key=lambda t: abs(float(t[1])))
        deduped_pairs.append(best)

    # Create sympy expressions only for surviving pairs + boundary filter
    results = []
    for ls1e, ls2e, fv in deduped_pairs:
        if mn_f + eps < fv < mx_f - eps:
            # Clearly in range
            results.append((make_expr(ls1e, ls2e), fv))
        else:
            # Near boundary — create expr and check exactly
            expr = make_expr(ls1e, ls2e)
            try:
                if bool(expr >= mn) and bool(expr <= mx):
                    results.append((expr, fv))
            except TypeError:
                pass

    results.sort(key=lambda vf: vf[1], reverse=not ascending)
    return [expr for expr, fv in results]

# ---------------------------------------------------------------------------
# Post-processing (kernel lines 65-96)
# ---------------------------------------------------------------------------

def _complexity_select(lst: list, threshold: float) -> list:
    """Keep only elements with formula_complexity <= threshold."""
    return [e for e in lst if formula_complexity(e) <= threshold]


def _step_select(lst: list, d) -> list:
    """Greedy step selection: successive elements must differ by >= d."""
    if not lst:
        return []
    dn = _nv(_to_sympy(d))
    result = [lst[0]]
    for e in lst[1:]:
        if abs(_nv(e) - _nv(result[-1])) >= dn - 1e-15:
            result.append(e)
    return result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcendental_range(
    r1: Number,
    r2: Optional[Number] = None,
    s: Optional[Number] = None,
    d: Number = 0,
    *,
    method: Union[str, list] = 'exp',
    generators_domain: str = 'rationals',
    farey_range: bool = False,
    formula_complexity_threshold: float = math.inf,
    working_precision: int = 15,
) -> list:
    """Generate transcendental numbers within a numeric range.

    Parameters
    ----------
    r1 : number
        Single-argument form gives range [1, r1] with step 1.
    r2 : number, optional
        End of range. Two-argument form gives [r1, r2] with step 1.
    s : number, optional
        Step upper bound.
    d : number
        Minimum absolute difference between successive elements (default 0).
    method : str or list of str
        Transcendental function(s) to use. One of: 'exp', 'log', 'power',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'sinh', 'cosh', 'tanh',
        'coth', 'sech', 'csch', 'asin', 'acos', 'atan', 'acot', 'asec',
        'acsc', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch',
        or 'all'. Can also be a list of method names.
    generators_domain : str
        'rationals' (default) or 'algebraics'.
    farey_range : bool
        Use Farey-sequence-based generator range (default False).
    formula_complexity_threshold : float
        Discard elements exceeding this complexity score (default inf).
    working_precision : int
        Decimal digits for numerical comparison (default 15).

    Returns
    -------
    list of sympy.Basic
        Sorted list of transcendental numbers.

    Raises
    ------
    NotAlgebraicError
        If range bounds are not algebraic numbers.
    """
    # Convert inputs
    r1 = _to_sympy(r1)

    # Single-argument form: range [1, r1]
    if r2 is None:
        x, y, z = Integer(1), r1, Integer(1)
    else:
        r2 = _to_sympy(r2)
        x = r1
        y = r2
        z = _to_sympy(s) if s is not None else Integer(1)

    d_sym = _to_sympy(d)

    # Validate: bounds must be algebraic
    for bound, name in [(x, 'r1'), (y, 'r2'), (z, 's')]:
        if bound.is_number:
            alg = ask(Q.algebraic(bound))
            if alg is False:
                raise NotAlgebraicError(
                    f"The range argument '{name}' = {bound} is not algebraic."
                )

    # Handle zero step
    if z == 0:
        return []

    ascending = _nv(z) > 0
    prec = working_precision

    # Resolve method list
    if isinstance(method, str):
        method = method.lower()
        if method == 'all':
            method_list = ALL_METHODS
        else:
            method_list = [method]
    else:
        method_list = [m.lower() for m in method]

    # Build generator range
    gen = _gen_range(x, y, z, generators_domain.lower(), farey_range)

    # Compute the transcendental range
    if len(method_list) == 1:
        m = method_list[0]
        if m in EFFICIENT_METHODS:
            full_range = _semifinal_range(m, x, y, z, gen, prec)
        else:
            raw = _elem_naive_range(x, y, z, m, gen)
            # Dedup: group by numerical value, keep smallest |arg|
            if raw:
                groups: dict[str, list] = {}
                for val in raw:
                    key = _dedup_key(val, prec)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(val)
                deduped = []
                for key, items in groups.items():
                    best = min(items,
                               key=lambda v: _extract_arg(v, m, prec))
                    deduped.append(best)
                raw = deduped
            full_range = _sort_range(raw, ascending, prec)
    else:
        raw = _combined_range(x, y, z, method_list, gen, prec)
        full_range = _sort_range(raw, ascending, prec)

    # Post-processing: complexity filter
    if formula_complexity_threshold < math.inf:
        full_range = _complexity_select(full_range, formula_complexity_threshold)

    # Post-processing: step selection
    if _nv(d_sym) > 0:
        full_range = _step_select(full_range, d_sym)

    return full_range
