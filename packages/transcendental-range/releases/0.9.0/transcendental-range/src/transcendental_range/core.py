"""Core implementation of transcendental_range.

A faithful Python port of the Wolfram Language resource function
``TranscendentalRange`` version 1.1.0, following the structure of the
definition notebook section by section:

- Auxiliary functions    -> complexity / step restriction of the output
- Generator ranges       -> ``_base_range`` (Range, AlgebraicRange, FareyRange)
- Method features        -> transcendental types, domains, turning points
- General testing method -> ``_elem_naive_range`` / ``_combined_naive_range`` /
                            ``_sort_naive_range`` (the ``test=True`` baseline,
                            also the default for direct trigonometric methods)
- Monotonic outer        -> ``_monotonic_outer``
- Range preprocessing    -> ``_split_range`` / ``_prepr_range``
- Method specifications  -> ``_METHOD_SPECS``
- Core range             -> ``_compute_outer`` / ``_core_range``
- Main definition        -> ``_transcendental_range_single``,
                            ``_combine_multiplicity``,
                            ``_transcendental_range_multiple``,
                            ``_combine_method``, ``transcendental_range``

Symbolic computation uses sympy; the generator ranges over the algebraics
and the Farey-sequence steps come from the sibling packages
``algebraic-range`` and ``farey`` (ports of the homonymous Wolfram
resource functions used by the original code).

Both the default (monotonic outer) path and the naive testing path share
the same cached float arithmetic and the same in-range decision, so that
their outputs agree bit-for-bit at machine precision — exactly what the
original VerificationTests check.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional, Union

import sympy
from sympy import (
    E, S, Integer, Rational,
    exp, log, sin, cos, tan, cot, sec, csc,
    sinh, cosh, tanh, coth, sech, csch,
    asin, acos, atan, acot, asec, acsc,
    asinh, acosh, atanh, acoth, asech, acsch,
)

from algebraic_range import algebraic_range, formula_complexity
from farey import farey_range as _farey_range_pkg, FareyError

__version__ = "0.9.0"

Number = Union[int, float, Fraction, sympy.Basic]


# ---------------------------------------------------------------------------
# Exceptions (WL: failureNotAlgebraics, failureFareyStep)
# ---------------------------------------------------------------------------

class TranscendentalRangeError(Exception):
    """Base exception for transcendental_range errors."""


class NotAlgebraicError(TranscendentalRangeError):
    """The range arguments provided are not all algebraic numbers."""


class FareyStepError(TranscendentalRangeError):
    """The step parameter is not allowed with the Farey range option."""


# ---------------------------------------------------------------------------
# Numeric helpers, shared by the default and the testing implementation
# ---------------------------------------------------------------------------

def _to_sympy(x) -> sympy.Basic:
    if isinstance(x, sympy.Basic):
        return x
    if isinstance(x, Fraction):
        return Rational(x.numerator, x.denominator)
    if isinstance(x, bool):
        raise TypeError("boolean is not a valid range argument")
    if isinstance(x, int):
        return Integer(x)
    if isinstance(x, float):
        return Rational(Fraction(x).limit_denominator(10 ** 12))
    return sympy.sympify(x)


def _float_of(expr) -> float:
    """Canonical float of a sympy generator element (cached by callers)."""
    try:
        return float(expr)
    except (TypeError, ValueError):
        return float("nan")


# Numeric counterparts of the transcendental heads (math has no sec/csc/...):
_MATH_F = {
    'exp': math.exp,
    'log': math.log,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'cot': lambda v: math.cos(v) / math.sin(v),
    'sec': lambda v: 1.0 / math.cos(v),
    'csc': lambda v: 1.0 / math.sin(v),
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'coth': lambda v: math.cosh(v) / math.sinh(v),
    'sech': lambda v: 1.0 / math.cosh(v),
    'csch': lambda v: 1.0 / math.sinh(v),
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'acot': lambda v: math.atan(1.0 / v),
    'asec': lambda v: math.acos(1.0 / v),
    'acsc': lambda v: math.asin(1.0 / v),
    'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh,
    'acoth': lambda v: math.atanh(1.0 / v),
    'asech': lambda v: math.acosh(1.0 / v),
    'acsch': lambda v: math.asinh(1.0 / v),
}

_FUNC_MAP = {
    'exp': exp, 'log': log,
    'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot, 'sec': sec, 'csc': csc,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'coth': coth, 'sech': sech, 'csch': csch,
    'asin': asin, 'acos': acos, 'atan': atan,
    'acot': acot, 'asec': asec, 'acsc': acsc,
    'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
    'acoth': acoth, 'asech': asech, 'acsch': acsch,
}


def _f_num(method: str, a_f: float) -> float:
    """Float value of f(a), the same in both implementations."""
    try:
        return _MATH_F[method](a_f)
    except (ValueError, OverflowError, ZeroDivisionError):
        return float("nan")


class _Bounds:
    """The in-range decision min <= t <= max, shared by both paths.

    Floats decide away from the boundary; within a small band around the
    bounds the exact sympy value is consulted (WL compares exact numbers).
    """

    __slots__ = ("mn", "mx", "mn_f", "mx_f", "eps")

    def __init__(self, mn, mx):
        self.mn, self.mx = mn, mx
        self.mn_f, self.mx_f = _float_of(mn), _float_of(mx)
        self.eps = 1e-10 * max(1.0, abs(self.mn_f), abs(self.mx_f))

    def position(self, fv: float, expr_thunk) -> int:
        """-1 below, 0 inside, +1 above, 2 undefined."""
        if math.isnan(fv) or math.isinf(fv):
            return 2
        if fv < self.mn_f - self.eps:
            return -1
        if fv > self.mx_f + self.eps:
            return 1
        if self.mn_f + self.eps < fv < self.mx_f - self.eps:
            return 0
        # Boundary band: exact comparison
        try:
            expr = expr_thunk()
            if sympy.N(expr - self.mn, 30) < 0:
                return -1
            if sympy.N(expr - self.mx, 30) > 0:
                return 1
            return 0
        except (TypeError, ValueError):
            return 0 if self.mn_f <= fv <= self.mx_f else 2


def _group_key(fv: float, expr_thunk, prec: int):
    """Dedup key: GroupBy[..., N[#, prec] &].

    At machine precision the float is rounded to 13 significant digits so
    that the same value reached through different representations (whose
    floats may differ in the last bits) falls in one group, as it does for
    the machine-precision N in the original."""
    if prec <= 15:
        return f"{fv if fv else 0.0:.13g}"
    return str(sympy.N(expr_thunk(), prec))


def _canonical_float(expr) -> float:
    """Float of an expression recomputed from its canonical sympy form
    (WL N[expr]); used where the fast float depends on evaluation order."""
    try:
        return float(sympy.N(expr, 17))
    except (TypeError, ValueError):
        return float("nan")


def _is_algebraic(expr) -> bool:
    """Element[expr, Algebraics]: True only when sympy can prove it."""
    if expr.is_rational:
        return True
    return expr.is_algebraic is True


# ---------------------------------------------------------------------------
# Auxiliary functions (WL: complexitySelect, stepSelect, restrictRange)
# ---------------------------------------------------------------------------

def _complexity_select(pairs, threshold):
    """WL complexitySelect: keep elements with formulaComplexity <= c."""
    return [(e, fv) for e, fv in pairs if formula_complexity(e) <= threshold]


def _step_select(pairs, d_f: float):
    """WL stepSelect: successive elements must differ by at least d."""
    if not pairs:
        return []
    out = [pairs[0]]
    for e, fv in pairs[1:]:
        if abs(fv - out[-1][1]) >= d_f - 1e-15:
            out.append((e, fv))
    return out


# ---------------------------------------------------------------------------
# Generator ranges (WL: range[x, y, z, opt] with algebraicRange, fareyRange)
# ---------------------------------------------------------------------------

def _range(x, y, z) -> list:
    """WL Range[x, y, z] with exact arithmetic."""
    if z == 0:
        return []
    try:
        n = int(sympy.floor((y - x) / z))
    except (TypeError, ValueError):
        return []
    if n < 0:
        return []
    return [x + k * z for k in range(n + 1)]


def _to_fraction(v):
    v = _to_sympy(v)
    if v.is_Integer:
        return int(v)
    if v.is_Rational:
        return Fraction(int(v.p), int(v.q))
    return float(v)


def _from_farey(v) -> sympy.Basic:
    if isinstance(v, Fraction):
        return Rational(v.numerator, v.denominator)
    return _to_sympy(v)


def _farey_gen(r1, r2, r3) -> list:
    """WL fareyRange: reroute to the FareyRange resource function (package
    ``farey``), with the intuitive alternative reciprocal step 1/n."""
    try:
        if r3 >= 1:
            vals = _farey_range_pkg(_to_fraction(r1), _to_fraction(r2),
                                    _to_fraction(r3))
        elif r3.is_Rational and r3.p == 1:
            vals = _farey_range_pkg(_to_fraction(r1), _to_fraction(r2),
                                    int(r3.q))
        elif r3.is_Rational and r3.p == -1:
            vals = list(reversed(_farey_range_pkg(
                _to_fraction(r2), _to_fraction(r1), int(r3.q))))
        else:
            raise FareyStepError(
                f"The step parameter {r3} is not allowed with the Farey "
                f"range option.")
    except FareyError as err:
        raise FareyStepError(str(err)) from err
    return [_from_farey(v) for v in vals]


def _base_range(x, y, z, opt: str) -> list:
    """WL range[x, y, z, opt]."""
    if opt == "Rational":
        return _range(x, y, z)
    if opt == "Algebraic":
        return algebraic_range(x, y, z, algebraics_only=False)
    if opt == "RationalFarey":
        return _farey_gen(x, y, z)
    if opt == "AlgebraicFarey":
        return algebraic_range(x, y, z, farey_range=True,
                               algebraics_only=False)
    raise ValueError(f"unknown generators option {opt!r}")


def _opt_generators(generators_domain: str, farey: bool) -> str:
    """WL optGenerators."""
    domain = generators_domain.lower()
    if domain == "rationals":
        plain = "Rational"
    elif domain == "algebraics":
        plain = "Algebraic"
    else:
        raise ValueError(
            "generators_domain must be 'rationals' or 'algebraics', "
            f"got {generators_domain!r}")
    return plain + ("Farey" if farey else "")


# ---------------------------------------------------------------------------
# Method features (WL: $trig, $hyp, ..., domain, turning points)
# ---------------------------------------------------------------------------

TRIG = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']
HYP = ['sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch']
INVTRIG = ['asin', 'acos', 'atan', 'acot', 'asec', 'acsc']
INVHYP = ['asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']
# WL $trsctypes, in the same order:
ALL_METHODS = ['exp', 'log', 'power'] + TRIG + HYP + INVTRIG + INVHYP


def _domain_check(method: str, a_f: float) -> bool:
    """WL domain[f]: is a in the domain of f?

    tan/sec exclude odd multiples of pi/2 and cot/csc multiples of pi;
    for algebraic arguments only 0 is reachable (pi is transcendental).
    """
    if method in ('log', 'power'):
        return a_f > 0
    if method in ('coth', 'csch', 'acsch', 'acot', 'cot', 'csc'):
        return a_f != 0
    if method == 'acosh':
        return a_f >= 1
    if method == 'atanh':
        return -1 < a_f < 1
    if method == 'acoth':
        return abs(a_f) > 1
    if method == 'asech':
        return 0 < a_f <= 1
    if method in ('asin', 'acos'):
        return abs(a_f) <= 1
    if method in ('asec', 'acsc'):
        return abs(a_f) >= 1
    return True


def _compute_turning_points():
    """WL turning[f, 2]: turning points of x f(x), computed by root finding
    (the Sech constant is the fixed point of Coth)."""
    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = 50
    try:
        t_sech = mp.findroot(lambda v: v - mp.coth(v), 1.2)
        t_asech = mp.findroot(
            lambda v: mp.asech(v) - 1 / mp.sqrt(1 - v ** 2), 0.55)
        t_acos = mp.findroot(
            lambda v: mp.acos(v) - v / mp.sqrt(1 - v ** 2), 0.65)
        t_asec = mp.re(mp.findroot(
            lambda v: mp.asec(v) - 1 / mp.sqrt(v ** 2 - 1), mp.mpf('-1.065')))
    finally:
        mp.mp.dps = old
    return (sympy.Float(str(t_sech), 30), sympy.Float(str(t_asech), 30),
            sympy.Float(str(t_acos), 30), sympy.Float(str(t_asec), 30))


_TURN_SECH, _TURN_ASECH, _TURN_ACOS, _TURN_ASEC = _compute_turning_points()
_TURN_POWER = -1 / E  # exact root of D[(-x)^x, x] == 0


# ---------------------------------------------------------------------------
# General testing method (WL: elemNaiveRange, combinedNaiveRange,
# sortNaiveRange) — brute-force Outer, used as baseline via ``test=True``
# and as the default implementation for the direct trigonometric methods
# ---------------------------------------------------------------------------

def _naive_pairs_m1(x, y, z, method, opt, bounds):
    """WL elemNaiveRange[x, y, z, {f, 1}, opt] as deferred tuples.

    Returns (items, make_expr) where items = [(c, a, fv, meas), ...].
    Rational exponents of Power (WL deletes them via the final algebraic
    filter) and arguments where f(a) is provably algebraic (whose products
    the WL DeleteCases[...Algebraics...] removes) are dropped up front.
    """
    base = _base_range(x, y, z, opt)
    is_power = method == 'power'

    if is_power:
        coeffs = [c for c in base if not c.is_rational]  # exponents
    else:
        coeffs = base

    c_f = {c: _float_of(c) for c in coeffs}
    args, a_f, fa_f, fa_sym = [], {}, {}, {}
    f_sym = None if is_power else _FUNC_MAP[method]
    for a in base:
        af = _float_of(a)
        if not _domain_check(method, af):
            continue
        if is_power:
            if a == 1:  # 1^b == 1 is algebraic
                continue
            fa_f[a] = af
            fa_sym[a] = a
        else:
            fs = f_sym(a)
            if _is_algebraic(fs):  # c*f(a) algebraic for all c
                continue
            fa_f[a] = _f_num(method, af)
            fa_sym[a] = fs
        args.append(a)
        a_f[a] = af

    if is_power:
        def make_expr(c, a):
            return a ** c

        def fval(c, a):
            try:
                return fa_f[a] ** c_f[c]
            except (OverflowError, ValueError, ZeroDivisionError):
                return float("nan")

        def measure(c, a):
            return abs(fa_f[a]) + abs(c_f[c])
    else:
        def make_expr(c, a):
            return c * fa_sym[a]

        def fval(c, a):
            return c_f[c] * fa_f[a]

        def measure(c, a):
            return abs(a_f[a])

    items = []
    for c in coeffs:
        if c_f[c] == 0.0:  # 0*f(a) == 0 is algebraic
            continue
        for a in args:
            fv = fval(c, a)
            if bounds.position(fv, lambda: make_expr(c, a)) == 0:
                items.append((c, a, fv, measure(c, a)))
    return items, make_expr


def _combine_op(method):
    """Symbolic combination for the multiplicity option: Plus, or Times for
    Power — where powers over the same base are combined as the WL
    automatic simplification does (the bases are positive by domain)."""
    if method == 'power':
        return lambda u, v: sympy.powsimp(u * v, force=True)
    return lambda u, v: u + v


def _dedup_pairs(triples, prec, canonical=False):
    """WL Values@Map[First@MinimalBy[#, rpl] &]@GroupBy[N[#, prec] &]
    on (expr, fv, meas) triples; returns them deduplicated.

    With ``canonical=True`` the numerical value is recomputed from the
    canonical sympy form: sums assembled in different orders carry
    order-dependent floats, while WL groups by N of the (auto-collected)
    symbolic expression."""
    best = {}
    for expr, fv, meas in triples:
        if canonical:
            fv = _canonical_float(expr)
        key = _group_key(fv, lambda e=expr: e, prec)
        cur = best.get(key)
        if cur is None or meas < cur[2]:
            best[key] = (expr, fv, meas)
    return list(best.values())


def _dedup_deferred(items, make_expr, prec):
    """Deduplicate deferred (e1, e2, fv, meas) tuples by numerical value,
    keeping the minimal measure; expressions are built only for the
    surviving representatives."""
    best = {}
    for e1, e2, fv, meas in items:
        key = _group_key(fv, lambda a=e1, b=e2: make_expr(a, b), prec)
        cur = best.get(key)
        if cur is None or meas < cur[3]:
            best[key] = (e1, e2, fv, meas)
    return [(make_expr(e1, e2), fv, meas)
            for e1, e2, fv, meas in best.values()]


def _elem_naive_range(x, y, z, method, m, opt, prec):
    """WL elemNaiveRange[x, y, z, {f, m}, opt, prec] -> (expr, fv, meas).

    The numerical-value deduplication of combinedNaiveRange is anticipated
    on the deferred tuples, so that symbolic expressions are only built
    for the surviving representatives (the resulting set is the same)."""
    bounds = _Bounds(sympy.Min(x, y), sympy.Max(x, y))
    items, make_expr = _naive_pairs_m1(x, y, z, method, opt, bounds)

    if m == 1:
        return _dedup_deferred(items, make_expr, prec)

    # Multiplicity > 1: iterated Outer[Plus|Times, combined, base]
    base = _dedup_deferred(items, make_expr, prec)
    op = _combine_op(method)
    fop = (lambda u, v: u * v) if method == 'power' else (lambda u, v: u + v)

    combined = base
    for _ in range(m - 1):
        raw = []
        for e1, f1, m1 in combined:
            for e2, f2, m2 in base:
                fv = fop(f1, f2)
                if bounds.position(fv, lambda: op(e1, e2)) == 0:
                    raw.append((op(e1, e2), fv, m1 + m2))
        combined = _dedup_pairs(raw, prec, canonical=True)

    tiny = 10.0 ** (-prec + 1)
    return [(e, fv, meas) for e, fv, meas in combined
            if not (_is_algebraic(e) or abs(fv) < tiny)]


def _combined_naive_range(x, y, z, funs, m, opt, prec):
    """WL combinedNaiveRange: join the elementary naive ranges and
    deduplicate by numerical value, keeping the smallest argument."""
    fun_list = ALL_METHODS if funs == ['all'] else funs
    joined = []
    for f in fun_list:
        joined.extend(_elem_naive_range(x, y, z, f, m, opt, prec))
    return _dedup_pairs(joined, prec)


def _sort_naive_range(x, y, z, fun, m, opt, prec):
    """WL sortNaiveRange -> [(expr, fv), ...] sorted by numerical value."""
    funs = fun if isinstance(fun, list) else [fun]
    deduped = _combined_naive_range(x, y, z, funs, m, opt, prec)
    ascending = _float_of(z) > 0
    deduped.sort(key=lambda t: t[1], reverse=not ascending)
    return [(e, fv) for e, fv, _ in deduped]


# ---------------------------------------------------------------------------
# Monotonic outer (WL: monotonicOuter) — bounded monotone scan of the
# outer product, breaking out as soon as the values leave the range
# ---------------------------------------------------------------------------

def _monotonic_outer(fnum, make_expr, bounds, ls1, ls2, dirs, sign):
    """WL monotonicOuter[fun, {min, max}, ls1, ls2, {mi, mj}, sign].

    Returns lightweight (e1, e2, fv) tuples; sympy expression creation
    is deferred to the caller (only boundary checks build one early).
    """
    if not ls1 or not ls2:
        return []
    mi, mj = dirs
    out = []
    idx1 = range(len(ls1)) if mi == 1 else range(len(ls1) - 1, -1, -1)
    for i in idx1:
        e1 = ls1[i]
        begin, end = True, False
        idx2 = range(len(ls2)) if mj == 1 else range(len(ls2) - 1, -1, -1)
        for j in idx2:
            e2 = ls2[j]
            fv = fnum(e1, e2)
            pos = bounds.position(fv, lambda: make_expr(e1, e2))
            if pos == 0:
                begin = False
                out.append((e1, e2, fv))
            else:
                if pos == 1 and sign == 1:
                    end = True
                elif pos == -1 and sign == -1:
                    end = True
                if begin and not end:
                    continue  # redundant region before entering the range
                break
    return out


# ---------------------------------------------------------------------------
# Range preprocessing (WL: splitRange, preprRange)
# ---------------------------------------------------------------------------

def _split_range(lst, floats, split_pts, ascending):
    """WL splitRange: split a range into segments at the given points
    (boundary points fall in both adjacent segments); segments are
    normalized to ascending order for the monotonic scan directions."""
    pts = sorted(_float_of(p) for p in split_pts)
    bnds = [float("-inf")] + pts + [float("inf")]
    segments = {}
    for k in range(len(bnds) - 1):
        seg = [e for e in lst if bnds[k] <= floats[e] <= bnds[k + 1]]
        if not ascending:
            seg.reverse()
        segments[k + 1] = seg
    return segments


def _prepr_range(x, y, z, opt, method, spec, ascending):
    """WL preprRange: coefficient and argument ranges for monotonicOuter,
    split independently at their own critical points."""
    base = _base_range(x, y, z, opt)
    floats = {e: _float_of(e) for e in base}

    c_alg, c_sing, c_split = spec["coeff"]
    a_alg, a_sing, a_split = spec["arg"]

    rc = [e for e in base if not (c_alg is not None and e == c_alg)
          and e not in c_sing]
    ra = [e for e in base if _domain_check(method, floats[e])
          and not (a_alg is not None and e == a_alg) and e not in a_sing]

    cs = _split_range(rc, floats, c_split, ascending)
    as_ = _split_range(ra, floats, a_split, ascending)
    return cs, as_, floats


# ---------------------------------------------------------------------------
# Method specifications (WL: $methodSpecs, version 1.1.0)
#
# "coeff" -> (cAlg, cSing, cSplit): coefficient preprocessing parameters
# "arg"   -> (aAlg, aSing, aSplit): argument preprocessing parameters
# "outp"/"outn" -> (ls1Idx, ls2Idx, (mi, mj), sign): monotonicOuter tuples
#                  for the positive/negative part of the range
# For non-Power methods ls1 = coefficient segments, ls2 = argument segments;
# for Power they are swapped (ls1 = argument = base, ls2 = coeff = exponent).
# ---------------------------------------------------------------------------

_METHOD_SPECS = {
    'exp': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (-S.One,)),
        "outp": ((2, 1, (1, -1), 1), (2, 2, (1, 1), 1)),
        "outn": ((1, 1, (-1, -1), -1), (1, 2, (-1, 1), -1)),
    },
    'log': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.One, (), (1 / E, S.One)),
        "outp": ((1, 1, (-1, 1), -1), (1, 2, (-1, -1), 1),
                 (2, 3, (1, 1), 1)),
        "outn": ((2, 1, (1, -1), -1), (2, 2, (1, 1), 1),
                 (1, 3, (-1, 1), -1)),
    },
    'sinh': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (S.Zero,)),
        "outp": ((1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)),
        "outn": ((2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)),
    },
    'cosh': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (S.Zero,)),
        "outp": ((2, 2, (1, 1), 1), (2, 1, (1, -1), 1)),
        "outn": ((1, 2, (-1, 1), -1), (1, 1, (-1, -1), -1)),
    },
    'tanh': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (S.Zero,)),
        "outp": ((2, 2, (1, 1), 1), (1, 1, (-1, -1), 1)),
        "outn": ((1, 2, (-1, 1), -1), (2, 1, (1, -1), -1)),
    },
    'coth': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (S.Zero,), (S.Zero,)),
        "outp": ((2, 2, (1, -1), 1), (1, 1, (-1, 1), 1)),
        "outn": ((1, 2, (-1, -1), -1), (2, 1, (1, 1), -1)),
    },
    'sech': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (-_TURN_SECH, S.Zero, _TURN_SECH)),
        "outp": ((2, 4, (1, 1), 1), (2, 3, (1, 1), 1),
                 (2, 2, (1, -1), 1), (2, 1, (1, -1), 1)),
        "outn": ((1, 4, (-1, 1), -1), (1, 3, (-1, 1), -1),
                 (1, 2, (-1, -1), -1), (1, 1, (-1, -1), -1)),
    },
    'csch': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (S.Zero,), (S.Zero,)),
        "outp": ((2, 2, (1, 1), -1), (1, 1, (-1, -1), -1)),
        "outn": ((1, 2, (-1, 1), 1), (2, 1, (1, -1), 1)),
    },
    'asinh': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (S.Zero,)),
        "outp": ((1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)),
        "outn": ((2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)),
    },
    'acosh': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.One, (), ()),
        "outp": ((2, 1, (1, 1), 1),),
        "outn": ((1, 1, (-1, 1), -1),),
    },
    'atanh': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (S.Zero,)),
        "outp": ((1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)),
        "outn": ((2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)),
    },
    'acoth': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (-S.One, S.One), (-S.One, S.One)),
        "outp": ((2, 3, (1, -1), 1), (1, 1, (-1, 1), 1)),
        "outn": ((1, 3, (-1, -1), -1), (2, 1, (1, 1), -1)),
    },
    'asech': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.One, (S.Zero,), (_TURN_ASECH,)),
        "outp": ((2, 1, (1, -1), 1), (2, 2, (1, -1), 1)),
        "outn": ((1, 1, (-1, -1), -1), (1, 2, (-1, -1), -1)),
    },
    'acsch': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (S.Zero,), (S.Zero,)),
        "outp": ((2, 2, (1, -1), 1), (1, 1, (-1, 1), 1)),
        "outn": ((1, 2, (-1, -1), -1), (2, 1, (1, 1), -1)),
    },
    'asin': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (S.Zero,)),
        "outp": ((1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)),
        "outn": ((2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)),
    },
    'acos': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.One, (), (_TURN_ACOS,)),
        "outp": ((2, 2, (1, -1), 1), (2, 1, (1, -1), 1)),
        "outn": ((1, 2, (-1, -1), -1), (1, 1, (-1, -1), -1)),
    },
    'atan': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (S.Zero,)),
        "outp": ((1, 1, (-1, -1), 1), (2, 2, (1, 1), 1)),
        "outn": ((2, 1, (1, -1), -1), (1, 2, (-1, 1), -1)),
    },
    'acot': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (S.Zero,), (S.Zero,)),
        "outp": ((2, 2, (1, -1), 1), (1, 1, (-1, 1), 1)),
        "outn": ((1, 2, (-1, -1), -1), (2, 1, (1, 1), -1)),
    },
    'asec': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.One, (), (_TURN_ASEC, -S.One, S.One)),
        "outp": ((2, 4, (1, 1), 1), (2, 2, (1, 1), 1), (2, 1, (1, 1), 1)),
        "outn": ((1, 4, (-1, 1), -1), (1, 2, (-1, 1), -1),
                 (1, 1, (-1, 1), -1)),
    },
    'acsc': {
        "coeff": (S.Zero, (), (S.Zero,)),
        "arg": (S.Zero, (), (-S.One, S.One)),
        "outp": ((2, 3, (1, -1), 1), (1, 1, (-1, 1), 1)),
        "outn": ((1, 3, (-1, -1), -1), (2, 1, (1, 1), -1)),
    },
    'power': {
        "coeff": (S.Zero, (), (_TURN_POWER, S.Zero)),
        "arg": (S.One, (), (S.One,)),
        "outp": ((1, 3, (1, 1), 1), (2, 3, (1, 1), 1),
                 (1, 2, (1, -1), 1), (2, 2, (1, -1), 1),
                 (1, 1, (1, -1), 1), (2, 1, (1, -1), 1)),
        "outn": (),
    },
}

EFFICIENT_METHODS = frozenset(_METHOD_SPECS)


# ---------------------------------------------------------------------------
# Core range (WL: computeOuter, coreRange)
# ---------------------------------------------------------------------------

def _compute_outer(fnum, make_expr, bounds, ls1, ls2, tuples):
    """WL computeOuter: join the monotonicOuter scans of the given tuples."""
    out = []
    for i1, i2, dirs, sign in tuples:
        out.extend(_monotonic_outer(fnum, make_expr, bounds,
                                    ls1.get(i1, []), ls2.get(i2, []),
                                    dirs, sign))
    return out


def _core_range(method, x, y, z, opt):
    """WL coreRange[{f, 1}, x, y, z, opt].

    Returns (raw, make_expr, measure) with raw = [(e1, e2, fv), ...];
    expression creation is deferred until after deduplication.
    """
    spec = _METHOD_SPECS[method]
    ascending = _float_of(z) > 0
    mn, mx = (x, y) if ascending else (y, x)
    bounds = _Bounds(mn, mx)
    is_power = method == 'power'

    cs, as_, floats = _prepr_range(x, y, z, opt, method, spec, ascending)

    if is_power:
        # Filter coefficient (exponent) segments to non-rationals
        for k in list(cs):
            cs[k] = [c for c in cs[k] if not c.is_rational]
        ls1, ls2 = as_, cs  # ls1 = base segments, ls2 = exponent segments

        def make_expr(b, e):
            return b ** e

        def fnum(b, e):
            try:
                return floats[b] ** floats[e]
            except (OverflowError, ValueError, ZeroDivisionError):
                return float("nan")

        def measure(b, e):
            return abs(floats[b]) + abs(floats[e])
    else:
        ls1, ls2 = cs, as_
        fa_sym = {}
        fa_f = {}
        f_sym = _FUNC_MAP[method]
        for seg in as_.values():
            for a in seg:
                if a not in fa_sym:
                    fa_sym[a] = f_sym(a)
                    fa_f[a] = _f_num(method, floats[a])

        def make_expr(c, a):
            return c * fa_sym[a]

        def fnum(c, a):
            return floats[c] * fa_f[a]

        def measure(c, a):
            return abs(floats[a])

    x_f, y_f = _float_of(x), _float_of(y)
    outp = lambda: _compute_outer(fnum, make_expr, bounds, ls1, ls2,
                                  spec["outp"])
    outn = lambda: _compute_outer(fnum, make_expr, bounds, ls1, ls2,
                                  spec["outn"])

    if 0 <= x_f <= y_f or 0 <= y_f <= x_f:
        raw = outp()
    elif x_f <= y_f <= 0 or y_f <= x_f <= 0:
        raw = outn()
    elif x_f <= 0 and y_f >= 0:
        raw = outn() + outp()
    else:
        raw = outp() + outn()
    return raw, make_expr, measure


# ---------------------------------------------------------------------------
# Main definition (WL: transcendentalRangeSingle, combineMultiplicity,
# transcendentalRangeMultiple, combineMethod, TranscendentalRange)
# ---------------------------------------------------------------------------

def _transcendental_range_single(method, x, y, z, opt, prec):
    """WL transcendentalRangeSingle: core range, dedup by numerical value
    keeping the minimal argument, sort by numerical value."""
    raw, make_expr, measure = _core_range(method, x, y, z, opt)

    best = {}
    for e1, e2, fv in raw:
        key = _group_key(fv, lambda: make_expr(e1, e2), prec)
        meas = measure(e1, e2)
        cur = best.get(key)
        if cur is None or meas < cur[3]:
            best[key] = (e1, e2, fv, meas)

    ascending = _float_of(z) > 0
    deduped = sorted(best.values(), key=lambda t: t[2],
                     reverse=not ascending)
    return [(make_expr(e1, e2), fv, meas) for e1, e2, fv, meas in deduped]


def _combine_multiplicity(triples, method, order, x, y, z, prec):
    """WL combineMultiplicity: monotone Plus (Times for Power) combinations
    of the single range with itself, up to the given multiplicity."""
    if order == 1:
        return triples
    lower = _combine_multiplicity(triples, method, order - 1, x, y, z, prec)

    bounds = _Bounds(sympy.Min(x, y), sympy.Max(x, y))
    ascending = _float_of(z) > 0
    ls1 = lower if ascending else list(reversed(lower))
    ls2 = triples if ascending else list(reversed(triples))
    dirs = (1, 1) if ascending else (-1, -1)
    sign = 1 if ascending else -1

    op = _combine_op(method)
    fop = (lambda u, v: u * v) if method == 'power' else (lambda u, v: u + v)

    def fnum(t1, t2):
        return fop(t1[1], t2[1])

    def make_expr(t1, t2):
        return op(t1[0], t2[0])

    raw = _monotonic_outer(fnum, make_expr, bounds, ls1, ls2, dirs, sign)
    return _dedup_pairs(
        [(make_expr(t1, t2), fv, t1[2] + t2[2]) for t1, t2, fv in raw],
        prec, canonical=True)


def _transcendental_range_multiple(triples, method, multi, x, y, z, prec):
    """WL transcendentalRangeMultiple: combine, dedup, sort and delete
    the algebraic or numerically tiny elements."""
    combined = _combine_multiplicity(triples, method, multi, x, y, z, prec)
    ascending = _float_of(z) > 0
    combined.sort(key=lambda t: t[1], reverse=not ascending)
    tiny = 10.0 ** (-prec + 1)
    return [(e, fv) for e, fv, _ in combined
            if not (_is_algebraic(e) or abs(fv) < tiny)]


def _combine_method(funs, x, y, z, d, opts):
    """WL combineMethod: map TranscendentalRange over the methods and
    deduplicate the merged ranges by numerical value (first method wins)."""
    prec = opts["working_precision"]
    merged = []
    for f in funs:
        sub = transcendental_range(x, y, z, d, method=f, **opts)
        merged.extend(sub)
    best = {}
    for e in merged:
        fv = float(sympy.N(e, 15))
        key = _group_key(fv, lambda e=e: e, prec)
        if key not in best:
            best[key] = (e, fv)
    ascending = _float_of(z) > 0
    return sorted(best.values(), key=lambda t: t[1],
                  reverse=not ascending)


def transcendental_range(
    r1: Number,
    r2: Optional[Number] = None,
    s: Optional[Number] = None,
    d: Number = 0,
    *,
    method: Union[str, list] = 'exp',
    multiplicity: int = 1,
    generators_domain: str = 'rationals',
    farey_range: bool = False,
    formula_complexity_threshold: float = math.inf,
    working_precision: int = 15,
    test: bool = False,
) -> List[sympy.Basic]:
    """Generate transcendental numbers within a numeric range.

    Port of the Wolfram Language resource function ``TranscendentalRange``
    version 1.1.0.

    Parameters
    ----------
    r1 : number
        Single-argument form gives the range [1, r1] with step 1.
    r2 : number, optional
        End of the range; two-argument form gives [r1, r2] with step 1.
    s : number, optional
        Step of the generator range (negative for descending output).
    d : number, default 0
        Lower bound on the difference between successive elements.
    method : str or list of str, default ``'exp'``
        Transcendental function generating the numbers: one of ``'exp'``,
        ``'log'``, ``'power'``, the trigonometric, hyperbolic, inverse
        trigonometric and inverse hyperbolic function names, a list of
        them, or ``'all'``.
    multiplicity : int, default 1
        Number of terms to combine linearly (multiplicatively for
        ``'power'``), as in the WL option "Multiplicity".
    generators_domain : str, default ``'rationals'``
        ``'rationals'`` or ``'algebraics'``: the domain of the generator
        arguments and coefficients (WL option "GeneratorsDomain").
    farey_range : bool, default False
        Set the step denominators as in the Farey sequence
        (WL option "FareyRange").
    formula_complexity_threshold : number, default ``math.inf``
        Discard elements whose formula complexity exceeds this
        (WL option "FormulaComplexity").
    working_precision : int, default 15
        Decimal digits for all internal numerical evaluations
        (WL option WorkingPrecision; 15 corresponds to MachinePrecision).
    test : bool, default False
        Development-only option: use the naive Outer-based testing
        implementation instead of the monotonic-outer one
        (WL option "Test").

    Returns
    -------
    list of sympy.Basic
        Exact transcendental numbers sorted by numerical value
        (descending for negative step).

    Raises
    ------
    NotAlgebraicError
        If the range arguments are not all algebraic numbers.
    FareyStepError
        If the step is not allowed with the Farey range option.
    """
    # TranscendentalRange[x] := TranscendentalRange[1, x, 1, 0]
    x = _to_sympy(r1)
    if r2 is None:
        x, y, z = Integer(1), x, Integer(1)
    else:
        y = _to_sympy(r2)
        z = _to_sympy(s) if s is not None else Integer(1)
    d_sym = _to_sympy(d)

    # If !Element[{x, y, z}, Algebraics], Return@failureNotAlgebraics
    for v in (x, y, z):
        if v.is_number and v.is_algebraic is False:
            raise NotAlgebraicError(
                f"The range arguments provided {(x, y, z)} are not all "
                f"algebraic numbers.")

    if not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError(
            f"multiplicity must be a positive integer, got {multiplicity!r}")

    if z == 0:
        return []

    prec = working_precision
    opt = _opt_generators(generators_domain, farey_range)

    if isinstance(method, str):
        meth = method.lower()
        method_list = ALL_METHODS if meth == 'all' else None
        single = None if meth == 'all' else meth
    else:
        method_list = [m.lower() for m in method]
        single = None

    if single is not None and single not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")

    # Full range
    if test or single in TRIG:
        fun = method_list if method_list is not None else single
        fullrange = _sort_naive_range(x, y, z, fun, multiplicity, opt, prec)
    elif single is not None:  # KeyExistsQ[$methodSpecs, optmeth]
        triples = _transcendental_range_single(single, x, y, z, opt, prec)
        if multiplicity == 1:
            fullrange = [(e, fv) for e, fv, _ in triples]
        else:
            fullrange = _transcendental_range_multiple(
                triples, single, multiplicity, x, y, z, prec)
    else:  # ListQ[optmeth] or All
        opts = dict(multiplicity=multiplicity,
                    generators_domain=generators_domain,
                    farey_range=farey_range,
                    formula_complexity_threshold=formula_complexity_threshold,
                    working_precision=working_precision, test=test)
        fullrange = _combine_method(method_list, x, y, z, d_sym, opts)

    # Restrict by formula complexity, then by the step lower bound
    if formula_complexity_threshold < math.inf:
        fullrange = _complexity_select(fullrange,
                                       formula_complexity_threshold)
    d_f = _float_of(d_sym)
    if d_f > 0:
        fullrange = _step_select(fullrange, d_f)

    return [e for e, _ in fullrange]
