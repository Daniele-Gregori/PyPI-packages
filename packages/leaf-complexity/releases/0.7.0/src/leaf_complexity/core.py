"""
Core implementation of leaf_complexity.

Any symbolic expression can be viewed as a tree of heads (operators,
function symbols, containers) and leaves (numbers, symbols). Counting
the leaves is the simplest measure of complexity; leaf_complexity
directly extends the count by weighing each numeric leaf by its
absolute value and then taking the total, with non-numeric leaves
counting 1. More generally, a custom function f can be applied to each
leaf before totaling, and the total itself can be replaced by
recursively applying another unary or binary wrapping function g.

This is a direct translation of the Wolfram Language resource function
LeafComplexity (by the same author), including the option
Heads -> True | False. It follows the same recursive decomposition:
rational numbers are split into numerator and denominator, complex
numeric literals into real and imaginary part, and dictionaries (the
analog of WL Associations) are scanned over their values only.

An important detail of the design, kept from the Wolfram original:
since in the most general usage case one may sometimes want to choose
as wrapping function g a product instead of the default sum, then in
order to avoid returning identically null results, all usage cases are
implemented as recursions starting from initial condition 1 instead of
0. This in practice induces the somewhat awkward "correspondence
principle" that if all leaves are equal to 1 or -1, leaf_complexity
equals the leaf count (heads included) plus 1.

Author: Daniele Gregori
"""

from fractions import Fraction

from sympy import Add, Basic, Float, Integer, Mul, Rational, S
from sympy.core.numbers import NumberSymbol
from sympy.functions.elementary.exponential import exp as _sympy_exp


def _gaussian_parts(expr):
    """Structurally decompose a Gaussian numeric literal a + b*I.

    In the Wolfram Language a numeric complex literal like 3 + 2 I is a
    single Complex atom, whereas SymPy spreads it over Add/Mul nodes.
    This helper detects subexpressions built purely from numeric
    literals and the imaginary unit, returning (re, im) as SymPy
    numbers, or None if expr is not such a literal.
    """
    if isinstance(expr, (Rational, Float)):
        return expr, S.Zero
    if expr is S.ImaginaryUnit:
        return S.Zero, S.One
    if isinstance(expr, Mul):
        coeff = S.One
        n_imaginary = 0
        for arg in expr.args:
            if arg is S.ImaginaryUnit:
                n_imaginary += 1
            elif isinstance(arg, (Rational, Float)):
                coeff = coeff * arg
            else:
                return None
        if n_imaginary > 1:
            return None
        return (coeff, S.Zero) if n_imaginary == 0 else (S.Zero, coeff)
    if isinstance(expr, Add):
        re = im = S.Zero
        for arg in expr.args:
            parts = _gaussian_parts(arg)
            if parts is None:
                return None
            re, im = re + parts[0], im + parts[1]
        return re, im
    return None


def _scan(expr, heads, emit):
    """Recursively scan expr, mirroring the WL Scan definitions.

    emit(value) is called once per visited tree node, in the same
    depth-first, head-first order as Scan[..., expr, Heads -> heads]:
    numeric leaves emit their (signed) value, directed infinities emit
    the infinity itself, and every other node (heads and non-numeric
    leaves) emits 1. With heads=False, heads and parent nodes are not
    visited at all (the plt/splt/pwlt definitions).
    """

    def rec(sub):
        return _scan(sub, heads, emit)

    def head():
        # a head / parent node, visited iff heads=True
        if heads:
            emit(1)

    # --- Python scalars -----------------------------------------------------
    if isinstance(expr, bool):
        emit(1)
    elif isinstance(expr, (int, float)):
        emit(expr)
    elif isinstance(expr, complex):
        # like WL Complex: scan over ReIm
        head()
        rec(expr.real)
        rec(expr.imag)
    elif isinstance(expr, Fraction):
        # like WL Rational: scan over NumeratorDenominator
        head()
        rec(expr.numerator)
        rec(expr.denominator)
    elif isinstance(expr, str):
        emit(1)

    # --- Python containers --------------------------------------------------
    elif isinstance(expr, dict):
        # like WL Association: scan over the values only
        head()
        for value in expr.values():
            rec(value)
    elif isinstance(expr, (list, tuple, set, frozenset)):
        # like WL List
        head()
        for element in expr:
            rec(element)

    # --- SymPy expressions --------------------------------------------------
    elif isinstance(expr, Basic):
        _scan_sympy(expr, emit, rec, head)

    # anything else is an opaque non-numeric atom
    else:
        emit(1)


def _scan_sympy(expr, emit, rec, head):
    """Scan a SymPy expression node."""
    if expr is S.NaN or expr is S.true or expr is S.false:
        emit(1)
        return

    # like WL lt[e_DirectedInfinity]: a single infinite contribution
    if (expr is S.Infinity or expr is S.NegativeInfinity
            or expr is S.ComplexInfinity):
        emit(expr)
        return

    # complex numeric literals, like WL Complex: scan over ReIm
    parts = _gaussian_parts(expr)
    if parts is not None and parts[1] != 0:
        head()
        rec(parts[0])
        rec(parts[1])
        return

    if isinstance(expr, Integer):
        emit(expr)
        return
    if isinstance(expr, Rational):
        # like WL Rational: scan over NumeratorDenominator
        head()
        rec(Integer(expr.p))
        rec(Integer(expr.q))
        return
    if isinstance(expr, Float):
        emit(expr)
        return
    if isinstance(expr, NumberSymbol):
        # pi, E, EulerGamma, ...: numeric symbols, like NumericQ atoms
        emit(expr)
        return

    if isinstance(expr, _sympy_exp):
        # SymPy writes E**z as exp(z); WL keeps Power[E, z]
        head()
        emit(S.Exp1)
        rec(expr.args[0])
        return

    if not expr.args:
        # Symbol and any other atom
        emit(1)
        return

    args = expr.args
    if isinstance(expr, (Add, Mul)):
        # WL collects numeric terms/factors into a single Complex atom
        # (e.g. 3 + 2 I + x has two args in WL, three in SymPy); group
        # Gaussian literal args back together to match
        literal_idx = {
            i for i, arg in enumerate(args) if _gaussian_parts(arg) is not None
        }
        if 2 <= len(literal_idx) < len(args):
            head()
            rec(expr.func(*[args[i] for i in literal_idx]))
            for i, arg in enumerate(args):
                if i not in literal_idx:
                    rec(arg)
            return

    head()
    for arg in args:
        rec(arg)


# The three accumulations below mirror, pairwise through the heads
# flag, the six specialized WL recursions: leafTotal/properLeafTotal,
# scalingLeafTotal/scalingProperLeafTotal and wrappingLeafTotal/
# wrappingProperLeafTotal. Each starts from the initial condition 1
# (see the module docstring), like Block[{s = 1}, ...].

def _leaf_total(expr, heads):
    s = 1

    def emit(value):
        nonlocal s
        s = s + abs(value)

    _scan(expr, heads, emit)
    return s


def _scaling_leaf_total(expr, f, heads):
    s = 1

    def emit(value):
        nonlocal s
        s = s + f(value)

    _scan(expr, heads, emit)
    return s


def _wrapping_leaf_total(expr, f, g, heads):
    s = 1

    def emit(value):
        nonlocal s
        s = g(s, f(value))

    _scan(expr, heads, emit)
    return s


def _normalize(value):
    """Convert exact SymPy numbers back to plain Python numbers."""
    if isinstance(value, Integer):
        return int(value)
    if isinstance(value, Float):
        return float(value)
    return value


def leaf_complexity(expr, f=None, g=None, heads=True):
    """
    Compute a sum or other complexity measure over all atoms of an
    expression.

    Direct translation of the Wolfram Language resource function
    LeafComplexity. In the basic form leaf_complexity(expr) gives the
    sum of all the numeric indivisible subexpressions in expr: each
    numeric leaf contributes its absolute value, and any other node
    (non-numeric leaves and heads) contributes 1. With a function f,
    leaf_complexity(expr, f) applies f to each indivisible
    subexpression (heads and non-numeric leaves as f(1), infinities as
    f of the infinity itself) and takes the total. With a further
    wrapping function g, leaf_complexity(expr, f, g) recursively
    applies s = g(s, f(leaf)) over the leaves instead of summing;
    choosing an additive g recovers the other usage cases.

    All usage cases are recursions starting from the initial condition
    1 instead of 0, so that a multiplicative wrapping g does not return
    identically null results. In particular, if all leaves are equal to
    1 or -1, leaf_complexity(expr) equals the leaf count (heads
    included) plus 1.

    Even if already atomic, rational numbers are decomposed into
    numerator and denominator, complex numeric literals into real and
    imaginary part, and dictionaries are scanned over their values.

    leaf_complexity can work as a measure of complexity for algebraic
    expressions, e.g. as the measure argument of sympy.simplify.

    Parameters
    ----------
    expr : sympy expression, number, or (nested) container
        The expression whose leaves are measured.
    f : callable, optional
        Function applied to each leaf value before totaling. Heads and
        non-numeric leaves contribute f(1); infinities contribute
        f(oo), f(-oo) or f(zoo). If None (default), numeric leaves
        contribute their absolute value and every other node
        contributes 1 (equivalent to f=abs).
    g : callable, optional
        Binary wrapping function recursively applied as g(s, f(leaf))
        in place of the sum s + f(leaf); requires f. The Wolfram
        "unary" usage corresponds to a two-argument callable ignoring
        one of its arguments.
    heads : bool, optional
        If False, only proper leaf nodes are visited (heads and parent
        nodes contribute nothing), like the option Heads -> False.

    Returns
    -------
    int, float, or sympy expression
        The complexity measure. Exact numeric results are returned as
        plain Python numbers; symbolic results (e.g. involving pi or
        E) as SymPy expressions.

    Examples
    --------
    >>> from sympy import symbols, log
    >>> x = symbols('x')
    >>> leaf_complexity(x + 2)
    5
    >>> leaf_complexity(x + 10)
    13
    >>> leaf_complexity(x + 1000000, lambda v: log(v, 10))
    7
    >>> leaf_complexity(x + 1, heads=False)
    3
    """
    if g is not None:
        if f is None:
            raise TypeError("the wrapping function g requires "
                            "a leaf function f")
        if not callable(f):
            raise TypeError(f"f must be callable, got {f!r}")
        if not callable(g):
            raise TypeError(f"g must be callable, got {g!r}")
        return _normalize(_wrapping_leaf_total(expr, f, g, heads))
    if f is not None:
        if not callable(f):
            raise TypeError(f"f must be callable, got {f!r}")
        return _normalize(_scaling_leaf_total(expr, f, heads))
    return _normalize(_leaf_total(expr, heads))
