"""Tests for the leaf-complexity package.

The TestBasicExamples, TestScope, TestCustomWrapping, TestHeadsOption,
TestApplications and TestPropertiesAndRelations classes translate, one
by one, all the examples of the original Wolfram Language LeafComplexity
definition notebook; the expected values are the ones recorded there.
The remaining classes add Heads -> False variants, edge cases and
Python-specific coverage, whose expected values were computed with the
original Wolfram Language definitions.
"""

import pytest
from fractions import Fraction

import sympy
from sympy import (
    Abs, E, Function, I, Mod, Mul, Rational, S, exp, log, nan, oo, pi,
    simplify, sqrt, symbols, zoo,
)

from leaf_complexity import leaf_complexity

x, y, a, b = symbols("x y a b")


# ===========================================================================
# Wolfram Language notebook — Basic Examples
# ===========================================================================

class TestBasicExamples:
    def test_same_leaf_count_different_complexity(self):
        """Expressions with the same LeafCount may have different total
        over their leaves. WL: LeafComplexity[x + 2] == 5,
        LeafComplexity[x + 10] == 13."""
        assert leaf_complexity(x + 2) == 5
        assert leaf_complexity(x + 10) == 13

    def test_custom_function_log10(self):
        """WL: LeafComplexity[x + 1000000, Log10] == 7"""
        assert leaf_complexity(x + 1000000, lambda v: log(v, 10)) == 7


# ===========================================================================
# Wolfram Language notebook — Scope
# ===========================================================================

class TestScope:
    def test_rational_expression(self):
        """WL: LeafComplexity[(x + 2)/(y - 2)] == 12"""
        assert leaf_complexity((x + 2) / (y - 2)) == 12

    def test_algebraic_expression(self):
        """WL: LeafComplexity[(Sqrt[x] - 1)/(y^(1/5) + 2)] == 24"""
        expr = (sqrt(x) - 1) / (y**Rational(1, 5) + 2)
        assert leaf_complexity(expr) == 24

    def test_complex_expression(self):
        """WL: LeafComplexity[(x + I)/(x - 2 - 2 I) - 4 I/x^2] == 26"""
        expr = (x + I) / (x - 2 - 2 * I) - 4 * I / x**2
        assert leaf_complexity(expr) == 26

    def test_floats(self):
        """WL: LeafComplexity[1.5 x^2 + 1.1 Log[y]] == 12.6"""
        expr = 1.5 * x**2 + 1.1 * log(y)
        assert leaf_complexity(expr) == pytest.approx(12.6)

    def test_symbolic_constants(self):
        """WL: LeafComplexity[E^x + Pi^2] == 7 + E + Pi"""
        assert leaf_complexity(E**x + pi**2) == 7 + E + pi

    def test_nested_list(self):
        """WL: LeafComplexity[{1, 2, {4, 0, 1, {0, 6, 3, 4, {2, 3, 0},
        {5, 2, 0}, 1}, 0}}] == 40"""
        expr = [1, 2, [4, 0, 1, [0, 6, 3, 4, [2, 3, 0], [5, 2, 0], 1], 0]]
        assert leaf_complexity(expr) == 40

    def test_nested_association(self):
        """WL: LeafComplexity[<|a -> {2, 3}, b -> <|{0, 1} -> <|gamma
        -> 20, delta -> 40|>, {-1, -2} -> 50|>|>] == 120 (values only)"""
        expr = {
            "a": [2, 3],
            "b": {(0, 1): {"gamma": 20, "delta": 40}, (-1, -2): 50},
        }
        assert leaf_complexity(expr) == 120

    def test_custom_function_abs_log(self):
        """WL: LeafComplexity[Exp[-Exp[-Exp[x]]], Abs@Log[#]&]
        == 4 + 2 Pi"""
        expr = exp(-exp(-exp(x)))
        assert leaf_complexity(expr, lambda v: Abs(log(v))) == 4 + 2 * pi

    def test_custom_function_mod(self):
        """WL: LeafComplexity[{a[2], a[E], a[4], a[8], a[E^2], a[16]},
        If[Mod[#, 2] =!= 0, 0, Log2[#]]&] == 12"""
        func = Function("a")
        expr = [func(2), func(E), func(4), func(8), func(E**2), func(16)]

        def f(v):
            return log(v, 2) if Mod(v, 2) == 0 else 0

        assert leaf_complexity(expr, f) == 12


# ===========================================================================
# Wolfram Language notebook — Scope: custom wrapping over Range[5]
# ===========================================================================

class TestCustomWrapping:
    RANGE = [1, 2, 3, 4, 5]

    def test_wrapping_times(self):
        """WL: LeafComplexity[Range[5], Exp, Times] == E^16"""
        assert leaf_complexity(self.RANGE, exp, Mul) == exp(16)

    def test_wrapping_exp_of_second(self):
        """WL: LeafComplexity[Range[5], Exp[#]&, Exp[#2]&] == E^E^5"""
        result = leaf_complexity(self.RANGE, exp, lambda s, v: exp(v))
        assert result == exp(exp(5))

    def test_wrapping_exp_of_first(self):
        """WL: LeafComplexity[Range[5], Exp[#]&, Exp[#1]&]
        == E^E^E^E^E^E"""
        result = leaf_complexity(self.RANGE, exp, lambda s, v: exp(s))
        assert result == exp(exp(exp(exp(exp(E)))))

    def test_wrapping_times_proper(self):
        """WL: wrappingProperLeafTotal[Range[5], Exp, Times] == E^15
        (the List head is not wrapped)."""
        assert leaf_complexity(self.RANGE, exp, Mul, heads=False) == exp(15)

    def test_wrapping_additive_recovers_scaling(self):
        """Setting g to Plus recovers the other definitions up to two
        arguments (see the notebook Details)."""
        expr = (sqrt(x) - 1) / (y**Rational(1, 5) + 2)
        assert (leaf_complexity(expr, Abs, lambda s, v: s + v)
                == leaf_complexity(expr, Abs)
                == leaf_complexity(expr))


# ===========================================================================
# Wolfram Language notebook — Options: Heads
# ===========================================================================

class TestHeadsOption:
    def test_heads_false_basic(self):
        """WL: LeafComplexity[x + 1, Heads -> False] == 3 (vs 4)"""
        assert leaf_complexity(x + 1, heads=False) == 3
        assert leaf_complexity(x + 1) == 4

    def test_heads_false_discards_four_heads(self):
        """WL: (x + 1)/(y - 3) has LeafComplexity 8 with Heads -> False
        and 4 discarded heads (Times, Plus, Power, Plus)."""
        expr = (x + 1) / (y - 3)
        leaftot = leaf_complexity(expr, heads=False)
        assert leaftot == 8
        assert leaf_complexity(expr) - leaftot == 4


# ===========================================================================
# Wolfram Language notebook — Applications
# ===========================================================================

class TestApplications:
    def test_simplify_measure(self):
        """leaf_complexity works as the measure of sympy.simplify, like
        ComplexityFunction -> LeafComplexity in WL Simplify:
        WL: Simplify[(-4 - 4 a b^2 - 2 a^2 b)/(a b),
        ComplexityFunction -> LeafComplexity] == 2 (-a - 2/(a b) - 2 b),
        with LeafComplexity 21 vs 23 of the default result."""
        expr = (-4 - 4 * a * b**2 - 2 * a**2 * b) / (a * b)
        default = simplify(expr)
        steered = simplify(expr, measure=leaf_complexity)
        assert sympy.simplify(steered - expr) == 0
        assert leaf_complexity(steered) <= leaf_complexity(default)

    def test_simplify_measure_logarithms(self):
        """WL: Simplify[2 Log[a] + 4 Log[-4],
        ComplexityFunction -> LeafComplexity] avoids the large Log[16]."""
        expr = 2 * log(a) + 4 * log(-4)
        default = simplify(expr)
        steered = simplify(expr, measure=leaf_complexity)
        assert sympy.simplify(steered - expr) == 0
        assert leaf_complexity(steered) <= leaf_complexity(default)

    def test_select_simplest(self):
        """Select the simplest element of a list according to a custom
        function applied to each leaf, like WL MinimalBy."""
        func = Function("f")
        candidates = [func(75), func(26), func(124), func(103)]

        def digitish(v):
            v = sympy.sympify(v)
            if v.is_Integer:
                return sum(int(d) for d in str(abs(int(v))))
            return 0

        simplest = min(candidates, key=lambda e: leaf_complexity(e, digitish))
        assert simplest == func(103)

    def test_filter_algebraic_closed_forms(self):
        """Functions like AlgebraicRange tend to overproduce complex
        closed forms; LeafComplexity[#] <= 30 selects the simpler ones.

        Note: for a square root of a rational, SymPy rationalizes the
        denominator (sqrt(611/2) -> sqrt(1222)/2) whereas WL keeps
        Sqrt[611/2], so those complexities differ (1293 vs 660 and 37
        vs 22); the measure itself agrees on identical trees, as the
        other values show."""
        closed_forms = [
            13 * sqrt(Rational(611, 2)) / 25,   # 13*sqrt(1222)/50
            89 * sqrt(57) / 50,                 # WL: 204
            3 * sqrt(5) / 2,                    # WL: 18
            sqrt(Rational(13, 2)),              # sqrt(26)/2
            2 * sqrt(7),                        # WL: 16
            sympy.Integer(12),                  # WL: 13
            15 * sqrt(5),                       # WL: 27
        ]
        assert [leaf_complexity(e) for e in closed_forms] == [
            1293, 204, 18, 37, 16, 13, 27]
        simple = [e for e in closed_forms if leaf_complexity(e) <= 30]
        assert simple == [closed_forms[2], closed_forms[4],
                          closed_forms[5], closed_forms[6]]


# ===========================================================================
# Wolfram Language notebook — Properties and Relations
# ===========================================================================

class TestPropertiesAndRelations:
    def test_equals_leaf_count_plus_one_for_unit_leaves(self):
        """WL: if all numeric leaves are 1 or -1, LeafComplexity[expr]
        == LeafCount[expr] + 1 == 14 for (x + 1)/(x - 1) + 1/x."""
        expr = (x + 1) / (x - 1) + 1 / x
        assert leaf_complexity(expr) == 14
        assert leaf_complexity(expr, lambda v: 1) == 14

    def test_greater_than_leaf_count_without_zeros(self):
        """WL: with no 0 in the expression, LeafComplexity[expr] >
        LeafCount[expr] + 1 (22 vs 20)."""
        expr = (2 * x + 1) / (sqrt(x) - 1) + 1 / x
        assert leaf_complexity(expr) == 22
        assert leaf_complexity(expr, lambda v: 1) == 20

    def test_less_than_leaf_count_with_zeros(self):
        """WL: LeafComplexity[{1, 0, 0, 0}] == 3 <
        LeafCount[{1, 0, 0, 0}] + 1 == 6."""
        assert leaf_complexity([1, 0, 0, 0]) == 3
        assert leaf_complexity([1, 0, 0, 0], lambda v: 1) == 6

    def test_infinity_reciprocal_vanishes(self):
        """WL: LeafComplexity[x + 1/Infinity] == 2"""
        assert leaf_complexity(x + 1 / oo) == 2

    def test_infinity_dominates(self):
        """WL: LeafComplexity[x + Infinity] == Infinity"""
        assert leaf_complexity(x + oo) == oo


# ===========================================================================
# Wolfram Language notebook — Neat Examples (adapted)
# ===========================================================================

class TestNeatExamples:
    def test_log2exp2_complexity(self):
        """Doubly nested Log/Exp wrapping shows no overflow.

        Adapted from the notebook to leaves > 1 and Heads -> False,
        because for a leaf equal to 1 the two languages branch: WL
        Log[0] == -Infinity whereas SymPy log(0) == zoo. WL:
        wrappingProperLeafTotal[Range[2, 5], Log[1 - Log[1 - #]]&,
        Exp[#1 - Exp[#2]]&] == -4 E^(-1 - 3 E^(-1 - 2/E^2))."""
        result = leaf_complexity(
            [2, 3, 4, 5],
            lambda v: log(1 - log(1 - v)),
            lambda s, v: exp(s - exp(v)),
            heads=False,
        )
        expected = -4 * exp(-1 - 3 * exp(-1 - 2 * exp(-2)))
        assert sympy.simplify(result - expected) == 0
        assert abs(complex(result.evalf()) - (-0.6340447491499852)) < 1e-12


# ===========================================================================
# Heads -> False variants of the notebook examples
# (expected values computed with the original WL definitions)
# ===========================================================================

class TestProperWolframVerified:
    CASES = [
        (x + 2, 4),
        (x + 10, 12),
        ((x + 2) / (y - 2), 8),
        ((sqrt(x) - 1) / (y**Rational(1, 5) + 2), 16),
        ((x + I) / (x - 2 - 2 * I) - 4 * I / x**2, 16),
        ((2 * x**Rational(1, 3) + I) / (x - 2 - 3 * I) - 5 / x**2, 24),
        (E**x + pi**2, 4 + E + pi),
        ([1, 2, [4, 0, 1, [0, 6, 3, 4, [2, 3, 0], [5, 2, 0], 1], 0]], 35),
        ({"a": [2, 3],
          "b": {(0, 1): {"gamma": 20, "delta": 40}, (-1, -2): 50}}, 116),
        ((x + 1) / (x - 1) + 1 / x, 8),
        ([1, 0, 0, 0], 2),
    ]

    @pytest.mark.parametrize("expr,expected", CASES, ids=str)
    def test_proper_complexity(self, expr, expected):
        assert leaf_complexity(expr, heads=False) == expected

    def test_proper_floats(self):
        """WL: LeafComplexity[1.5 x^2 + 1.1 Log[y], Heads -> False]
        == 7.6"""
        expr = 1.5 * x**2 + 1.1 * log(y)
        assert leaf_complexity(expr, heads=False) == pytest.approx(7.6)

    def test_proper_scaled_square(self):
        """WL: scalingProperLeafTotal[x + 3, #^2&] == 11"""
        assert leaf_complexity(x + 3, lambda v: v**2, heads=False) == 11


# ===========================================================================
# Complex literal edge cases
# (expected values computed with the original WL definitions)
# ===========================================================================

class TestComplexLiteralEdgeCases:
    def test_full_complex_algebraic(self):
        """WL: LeafComplexity[(2 x^(1/3) + I)/(x - 2 - 3 I) - 5/x^2]
        == 36 (the notebook Tests section)."""
        expr = (2 * x**Rational(1, 3) + I) / (x - 2 - 3 * I) - 5 / x**2
        assert leaf_complexity(expr) == 36

    def test_float_imaginary(self):
        """WL: LeafComplexity[2.5 I] == 4.5"""
        assert leaf_complexity(2.5 * I) == 4.5

    def test_rational_imaginary(self):
        """WL: LeafComplexity[I/2] == 6 (the imaginary part is a
        Rational and decomposes further), 4 without heads."""
        assert leaf_complexity(I / 2) == 6
        assert leaf_complexity(I / 2, heads=False) == 4

    def test_negative_imaginary_unit(self):
        """WL: LeafComplexity[-I] == 3"""
        assert leaf_complexity(-I) == 3

    def test_gaussian_integer(self):
        """WL: LeafComplexity[3 - I] == 6"""
        assert leaf_complexity(3 - I) == 6

    def test_mixed_symbolic_factor_not_absorbed(self):
        """WL: LeafComplexity[2 I Pi] == 5 + Pi: only the Gaussian
        literal 2 I is a complex atom; Pi stays a separate factor."""
        assert leaf_complexity(2 * I * pi) == 5 + pi


# ===========================================================================
# Miscellaneous cases verified against the WL definitions
# ===========================================================================

class TestMiscWolframVerified:
    def test_nested_functions(self):
        """WL: LeafComplexity[Sin[Cos[x]]] == 4, proper == 2"""
        expr = sympy.sin(sympy.cos(x))
        assert leaf_complexity(expr) == 4
        assert leaf_complexity(expr, heads=False) == 2

    def test_exp_of_number(self):
        """WL: LeafComplexity[E^2] == 4 + E (SymPy writes E**2 as
        exp(2))."""
        assert leaf_complexity(E**2) == 4 + E

    def test_number_symbols(self):
        """WL: LeafComplexity[GoldenRatio + EulerGamma] ==
        2 + EulerGamma + GoldenRatio"""
        expr = sympy.GoldenRatio + sympy.EulerGamma
        assert (leaf_complexity(expr)
                == 2 + sympy.EulerGamma + sympy.GoldenRatio)

    def test_rational_sum(self):
        """WL: LeafComplexity[x + 3/2] == 9, proper == 7"""
        assert leaf_complexity(x + Rational(3, 2)) == 9
        assert leaf_complexity(x + Rational(3, 2), heads=False) == 7

    def test_negative_powers(self):
        """WL: LeafComplexity[x^-2 - x^-3] == 13"""
        assert leaf_complexity(x**-2 - x**-3) == 13

    def test_generic_function_head(self):
        """WL: LeafComplexity[Fibonacci[10 x]] == 14"""
        assert leaf_complexity(sympy.fibonacci(10 * x)) == 14

    def test_logarithms(self):
        """WL: LeafComplexity[2 Log[a] + 4 Log[-4]] == 21 + Pi.

        SymPy gives 23 + Pi instead: it automatically distributes the
        numeric coefficient, 4*(log(4) + I*pi) -> 4*log(4) + 4*I*pi,
        whereas WL keeps the nested Times[4, Plus[...]] tree. Same
        expression, slightly different canonical tree."""
        assert leaf_complexity(2 * log(a) + 4 * log(-4)) == 23 + pi

    def test_empty_dict(self):
        """WL: LeafComplexity[<||>] == 2, proper == 1"""
        assert leaf_complexity({}) == 2
        assert leaf_complexity({}, heads=False) == 1

    def test_special_values(self):
        """WL: LeafComplexity[Indeterminate] == 2,
        LeafComplexity[ComplexInfinity] == Infinity"""
        assert leaf_complexity(nan) == 2
        assert leaf_complexity(zoo) == oo
        assert leaf_complexity(S.true) == 2

    def test_infinity_scaled(self):
        """In the scaling case, f is applied to the infinity itself."""
        assert leaf_complexity(oo, lambda v: 5) == 6
        assert leaf_complexity(-oo, lambda v: v) == 1 - oo


# ===========================================================================
# Additional coverage — Python objects
# ===========================================================================

class TestPythonObjects:
    def test_python_integers(self):
        assert leaf_complexity(5) == 6
        assert leaf_complexity(-5) == 6

    def test_python_floats(self):
        assert leaf_complexity(2.5) == 3.5

    def test_python_complex(self):
        assert leaf_complexity(3 + 4j) == 9.0
        assert leaf_complexity(3 + 4j, heads=False) == 8.0

    def test_python_fraction(self):
        assert leaf_complexity(Fraction(-3, 2)) == 7
        assert leaf_complexity(Fraction(-3, 2), heads=False) == 6

    def test_strings_and_booleans(self):
        assert leaf_complexity("hello") == 2
        assert leaf_complexity(["a", "b"]) == 4
        assert leaf_complexity([True, False]) == 4

    def test_mixed_containers(self):
        assert leaf_complexity((1, {2, 3}, [4])) == 14

    def test_negative_leaves_absolute_value(self):
        assert leaf_complexity(x - 3) == leaf_complexity(x + 3) == 6

    def test_custom_function_gets_raw_values(self):
        """Like WL, a custom f receives the signed leaf value."""
        assert leaf_complexity(x - 3, lambda v: v) == 0

    def test_dict_keys_are_ignored(self):
        assert leaf_complexity({x**5: 1}) == 3
        assert leaf_complexity({(1, 2, 3): 1}) == 3

    def test_deep_nesting(self):
        """A deeply nested list neither crashes nor miscounts."""
        expr = [1]
        for _ in range(300):
            expr = [expr]
        assert leaf_complexity(expr) == 303  # 301 heads + leaf + 1
        assert leaf_complexity(expr, heads=False) == 2


# ===========================================================================
# Structural properties
# ===========================================================================

class TestStructuralProperties:
    FINITE_EXPRESSIONS = [
        x + 10,
        (x + 2) / (y - 2),
        (sqrt(x) - 1) / (y**Rational(1, 5) + 2),
        (x + I) / (x - 2 - 2 * I) - 4 * I / x**2,
        E**x + pi**2,
        [1, 2, [4, 0, {"k": Rational(3, 2)}]],
        sympy.sin(sympy.cos(x)),
        exp(3 + 2 * I) + 2 * a**2,
        Rational(-7, 3) * x + sqrt(2) * I / 5,
    ]

    @pytest.mark.parametrize("expr", FINITE_EXPRESSIONS, ids=str)
    def test_abs_scaling_equals_identity(self, expr):
        """leaf_complexity(e, Abs) coincides with the default measure,
        as the WL definitions do rule by rule."""
        assert leaf_complexity(expr, Abs) == leaf_complexity(expr)
        assert (leaf_complexity(expr, Abs, heads=False)
                == leaf_complexity(expr, heads=False))

    @pytest.mark.parametrize("expr", FINITE_EXPRESSIONS, ids=str)
    def test_additive_wrapping_equals_scaling(self, expr):
        """g = Plus recovers the scaling case (notebook Details)."""
        f = lambda v: Abs(v) + 1
        assert (leaf_complexity(expr, f, lambda s, v: s + v)
                == leaf_complexity(expr, f))

    @pytest.mark.parametrize("expr", FINITE_EXPRESSIONS, ids=str)
    def test_proper_never_exceeds_improper(self, expr):
        assert (leaf_complexity(expr, heads=False)
                <= leaf_complexity(expr))

    def test_return_types(self):
        assert isinstance(leaf_complexity(x + 1), int)
        assert isinstance(leaf_complexity(1.5 * x), float)
        assert isinstance(leaf_complexity([1, 2, 3]), int)
        assert leaf_complexity(pi) == 1 + pi

    def test_not_callable_raises(self):
        with pytest.raises(TypeError):
            leaf_complexity(x + 1, 3)
        with pytest.raises(TypeError):
            leaf_complexity(x + 1, Abs, 3)

    def test_wrapping_without_f_raises(self):
        with pytest.raises(TypeError):
            leaf_complexity(x + 1, None, Mul)
