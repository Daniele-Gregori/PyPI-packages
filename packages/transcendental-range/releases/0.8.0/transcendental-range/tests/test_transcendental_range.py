"""Test suite for transcendental_range — translated from TranscendentalRangeTests.wlt.

Each test compares by numerical value with tolerance, since symbolic forms
may differ between Wolfram Language and sympy.
"""

import math
import pytest
from sympy import (
    E, pi, exp, log, sin, cos, tan, cot, sec, csc,
    sinh, cosh, tanh, coth, sech, csch,
    asin, acos, atan, acot, asec, acsc,
    asinh, acosh, atanh, acoth, asech, acsch,
    sqrt, Rational, Integer, S, ceiling,
)

from transcendental_range import transcendental_range, NotAlgebraicError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _nv(expr) -> float:
    """Numerical value."""
    return float(expr.evalf(15))


def _assert_list_match(result, expected, tol=1e-10):
    """Assert two lists of sympy expressions match numerically."""
    r_vals = sorted([_nv(e) for e in result])
    e_vals = sorted([_nv(e) for e in expected])
    assert len(r_vals) == len(e_vals), (
        f"Length mismatch: got {len(r_vals)}, expected {len(e_vals)}\n"
        f"  result:   {r_vals}\n"
        f"  expected: {e_vals}"
    )
    for i, (rv, ev) in enumerate(zip(r_vals, e_vals)):
        assert abs(rv - ev) < tol, (
            f"Element {i} mismatch: got {rv}, expected {ev} (diff={abs(rv-ev)})"
        )


# ===========================================================================
# BasicUsage
# ===========================================================================

class TestBasicUsage:
    def test_single_arg_5(self):
        """Basic-SingleArg-5: TranscendentalRange[5] -> {E}"""
        result = transcendental_range(5)
        expected = [E]
        _assert_list_match(result, expected)

    def test_single_arg_10(self):
        """Basic-SingleArg-10: TranscendentalRange[10] -> {E, 2*E, E^2, 3*E}"""
        result = transcendental_range(10)
        expected = [E, 2*E, E**2, 3*E]
        _assert_list_match(result, expected)

    def test_two_args_neg2_to_2(self):
        """Basic-TwoArgs-Neg2To2"""
        result = transcendental_range(-2, 2)
        expected = [-2/E, -exp(-1), -2/E**2, -exp(-2),
                    exp(-2), 2/E**2, exp(-1), 2/E]
        _assert_list_match(result, expected)

    def test_three_args_0_to_3(self):
        """Basic-ThreeArgs-0To3"""
        result = transcendental_range(0, 3, 1)
        expected = [E]
        _assert_list_match(result, expected)

    def test_three_args_0_to_4_half(self):
        """Basic-ThreeArgs-0To4Half"""
        result = transcendental_range(0, 4, Rational(1, 2))
        expected = [
            sqrt(E)/2, E/2, sqrt(E), exp(Rational(3,2))/2,
            3*sqrt(E)/2, E, 2*sqrt(E), E**2/2,
        ]
        _assert_list_match(result, expected)

    def test_four_args(self):
        """Basic-FourArgs"""
        result = transcendental_range(-2, 2, Rational(1, 2), Rational(1, 4))
        expected = [
            -sqrt(E), -E/2, -3/(2*sqrt(E)), -1/sqrt(E),
            -3/(2*exp(Rational(3,2))), -1/(2*E**2),
            1/(2*E), 2/exp(Rational(3,2)), 2/E,
            2/sqrt(E), sqrt(E),
        ]
        _assert_list_match(result, expected)


# ===========================================================================
# NegativeStep
# ===========================================================================

class TestNegativeStep:
    def test_5_to_1(self):
        """NegStep-5To1"""
        result = transcendental_range(5, 1, -1)
        expected = [E]
        _assert_list_match(result, expected)

    def test_10_to_1(self):
        """NegStep-10To1"""
        result = transcendental_range(10, 1, -1)
        # Reversed: {3*E, E^2, 2*E, E}
        expected = [3*E, E**2, 2*E, E]
        r_vals = [_nv(e) for e in result]
        e_vals = [_nv(e) for e in expected]
        assert len(r_vals) == len(e_vals)
        for rv, ev in zip(r_vals, e_vals):
            assert abs(rv - ev) < 1e-10


# ===========================================================================
# MethodExp
# ===========================================================================

class TestMethodExp:
    def test_explicit_5(self):
        """Exp-Explicit-5"""
        result = transcendental_range(5, method='exp')
        expected = [E]
        _assert_list_match(result, expected)

    def test_two_args(self):
        """Exp-TwoArgs"""
        result = transcendental_range(-2, 2, method='exp')
        expected = [-2/E, -exp(-1), -2/E**2, -exp(-2),
                    exp(-2), 2/E**2, exp(-1), 2/E]
        _assert_list_match(result, expected)

    def test_is_default(self):
        """Exp-IsDefault: default method is exp"""
        r1 = transcendental_range(5)
        r2 = transcendental_range(5, method='exp')
        v1 = sorted([_nv(e) for e in r1])
        v2 = sorted([_nv(e) for e in r2])
        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            assert abs(a - b) < 1e-10


# ===========================================================================
# MethodLog
# ===========================================================================

class TestMethodLog:
    def test_single_arg_5(self):
        """Log-SingleArg-5"""
        result = transcendental_range(5, method='log')
        expected = [
            log(3), 2*log(2), log(5), 3*log(2), 2*log(3),
            4*log(2), 2*log(5), 3*log(3), 5*log(2), 3*log(4),
            4*log(3), 3*log(5),
        ]
        _assert_list_match(result, expected)

    def test_neg3_to_3(self):
        """Log-Neg3To3"""
        result = transcendental_range(-3, 3, method='log')
        expected = [
            -2*log(3), -3*log(2), -2*log(2), -log(3), -log(2),
            log(2), log(3), 2*log(2), 3*log(2), 2*log(3),
        ]
        _assert_list_match(result, expected)


# ===========================================================================
# MethodPower
# ===========================================================================

class TestMethodPower:
    def test_default_5(self):
        """Power-Default-5: empty with rationals domain"""
        result = transcendental_range(5, method='power')
        assert result == []

    def test_algebraics(self):
        """Power-Algebraics: 54 elements with algebraics domain"""
        result = transcendental_range(
            -3, 3, method='power', generators_domain='algebraics',
        )
        # Just check count — 54 elements from WL
        assert len(result) == 54


# ===========================================================================
# MethodTrig
# ===========================================================================

class TestMethodTrig:
    def test_sin_neg2_to_2(self):
        """Sin-Neg2To2"""
        result = transcendental_range(-2, 2, method='sin')
        expected = [
            -2*sin(2), -2*sin(1), -sin(2), -sin(1),
            sin(1), sin(2), 2*sin(1), 2*sin(2),
        ]
        _assert_list_match(result, expected)

    def test_cos_neg2_to_2(self):
        """Cos-Neg2To2"""
        result = transcendental_range(-2, 2, method='cos')
        expected = [
            -2*cos(1), 2*cos(2), -cos(1), cos(2),
            -cos(2), cos(1), -2*cos(2), 2*cos(1),
        ]
        _assert_list_match(result, expected)

    def test_tan_neg2_to_2(self):
        """Tan-Neg2To2"""
        result = transcendental_range(-2, 2, method='tan')
        expected = [-tan(1), tan(1)]
        _assert_list_match(result, expected)

    def test_cot_neg2_to_2(self):
        """Cot-Neg2To2"""
        result = transcendental_range(-2, 2, method='cot')
        expected = [
            -2*cot(1), 2*cot(2), -cot(1), cot(2),
            -cot(2), cot(1), -2*cot(2), 2*cot(1),
        ]
        _assert_list_match(result, expected)

    def test_sec_neg2_to_2(self):
        """Sec-Neg2To2"""
        result = transcendental_range(-2, 2, method='sec')
        expected = [-sec(1), sec(1)]
        _assert_list_match(result, expected)

    def test_csc_neg2_to_2(self):
        """Csc-Neg2To2"""
        result = transcendental_range(-2, 2, method='csc')
        expected = [-csc(1), -csc(2), csc(2), csc(1)]
        _assert_list_match(result, expected)


# ===========================================================================
# MethodInvTrig
# ===========================================================================

class TestMethodInvTrig:
    def test_arcsin_neg2_to_2(self):
        """ArcSin-Neg2To2"""
        result = transcendental_range(-2, 2, method='asin')
        expected = [-pi/2, pi/2]
        _assert_list_match(result, expected)

    def test_arccos_neg2_to_2(self):
        """ArcCos-Neg2To2"""
        result = transcendental_range(-2, 2, method='acos')
        expected = [-pi/2, pi/2]
        _assert_list_match(result, expected)

    def test_arctan_neg2_to_2(self):
        """ArcTan-Neg2To2"""
        result = transcendental_range(-2, 2, method='atan')
        expected = [-pi/2, -atan(2), -pi/4, pi/4, atan(2), pi/2]
        _assert_list_match(result, expected)

    def test_arccot_neg2_to_2(self):
        """ArcCot-Neg2To2"""
        result = transcendental_range(-2, 2, method='acot')
        expected = [
            -pi/2, -2*acot(2), -pi/4, -acot(2),
            acot(2), pi/4, 2*acot(2), pi/2,
        ]
        _assert_list_match(result, expected)

    def test_arcsec_neg2_to_2(self):
        """ArcSec-Neg2To2"""
        result = transcendental_range(-2, 2, method='asec')
        expected = [-pi/3, pi/3]
        _assert_list_match(result, expected)

    def test_arccsc_neg2_to_2(self):
        """ArcCsc-Neg2To2"""
        result = transcendental_range(-2, 2, method='acsc')
        expected = [-pi/2, -pi/3, -pi/6, pi/6, pi/3, pi/2]
        _assert_list_match(result, expected)


# ===========================================================================
# MethodHyp
# ===========================================================================

class TestMethodHyp:
    def test_sinh_neg3_to_3(self):
        """Sinh-Neg3To3"""
        result = transcendental_range(-3, 3, method='sinh')
        expected = [-2*sinh(1), -sinh(1), sinh(1), 2*sinh(1)]
        _assert_list_match(result, expected)

    def test_cosh_neg3_to_3(self):
        """Cosh-Neg3To3"""
        result = transcendental_range(-3, 3, method='cosh')
        expected = [-cosh(1), cosh(1)]
        _assert_list_match(result, expected)

    def test_tanh_neg2_to_2(self):
        """Tanh-Neg2To2"""
        result = transcendental_range(-2, 2, method='tanh')
        expected = [
            -2*tanh(2), -2*tanh(1), -tanh(2), -tanh(1),
            tanh(1), tanh(2), 2*tanh(1), 2*tanh(2),
        ]
        _assert_list_match(result, expected)

    def test_coth_neg2_to_2(self):
        """Coth-Neg2To2"""
        result = transcendental_range(-2, 2, method='coth')
        expected = [-coth(1), -coth(2), coth(2), coth(1)]
        _assert_list_match(result, expected)

    def test_sech_neg2_to_2(self):
        """Sech-Neg2To2"""
        result = transcendental_range(-2, 2, method='sech')
        expected = [
            -2*sech(1), -sech(1), -2*sech(2), -sech(2),
            sech(2), 2*sech(2), sech(1), 2*sech(1),
        ]
        _assert_list_match(result, expected)

    def test_csch_neg2_to_2(self):
        """Csch-Neg2To2"""
        result = transcendental_range(-2, 2, method='csch')
        expected = [
            -2*csch(1), -csch(1), -2*csch(2), -csch(2),
            csch(2), 2*csch(2), csch(1), 2*csch(1),
        ]
        _assert_list_match(result, expected)


# ===========================================================================
# MethodInvHyp
# ===========================================================================

class TestMethodInvHyp:
    def test_arcsinh_neg2_to_2(self):
        """ArcSinh-Neg2To2"""
        result = transcendental_range(-2, 2, method='asinh')
        expected = [
            -2*asinh(1), -asinh(2), -asinh(1),
            asinh(1), asinh(2), 2*asinh(1),
        ]
        _assert_list_match(result, expected)

    def test_arccosh_1_to_3(self):
        """ArcCosh-1To3"""
        result = transcendental_range(1, 3, method='acosh')
        expected = [acosh(2), acosh(3), 2*acosh(2)]
        _assert_list_match(result, expected)

    def test_arctanh_neg2_to_2_half(self):
        """ArcTanh-Neg2To2Half"""
        result = transcendental_range(-2, 2, Rational(1, 2), method='atanh')
        expected = [
            -2*atanh(Rational(1,2)), -3*atanh(Rational(1,2))/2,
            -atanh(Rational(1,2)), -atanh(Rational(1,2))/2,
            atanh(Rational(1,2))/2, atanh(Rational(1,2)),
            3*atanh(Rational(1,2))/2, 2*atanh(Rational(1,2)),
        ]
        _assert_list_match(result, expected)

    def test_arccoth_neg3_to_3(self):
        """ArcCoth-Neg3To3"""
        result = transcendental_range(-3, 3, method='acoth')
        expected = [
            -3*acoth(2), -2*acoth(2), -3*acoth(3), -2*acoth(3),
            -acoth(2), -acoth(3), acoth(3), acoth(2),
            2*acoth(3), 3*acoth(3), 2*acoth(2), 3*acoth(2),
        ]
        _assert_list_match(result, expected)

    def test_arcsech_neg2_to_2(self):
        """ArcSech-Neg2To2: empty"""
        result = transcendental_range(-2, 2, method='asech')
        assert result == []

    def test_arccsch_neg2_to_2(self):
        """ArcCsch-Neg2To2"""
        result = transcendental_range(-2, 2, method='acsch')
        expected = [
            -2*acsch(1), -2*acsch(2), -acsch(1), -acsch(2),
            acsch(2), acsch(1), 2*acsch(2), 2*acsch(1),
        ]
        _assert_list_match(result, expected)


# ===========================================================================
# MethodList
# ===========================================================================

class TestMethodList:
    def test_arctan_arcsinh(self):
        """MethodList-ArcTanArcSinh"""
        result = transcendental_range(-2, 2, method=['atan', 'asinh'])
        expected = [
            -2*asinh(1), -pi/2, -asinh(2), -atan(2),
            -asinh(1), -pi/4, pi/4, asinh(1),
            atan(2), asinh(2), pi/2, 2*asinh(1),
        ]
        _assert_list_match(result, expected)

    def test_sin_cos_exp(self):
        """MethodList-SinCosExp"""
        result = transcendental_range(-2, 2, method=['sin', 'cos', 'exp'])
        expected = [
            -2*sin(2), -2*sin(1), -2*cos(1), -sin(2), -sin(1),
            2*cos(2), -2/E, -cos(1), cos(2), -exp(-1), -2/E**2,
            -exp(-2), exp(-2), 2/E**2, exp(-1), -cos(2), cos(1),
            2/E, -2*cos(2), sin(1), sin(2), 2*cos(1),
            2*sin(1), 2*sin(2),
        ]
        _assert_list_match(result, expected)


# ===========================================================================
# MethodAll
# ===========================================================================

class TestMethodAll:
    def test_single_arg_3(self):
        """All-SingleArg-3: 59 elements"""
        result = transcendental_range(3, method='all')
        assert len(result) == 59


# ===========================================================================
# GeneratorsDomain
# ===========================================================================

class TestGeneratorsDomain:
    def test_rationals_5(self):
        """GenDom-Rationals-5"""
        result = transcendental_range(5, generators_domain='rationals')
        expected = [E]
        _assert_list_match(result, expected)

    def test_algebraics_5(self):
        """GenDom-Algebraics-5"""
        result = transcendental_range(5, generators_domain='algebraics')
        expected = [E, sqrt(2)*E, E**sqrt(2), sqrt(3)*E]
        _assert_list_match(result, expected)

    def test_default_is_rationals(self):
        """GenDom-DefaultIsRationals"""
        r1 = transcendental_range(5)
        r2 = transcendental_range(5, generators_domain='rationals')
        v1 = sorted([_nv(e) for e in r1])
        v2 = sorted([_nv(e) for e in r2])
        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            assert abs(a - b) < 1e-10


# ===========================================================================
# FareyRange
# ===========================================================================

class TestFareyRange:
    def test_farey_true_1_to_5(self):
        """Farey-True-1To5"""
        result = transcendental_range(
            1, 5, Rational(1, 2), farey_range=True,
        )
        expected = [E, 3*E/2, exp(Rational(3,2))]
        _assert_list_match(result, expected)

    def test_farey_false_1_to_5(self):
        """Farey-False-1To5"""
        result = transcendental_range(
            1, 5, Rational(1, 2), farey_range=False,
        )
        expected = [E, 3*E/2, exp(Rational(3,2))]
        _assert_list_match(result, expected)

    def test_default_is_false(self):
        """Farey-DefaultIsFalse"""
        r1 = transcendental_range(1, 5, Rational(1, 2))
        r2 = transcendental_range(1, 5, Rational(1, 2), farey_range=False)
        v1 = sorted([_nv(e) for e in r1])
        v2 = sorted([_nv(e) for e in r2])
        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            assert abs(a - b) < 1e-10


# ===========================================================================
# FormulaComplexity
# ===========================================================================

class TestFormulaComplexity:
    def test_complexity_5(self):
        """Complexity-5"""
        result = transcendental_range(
            -3, 3, formula_complexity_threshold=5,
        )
        expected = [
            -E, -3/E, -2/E, -3/E**2, -exp(-1), -2/E**2,
            -3/E**3, -exp(-2), -2/E**3, -exp(-3),
            exp(-3), 2/E**3, exp(-2), 3/E**3,
            2/E**2, exp(-1), 3/E**2, 2/E, 3/E, E,
        ]
        _assert_list_match(result, expected)

    def test_complexity_inf(self):
        """Complexity-Inf"""
        result = transcendental_range(
            -3, 3, formula_complexity_threshold=math.inf,
        )
        expected = [
            -E, -3/E, -2/E, -3/E**2, -exp(-1), -2/E**2,
            -3/E**3, -exp(-2), -2/E**3, -exp(-3),
            exp(-3), 2/E**3, exp(-2), 3/E**3,
            2/E**2, exp(-1), 3/E**2, 2/E, 3/E, E,
        ]
        _assert_list_match(result, expected)

    def test_default_is_inf(self):
        """Complexity-DefaultIsInf"""
        r1 = transcendental_range(-3, 3)
        r2 = transcendental_range(-3, 3, formula_complexity_threshold=math.inf)
        v1 = sorted([_nv(e) for e in r1])
        v2 = sorted([_nv(e) for e in r2])
        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            assert abs(a - b) < 1e-10


# ===========================================================================
# WorkingPrecision
# ===========================================================================

class TestWorkingPrecision:
    def test_default_tanh_20_to_25(self):
        """WP-Default-Tanh20To25: 5 elements at default precision"""
        result = transcendental_range(20, 25, method='tanh')
        expected = [
            21*tanh(20), 22*tanh(20), 23*tanh(20),
            24*tanh(20), 25*tanh(20),
        ]
        _assert_list_match(result, expected)

    def test_wp30_tanh_20_to_25(self):
        """WP-30-Tanh20To25: 30 elements at precision 30"""
        result = transcendental_range(
            20, 25, method='tanh', working_precision=30,
        )
        expected = [
            21*tanh(20), 21*tanh(21), 21*tanh(22), 21*tanh(23),
            21*tanh(24), 21*tanh(25), 22*tanh(20), 22*tanh(21),
            22*tanh(22), 22*tanh(23), 22*tanh(24), 22*tanh(25),
            23*tanh(20), 23*tanh(21), 23*tanh(22), 23*tanh(23),
            23*tanh(24), 23*tanh(25), 24*tanh(20), 24*tanh(21),
            24*tanh(22), 24*tanh(23), 24*tanh(24), 24*tanh(25),
            25*tanh(20), 25*tanh(21), 25*tanh(22), 25*tanh(23),
            25*tanh(24), 25*tanh(25),
        ]
        _assert_list_match(result, expected)


# ===========================================================================
# EdgeCases
# ===========================================================================

class TestEdgeCases:
    def test_single_point(self):
        """Edge-SinglePoint: [1,1] -> empty"""
        result = transcendental_range(1, 1)
        assert result == []

    def test_small_range_1_to_2(self):
        """Edge-SmallRange-1To2: [1,2] -> empty"""
        result = transcendental_range(1, 2)
        assert result == []

    def test_alg_bound_sqrt2(self):
        """Edge-AlgBound-Sqrt2: [0, sqrt(2)] -> empty"""
        result = transcendental_range(0, sqrt(2))
        assert result == []

    def test_rat_step(self):
        """Edge-RatStep: [-1, 1, 1/3] -> 26 elements"""
        result = transcendental_range(-1, 1, Rational(1, 3))
        expected = [
            -2*exp(Rational(1,3))/3, -E/3, -exp(-Rational(1,3)),
            -exp(Rational(2,3))/3, -exp(-Rational(2,3)),
            -2/(3*exp(Rational(1,3))), -exp(Rational(1,3))/3,
            -exp(-1), -2/(3*exp(Rational(2,3))),
            -2/(3*E), -1/(3*exp(Rational(1,3))),
            -1/(3*exp(Rational(2,3))), -1/(3*E),
            1/(3*E), 1/(3*exp(Rational(2,3))),
            1/(3*exp(Rational(1,3))), 2/(3*E),
            2/(3*exp(Rational(2,3))), exp(-1),
            exp(Rational(1,3))/3, 2/(3*exp(Rational(1,3))),
            exp(-Rational(2,3)), exp(Rational(2,3))/3,
            exp(-Rational(1,3)), E/3,
            2*exp(Rational(1,3))/3,
        ]
        _assert_list_match(result, expected)

    def test_ceiling_workaround(self):
        """Edge-CeilingWorkaround: [0, Ceiling[E], 1/3] -> 17 elements"""
        result = transcendental_range(0, ceiling(E), Rational(1, 3))
        expected = [
            exp(Rational(1,3))/3, exp(Rational(2,3))/3, E/3,
            2*exp(Rational(1,3))/3, exp(Rational(4,3))/3,
            2*exp(Rational(2,3))/3, exp(Rational(1,3)),
            exp(Rational(5,3))/3, 2*E/3,
            4*exp(Rational(1,3))/3, exp(Rational(2,3)),
            5*exp(Rational(1,3))/3, E**2/3,
            2*exp(Rational(4,3))/3, 4*exp(Rational(2,3))/3,
            E, 2*exp(Rational(1,3)),
        ]
        _assert_list_match(result, expected)

    def test_transc_bound_fails(self):
        """Edge-TranscBound-Fails: TranscendentalRange[0, E, 1/3] -> error"""
        with pytest.raises(NotAlgebraicError):
            transcendental_range(0, E, Rational(1, 3))

    def test_zero_step_no_hang(self):
        """Edge-ZeroStep-NoHang: step=0 should not hang"""
        result = transcendental_range(1, 5, 0)
        assert isinstance(result, list)


# ===========================================================================
# CombinedOptions
# ===========================================================================

class TestCombinedOptions:
    def test_sin_algebraics(self):
        """Combined-SinAlg"""
        result = transcendental_range(
            -1, 1, method='sin', generators_domain='algebraics',
        )
        expected = [-sin(1), sin(1)]
        _assert_list_match(result, expected)

    def test_exp_high_precision(self):
        """Combined-ExpHP"""
        result = transcendental_range(
            -3, 3, method='exp', working_precision=40,
        )
        expected = [
            -E, -3/E, -2/E, -3/E**2, -exp(-1), -2/E**2,
            -3/E**3, -exp(-2), -2/E**3, -exp(-3),
            exp(-3), 2/E**3, exp(-2), 3/E**3,
            2/E**2, exp(-1), 3/E**2, 2/E, 3/E, E,
        ]
        _assert_list_match(result, expected)


# ===========================================================================
# PropertiesRelations
# ===========================================================================

class TestPropertiesRelations:
    def test_equivalent_to_outer(self):
        """Props-EquivalentToOuter: equivalence with direct Outer construction"""
        from sympy import Rational as R
        # Manual outer: {b * exp(a) for b in Range[-2,2,1/2], a in Range[-2,2,1/2]}
        # filtered to [-2, 2] and non-algebraic
        rng = []
        cur = R(-2)
        while cur <= R(2):
            rng.append(cur)
            cur += R(1, 2)

        naive = []
        for b in rng:
            for a in rng:
                val = b * exp(a)
                vn = float(val.evalf(15))
                if -2 <= vn <= 2 and not _is_algebraic_check(val):
                    naive.append(val)

        # Dedup by numerical value
        seen = {}
        for v in naive:
            key = round(float(v.evalf(15)), 12)
            if key not in seen:
                seen[key] = v
        naive_deduped = sorted(seen.values(), key=lambda e: float(e.evalf(15)))

        result = transcendental_range(-2, 2, R(1, 2))
        _assert_list_match(result, naive_deduped)


def _is_algebraic_check(expr) -> bool:
    """Check algebraic-ness for the outer equivalence test."""
    from sympy import ask, Q
    r = ask(Q.algebraic(expr))
    if r is True:
        return True
    if expr.is_rational:
        return True
    if expr.is_zero:
        return True
    return False
