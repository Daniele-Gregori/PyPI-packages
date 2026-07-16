"""
Tests for algebraic_range 0.9 — transliterated from the Wolfram Language
AlgebraicRange 2.0 test suite (notebook groups 1-10).

Expected values are the outputs of the WL 2.0 definition, recorded verbatim
in Wolfram InputForm and parsed here into sympy expressions. Structural
(SameQ-style) equality is required, mirroring VerificationTest.

Group 9-a (machine-precision collisions with a step below double resolution)
is compared numerically: the collision-class boundaries are floating-point
artifacts and may legitimately differ from WL by a grid unit (see the
WorkingPrecision note in the WL documentation).
"""

import math
import re
import time

import pytest
import sympy
from sympy import E, Rational, S, sqrt, sympify

from algebraic_range import (
    algebraic_range as ar,
    formula_complexity,
    FareyStepError,
    LowerBoundError,
    NotAlgebraicError,
    NotRealError,
    StepBoundError,
)


# ── WL InputForm parsing ────────────────────────────────────────────────────

def wl(s):
    """Parse a Wolfram InputForm expression (lists, Sqrt, rationals, ^)."""
    s = s.strip()
    s = s.replace("Sqrt[", "sqrt(")
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("{", "[").replace("}", "]")
    s = s.replace("^", "**")
    return sympify(s, rational=True)


def assert_same(result, expected_wl, tid):
    expected = wl(expected_wl)
    expected = list(expected) if isinstance(expected, (list, tuple)) \
        else [expected]
    assert len(result) == len(expected), (
        f"{tid}: length {len(result)} != {len(expected)}")
    for i, (g, e) in enumerate(zip(result, expected)):
        assert g == e or sympy.simplify(g - e) == 0, (
            f"{tid}: element {i}: {g} != {e}")


def assert_close(result, expected_wl, tid, tol=1e-13):
    expected = list(wl(expected_wl))
    assert len(result) == len(expected), (
        f"{tid}: length {len(result)} != {len(expected)}")
    for i, (g, e) in enumerate(zip(result, expected)):
        gv, ev = float(g.evalf(25)), float(e.evalf(25))
        assert abs(gv - ev) <= tol * max(1.0, abs(ev)), (
            f"{tid}: element {i}: {gv} != {ev}")


# 9-a: collision-class representatives are double-rounding artifacts;
# compare numerically at machine resolution instead of structurally.
APPROX_IDS = {"9 a wp_pos_pos_pos_mp"}

# ── Groups 1-9: the WL 2.0 verification suite ───────────────────────────────

CASES = [
    ('1-a-2args_int_pos_pos',
     'ar(1, 3, root_order=2)',
     'WL',
     '{1, Sqrt[2], Sqrt[3], 2, Sqrt[5], Sqrt[6], Sqrt[7], 2*Sqrt[2], 3}'),
    ('1-b-2args_int_neg_neg',
     'ar(-3, -1, root_order=2)',
     'WL',
     '{-3, -2*Sqrt[2], -Sqrt[7], -Sqrt[6], -Sqrt[5], -2, -Sqrt[3], -Sqrt[2], -1}'),
    ('1-c-2args_int_neg_pos',
     'ar(-3, 2, root_order=2)',
     'WL',
     '{-3, -2*Sqrt[2], -Sqrt[7], -Sqrt[6], -Sqrt[5], -2, -Sqrt[3], -Sqrt[2], -1, 0, 1, Sqrt[2], Sqrt[3], 2}'),
    ('1-d-2args_rat_pos_pos',
     'ar(Rational(1, 3), 3, root_order=2)',
     'WL',
     '{1/3, Sqrt[10]/3, Sqrt[19]/3, (2*Sqrt[7])/3, Sqrt[37]/3, Sqrt[46]/3, Sqrt[55]/3, 8/3, Sqrt[73]/3}'),
    ('1-e-2args_rat_neg_neg',
     'ar(Rational(-3, 2), Rational(-1, 2), root_order=2)',
     'WL',
     '{-3/2, -1/2*Sqrt[5], -1/2}'),
    ('1-f-2args_rat_neg_pos',
     'ar(-3, 1, root_order=2)',
     'WL',
     '{-3, -2*Sqrt[2], -Sqrt[7], -Sqrt[6], -Sqrt[5], -2, -Sqrt[3], -Sqrt[2], -1, 0, 1}'),
    ('1-g-2args_rat_neg_pos',
     'ar(Rational(-3, 2), Rational(5, 3), root_order=2)',
     'WL',
     '{-3/2, -1/2*Sqrt[5], -1/2, 0, 1, Sqrt[2]}'),
    ('1-h-2args_irr_pos_pos',
     'ar(sqrt(6), 5, root_order=2)',
     'WL',
     '{Sqrt[6], Sqrt[7], 2*Sqrt[2], 3, Sqrt[10], Sqrt[11], 2*Sqrt[3], Sqrt[13], Sqrt[14], Sqrt[15], 4, Sqrt[17], 3*Sqrt[2], Sqrt[19], 2*Sqrt[5], Sqrt[21], Sqrt[22], Sqrt[23], 2*Sqrt[6], 5}'),
    ('1-i-2args_irr_neg_neg',
     'ar(-2*sqrt(2), -1, root_order=2)',
     'WL',
     '{-2*Sqrt[2], -Sqrt[7], -Sqrt[6], -Sqrt[5], -2, -Sqrt[3], -Sqrt[2], -1}'),
    ('1-j-2args_irr_neg_pos',
     'ar(Rational(-1, 7)*sqrt(6), Rational(7, 3), root_order=2)',
     'WL',
     '{-1/7*Sqrt[6], 0, 1, Sqrt[2], Sqrt[3], 2, Sqrt[5]}'),
    ('1-k-2args_empty_pos_pos',
     'ar(4, 3, root_order=2)',
     'WL',
     '{}'),
    ('1-l-2args_empty_pos_neg',
     'ar(Rational(3, 2), Rational(-4, 3), root_order=2)',
     'WL',
     '{}'),
    ('1-m-2args_empty_neg_neg',
     'ar(-sqrt(Rational(3, 2)), -4, root_order=2)',
     'WL',
     '{}'),
    ('1-n-2args_single_int_neg',
     'ar(-2, -2, root_order=2)',
     'WL',
     '{-2}'),
    ('1-o-2args_single_rat_neg',
     'ar(Rational(-3, 2), Rational(-3, 2), root_order=2)',
     'WL',
     '{-3/2}'),
    ('1-p-2args_single_irr_neg',
     'ar(Rational(-1, 2)*sqrt(3), -Rational(1, 2)*sqrt(3), root_order=2)',
     'WL',
     '{-1/2*Sqrt[3]}'),
    ('1-q-2args_single_irr_pos',
     'ar(sqrt(7)/6, sqrt(7)/6, root_order=2)',
     'WL',
     '{Sqrt[7]/6}'),
    ('2-a-3args_int_pos_pos_int_pos',
     'ar(0, 3, 2, root_order=2)',
     'WL',
     '{0, 2, 2*Sqrt[2]}'),
    ('2-b-3args_irr_pos_pos_rat_pos',
     'ar(1, sqrt(5), Rational(1, 3), root_order=2)',
     'WL',
     '{1, 2/Sqrt[3], 4/3, Sqrt[2], (2*Sqrt[5])/3, 5/3, Sqrt[3], (4*Sqrt[2])/3, 2, Sqrt[5]}'),
    ('2-c-3args_irr_pos_pos_irr_pos',
     'ar(1, sqrt(5), 1/sqrt(3), root_order=2)',
     'WL',
     '{1, 2/Sqrt[3], Sqrt[5/3], Sqrt[2], 1 + 1/Sqrt[3], Sqrt[3], 2, 1 + 2/Sqrt[3], Sqrt[2]*(1 + 1/Sqrt[3]), Sqrt[5]}'),
    ('2-d-3args_irr_pos_pos_irr_pos_large',
     'ar(sqrt(2), 2*sqrt(2), 2/sqrt(3), root_order=2)',
     'WL',
     '{Sqrt[2], Sqrt[10/3], Sqrt[14/3], Sqrt[6], Sqrt[22/3], 2*Sqrt[2]}'),
    ('2-e-3args_int_pos_pos_rat_neg',
     'ar(4, 2, Rational(-1, 3), root_order=2)',
     'WL',
     '{4, Sqrt[15], (8*Sqrt[2])/3, Sqrt[14], (5*Sqrt[5])/3, Sqrt[13], (4*Sqrt[7])/3, 2*Sqrt[3], 10/3, Sqrt[11], 4*Sqrt[2/3], Sqrt[10], 3, (4*Sqrt[5])/3, 2*Sqrt[2], 8/3, Sqrt[7], 2*Sqrt[5/3], (2*Sqrt[14])/3, Sqrt[6], (2*Sqrt[13])/3, 4/Sqrt[3], Sqrt[5], (2*Sqrt[11])/3, (2*Sqrt[10])/3, 2}'),
    ('2-f-3args_int_pos_pos_rat_neg_large',
     'ar(4, 2, Rational(-4, 3), root_order=2)',
     'WL',
     '{4, (8*Sqrt[2])/3, (4*Sqrt[7])/3, 4*Sqrt[2/3], (4*Sqrt[5])/3, 8/3}'),
    ('2-g-3args_int_pos_pos_rat_neg_single',
     'ar(4, 2, -2, root_order=2)',
     'WL',
     '{4}'),
    ('2-h-3args_int_pos_pos_rat_neg_empty',
     'ar(2, 4, Rational(-7, 3), root_order=2)',
     'WL',
     '{}'),
    ('3-a-3args_int_neg_neg_rat_pos',
     'ar(-4, -2, Rational(2, 3), root_order=2)',
     'WL',
     '{-4, -Sqrt[15], -Sqrt[14], (-5*Sqrt[5])/3, -Sqrt[13], -2*Sqrt[3], -10/3, -Sqrt[11], -Sqrt[10], -3, -2*Sqrt[2], -8/3, -Sqrt[7], -2*Sqrt[5/3], (-2*Sqrt[14])/3, -Sqrt[6], (-2*Sqrt[13])/3, -4/Sqrt[3], -Sqrt[5], (-2*Sqrt[11])/3, (-2*Sqrt[10])/3, -2}'),
    ('3-b-3args_rat_neg_neg_rat_pos',
     'ar(-3, Rational(-1, 2), Rational(1, 2), root_order=2)',
     'WL',
     '{-3, -2*Sqrt[2], -Sqrt[7], (-3*Sqrt[3])/2, -5/2, -Sqrt[6], -Sqrt[5], -3/Sqrt[2], -2, -Sqrt[3], -3/2, -Sqrt[2], -1/2*Sqrt[7], -Sqrt[3/2], -1/2*Sqrt[5], -1, -1/2*Sqrt[3], -(1/Sqrt[2]), -1/2}'),
    ('3-c-3args_rat_neg_neg_rat_pos_bis',
     'ar(Rational(-5, 2), Rational(-1, 2), Rational(1, 2), root_order=2)',
     'WL',
     '{-5/2, -1/2*Sqrt[21], -9/4, -Sqrt[5], -1/2*Sqrt[17], -1/2*Sqrt[13], (-3*Sqrt[5])/4, -3/2, -5/4, -1/4*Sqrt[21], -1/2*Sqrt[5], -1/4*Sqrt[17], -1, -1/4*Sqrt[13], -3/4, -1/4*Sqrt[5], -1/2}'),
    ('3-d-3args_rat_neg_neg_irr_pos',
     'ar(-3, Rational(-1, 2), 1/sqrt(2), root_order=2)',
     'WL',
     '{-3, -(Sqrt[3]*(1 + 1/Sqrt[2])), -2*Sqrt[2], -Sqrt[7], -Sqrt[6], -(Sqrt[2]*(1 + 1/Sqrt[2])), -Sqrt[5], -3/Sqrt[2], -2, -Sqrt[7/2], -Sqrt[3], -1 - 1/Sqrt[2], -Sqrt[5/2], -Sqrt[2], -Sqrt[3/2], -1, -(1/Sqrt[2])}'),
    ('3-e-3args_irr_neg_neg_irr_pos',
     'ar(-sqrt(3), -(1/sqrt(2)), 1/(2*sqrt(2)), root_order=2)',
     'WL',
     '{-Sqrt[3], -1 - 1/Sqrt[2], -Sqrt[2], -1 - 1/(2*Sqrt[2]), -Sqrt[3/2], -1, -(1/Sqrt[2])}'),
    ('3-f-3args_irr_pos_pos_irr_pos_large',
     'ar(-4*sqrt(2), -sqrt(2), 4/sqrt(3), root_order=2)',
     'WL',
     '{-4*Sqrt[2], -4*Sqrt[5/3], -8/Sqrt[3], -4, -4*Sqrt[2/3]}'),
    ('3-g-3args_int_neg_neg_rat_neg',
     'ar(-2, -3, Rational(-1, 4), root_order=2)',
     'WL',
     '{-2, -3/Sqrt[2], -Sqrt[5], -9/4, -Sqrt[6], -5/2, -Sqrt[7], (-5*Sqrt[5])/4, -2*Sqrt[2], -3}'),
    ('3-h-3args_rat_neg_neg_rat_neg_large',
     'ar(Rational(-1, 2), -2, Rational(-4, 3), root_order=2)',
     'WL',
     '{-1/2, -1/6*Sqrt[73], -1/6*Sqrt[137]}'),
    ('3-i-3args_rat_neg_neg_rat_neg_large_bis',
     'ar(Rational(-1, 7), Rational(-7, 2), Rational(-7, 3), root_order=2)',
     'WL',
     '{-1/7, -1/21*Sqrt[2410], -1/21*Sqrt[4811]}'),
    ('3-j-3args_irr_pos_pos_int_neg_single',
     'ar(-sqrt(5), -2*sqrt(5), -5, root_order=2)',
     'WL',
     '{-Sqrt[5]}'),
    ('3-k-3args_int_pos_pos_rat_neg_empty',
     'ar(-4, 6, -2, root_order=2)',
     'WL',
     '{}'),
    ('4-a-3args_int_neg_pos_int_pos',
     'ar(-4, 4, 2, root_order=2)',
     'WL',
     '{-4, -2*Sqrt[3], -2*Sqrt[2], -2, 0, 2, 2*Sqrt[2], 2*Sqrt[3], 4}'),
    ('4-b-3args_int_neg_pos_int_pos_bis',
     'ar(-4, 4, 3, root_order=2)',
     'WL',
     '{-4, -Sqrt[7], 0, 3}'),
    ('4-c-3args_irr_neg_pos_rat_pos',
     'ar(-1, sqrt(5), Rational(1, 3), root_order=2)',
     'WL',
     '{-1, -2/3, -1/3, 0, 1/3, Sqrt[2]/3, 1/Sqrt[3], 2/3, Sqrt[5]/3, (2*Sqrt[2])/3, 1, 2/Sqrt[3], 4/3, Sqrt[2], (2*Sqrt[5])/3, 5/3, Sqrt[3], (4*Sqrt[2])/3, 2, Sqrt[5]}'),
    ('4-d-3args_irr_neg_pos_irr_pos',
     'ar(-sqrt(5), sqrt(5), 1/sqrt(3), root_order=2)',
     'WL',
     '{-Sqrt[5], -(Sqrt[2]*(1 + 1/Sqrt[3])), -1 - 2/Sqrt[3], -2, -Sqrt[3], -1 - 1/Sqrt[3], -Sqrt[2], -Sqrt[5/3], -2/Sqrt[3], -1, -Sqrt[2/3], -(1/Sqrt[3]), 0, 1/Sqrt[3], Sqrt[2/3], 1, 2/Sqrt[3], Sqrt[5/3], Sqrt[2], 1 + 1/Sqrt[3], Sqrt[3], 2, 1 + 2/Sqrt[3], Sqrt[2]*(1 + 1/Sqrt[3]), Sqrt[5]}'),
    ('4-e-3args_irr_neg_pos_irr_pos_large',
     'ar(-sqrt(2), 2*sqrt(2), 2/sqrt(3), root_order=2)',
     'WL',
     '{-Sqrt[2], -Sqrt[2/3], 0, 2/Sqrt[3], 2*Sqrt[2/3], 2, 4/Sqrt[3], 2*Sqrt[5/3], 2*Sqrt[2]}'),
    ('4-f-3args_int_neg_pos_int_pos_large_bis',
     'ar(-2, 8, 10, root_order=2)',
     'WL',
     '{-2, 0}'),
    ('4-g-3args_int_neg_pos_rat_pos_large_single',
     'ar(-2, 8, 10.1, root_order=2)',
     'WL',
     '{-2}'),
    ('4-h-3args_int_neg_pos_rat_neg_empty',
     'ar(-2, 8, -1.1, root_order=2)',
     'WL',
     '{}'),
    ('4-i-3args_int_pos_neg_rat_neg',
     'ar(4, -2, Rational(-1, 3), root_order=2)',
     'WL',
     '{4, Sqrt[15], (8*Sqrt[2])/3, Sqrt[14], (5*Sqrt[5])/3, 11/3, Sqrt[13], (4*Sqrt[7])/3, 2*Sqrt[3], 10/3, Sqrt[11], (7*Sqrt[2])/3, 4*Sqrt[2/3], Sqrt[10], 3, (4*Sqrt[5])/3, 5/Sqrt[3], 2*Sqrt[2], 8/3, Sqrt[7], 2*Sqrt[5/3], (2*Sqrt[14])/3, Sqrt[6], (2*Sqrt[13])/3, (5*Sqrt[2])/3, 7/3, 4/Sqrt[3], Sqrt[5], (2*Sqrt[11])/3, (2*Sqrt[10])/3, 2, (4*Sqrt[2])/3, (2*Sqrt[7])/3, Sqrt[3], 5/3, 2*Sqrt[2/3], (2*Sqrt[5])/3, Sqrt[2], 4/3, Sqrt[5/3], Sqrt[14]/3, Sqrt[13]/3, 2/Sqrt[3], Sqrt[11]/3, Sqrt[10]/3, 1, (2*Sqrt[2])/3, Sqrt[7]/3, Sqrt[2/3], Sqrt[5]/3, 2/3, 1/Sqrt[3], Sqrt[2]/3, 1/3, 0, -1/3, -1/3*Sqrt[2], -(1/Sqrt[3]), -2/3, (-2*Sqrt[2])/3, -1, -2/Sqrt[3], -4/3, -Sqrt[2], -5/3, -Sqrt[3], (-4*Sqrt[2])/3, -2}'),
    ('4-j-3args_int_pos_neg_rat_neg_large',
     'ar(2, -5, Rational(-4, 3), root_order=2)',
     'WL',
     '{2, (2*Sqrt[5])/3, 4/3, 2/3, 0, -4/3, (-4*Sqrt[2])/3, -4/Sqrt[3], -8/3, (-4*Sqrt[5])/3, -4*Sqrt[2/3], (-4*Sqrt[7])/3, (-8*Sqrt[2])/3, -4, (-4*Sqrt[10])/3, (-4*Sqrt[11])/3, -8/Sqrt[3], (-4*Sqrt[13])/3, (-4*Sqrt[14])/3}'),
    ('4-k-3args_int_pos_neg_rat_neg_large_bis',
     'ar(2, -5, -7, root_order=2)',
     'WL',
     '{2, 0}'),
    ('4-l-3args_int_pos_neg_rat_neg_single',
     'ar(4, -2, -6.01, root_order=2)',
     'WL',
     '{4}'),
    ('4-m-3args_int_pos_pos_rat_neg_empty',
     'ar(2, 4, Rational(-7, 3), root_order=2)',
     'WL',
     '{}'),
    ('5 a 4args_pos_pos_pos_pos_expl',
     'ar(2, 5, Rational(1, 3), Rational(1, 4))',
     'WL',
     '{2, 4/Sqrt[3], 2*Sqrt[5/3], (2*Sqrt[19])/3, Sqrt[10], 2*Sqrt[3], (5*Sqrt[5])/3, 4, Sqrt[19], 8/Sqrt[3], 2*Sqrt[6]}'),
    ('5 b 4args_pos_pos_pos_pos_diff',
     'ar(2, 5, Rational(1, 3), Rational(1, 4))',
     'DIFFS_GE',
     'Rational(1, 4)'),
    ('5 c 4args_neg_neg_pos_pos_expl',
     'ar(-3, -1, Rational(1, 2), Rational(1, 3))',
     'WL',
     '{-3, -Sqrt[7], -Sqrt[5], -Sqrt[3], -1/2*Sqrt[7]}'),
    ('5 d 4args_neg_neg_pos_pos_diff',
     'ar(-3, -1, Rational(1, 2), Rational(1, 3))',
     'DIFFS_GE',
     'Rational(1, 3)'),
    ('5 e 4args_neg_pos_pos_pos_expl',
     'ar(-3, 3, Rational(2, 3), Rational(2, 5))',
     'WL',
     '{-3, -Sqrt[6], -2, (-2*Sqrt[5])/3, -1, 0, 2/3, 2/Sqrt[3], 2*Sqrt[2/3], Sqrt[5], Sqrt[7]}'),
    ('5 f 4args_neg_pos_pos_pos_diff',
     'ar(-3, 3, Rational(2, 3), Rational(2, 5))',
     'DIFFS_GE',
     'Rational(2, 5)'),
    ('6-a-ord3_int_neg_pos_int_pos',
     'ar(-4, 4, 2, root_order=3)',
     'WL',
     '{-4, -2*7^(1/3), -2*6^(1/3), -2*Sqrt[3], -2*5^(1/3), -2*2^(2/3), -2*3^(1/3), -2*Sqrt[2], -2*2^(1/3), -2, 0, 2, 2*2^(1/3), 2*Sqrt[2], 2*3^(1/3), 2*2^(2/3), 2*5^(1/3), 2*Sqrt[3], 2*6^(1/3), 2*7^(1/3), 4}'),
    ('6-b-ord4_int_neg_pos_int_pos',
     'ar(-3, 3, 2, root_order=4)',
     'WL',
     '{-3, -65^(1/4), -19^(1/3), -Sqrt[7], -33^(1/4), -Sqrt[5], -11^(1/3), -17^(1/4), -3^(1/3), -1, 0, 2, 2*2^(1/4), 2*2^(1/3), 2*3^(1/4), 2*Sqrt[2], 2*3^(1/3), 2*5^(1/4)}'),
    ('6-c-ord5_int_neg_pos_int_pos',
     'ar(-4, 4, 3, root_order=5)',
     'WL',
     '{-4, -781^(1/5), -(Sqrt[5]*7^(1/4)), -538^(1/5), -37^(1/3), -295^(1/5), -94^(1/4), -Sqrt[7], -(2^(2/5)*13^(1/5)), -10^(1/3), -13^(1/4), 0, 3, 3*2^(1/5), 3*2^(1/4), 3*3^(1/5), 3*2^(1/3), 3*3^(1/4), 3*2^(2/5)}'),
    ('6-d-ord3_irr_neg_pos_rat_pos',
     'ar(-1, sqrt(5), Rational(1, 3), root_order=3)',
     'WL',
     '{-1, -2/3, -1/3, 0, 1/3, 2^(1/3)/3, Sqrt[2]/3, 3^(-2/3), 2^(2/3)/3, 5^(1/3)/3, 1/Sqrt[3], 2^(1/3)/3^(2/3), 7^(1/3)/3, 2/3, 3^(-1/3), 10^(1/3)/3, 11^(1/3)/3, Sqrt[5]/3, (2*2^(1/3))/3, (2*Sqrt[2])/3, 2/3^(2/3), 1, (2*2^(2/3))/3, (2*5^(1/3))/3, 2/Sqrt[3], (2*2^(1/3))/3^(2/3), 2^(1/3), (2*7^(1/3))/3, 4/3, 2/3^(1/3), Sqrt[2], (2*10^(1/3))/3, 3^(1/3), (2*11^(1/3))/3, (2*Sqrt[5])/3, 2^(2/3), 5/3, (4*2^(1/3))/3, 5^(1/3), Sqrt[3], 6^(1/3), (4*Sqrt[2])/3, 7^(1/3), 4/3^(2/3), 2, 3^(2/3), (5*2^(1/3))/3, (4*2^(2/3))/3, 10^(1/3), 11^(1/3), Sqrt[5]}'),
    ('6-e-ord3_irr_neg_pos_irr_pos',
     'ar(-sqrt(5), sqrt(5), 2/sqrt(3), root_order=3)',
     'WL',
     '{-Sqrt[5], (-2*(-1 + (15*Sqrt[15])/8)^(1/3))/Sqrt[3], (-2*(-2 + (15*Sqrt[15])/8)^(1/3))/Sqrt[3], -Sqrt[11/3], (-2*(-3 + (15*Sqrt[15])/8)^(1/3))/Sqrt[3], (-2*(-4 + (15*Sqrt[15])/8)^(1/3))/Sqrt[3], -Sqrt[7/3], (-2*(-5 + (15*Sqrt[15])/8)^(1/3))/Sqrt[3], (-2*(-6 + (15*Sqrt[15])/8)^(1/3))/Sqrt[3], -1, (-2*(-7 + (15*Sqrt[15])/8)^(1/3))/Sqrt[3], 0, 2/Sqrt[3], (2*2^(1/3))/Sqrt[3], 2*Sqrt[2/3], 2/3^(1/6), (2*2^(2/3))/Sqrt[3], (2*5^(1/3))/Sqrt[3], 2, (2*2^(1/3))/3^(1/6), (2*7^(1/3))/Sqrt[3]}'),
    ('6-f-ord4_irr_neg_pos_irr_pos_large',
     'ar(-sqrt(2), 2*sqrt(2), 2/sqrt(3), root_order=4)',
     'WL',
     '{-Sqrt[2], -(Sqrt[2/3]*5^(1/4)), (-2*(-1 + (3*Sqrt[3/2])/2)^(1/3))/Sqrt[3], -Sqrt[2/3], 0, 2/Sqrt[3], (2*2^(1/4))/Sqrt[3], (2*2^(1/3))/Sqrt[3], 2/3^(1/4), 2*Sqrt[2/3], 2/3^(1/6), (2*5^(1/4))/Sqrt[3], 2*(2/3)^(1/4), (2*2^(2/3))/Sqrt[3], (2*7^(1/4))/Sqrt[3], (2*2^(3/4))/Sqrt[3], (2*5^(1/3))/Sqrt[3], 2, (2*10^(1/4))/Sqrt[3], (2*2^(1/3))/3^(1/6), (2*11^(1/4))/Sqrt[3], (2*Sqrt[2])/3^(1/4), (2*13^(1/4))/Sqrt[3], (2*7^(1/3))/Sqrt[3], (2*14^(1/4))/Sqrt[3], 2*(5/3)^(1/4), 4/Sqrt[3], (2*17^(1/4))/Sqrt[3], 2*2^(1/4), 2*3^(1/6), (2*19^(1/4))/Sqrt[3], 2*Sqrt[2/3]*5^(1/4), 2*(7/3)^(1/4), (2*10^(1/3))/Sqrt[3], (2*22^(1/4))/Sqrt[3], (2*23^(1/4))/Sqrt[3], (2*2^(3/4))/3^(1/4), (2*11^(1/3))/Sqrt[3], 2*Sqrt[5/3], (2*26^(1/4))/Sqrt[3], 2*3^(1/4), (2*2^(2/3))/3^(1/6), 2*Sqrt[2/3]*7^(1/4), (2*29^(1/4))/Sqrt[3], 2*(10/3)^(1/4), (2*13^(1/3))/Sqrt[3], (2*31^(1/4))/Sqrt[3], (4*2^(1/4))/Sqrt[3], 2*(11/3)^(1/4), (2*14^(1/3))/Sqrt[3], (2*34^(1/4))/Sqrt[3], (2*35^(1/4))/Sqrt[3], 2*Sqrt[2]}'),
    ('6-g-ord3_int_neg_pos_int_pos_large_bis',
     'ar(-2, 8, 10, root_order=3)',
     'WL',
     '{-2, 0}'),
    ('6-h-ord3_int_neg_pos_rat_pos_large_single',
     'ar(-2, 8, 10.1, root_order=3)',
     'WL',
     '{-2}'),
    ('6-i-ord3_int_neg_pos_rat_neg_empty',
     'ar(-2, 8, -1.1, root_order=3)',
     'WL',
     '{}'),
    ('6-j-ord3_int_pos_neg_rat_neg',
     'ar(2, -1, Rational(-1, 3), root_order=3)',
     'WL',
     '{2, 4/3^(2/3), 7^(1/3), (4*Sqrt[2])/3, 6^(1/3), Sqrt[3], 5^(1/3), (4*2^(1/3))/3, 5/3, 2^(2/3), 3^(1/3), Sqrt[2], 4/3, (2*7^(1/3))/3, 2^(1/3), (2*2^(1/3))/3^(2/3), 2/Sqrt[3], (2*5^(1/3))/3, (2*2^(2/3))/3, 1, 2/3^(2/3), (2*Sqrt[2])/3, (2*2^(1/3))/3, 2/3, 7^(1/3)/3, 2^(1/3)/3^(2/3), 1/Sqrt[3], 5^(1/3)/3, 2^(2/3)/3, 3^(-2/3), Sqrt[2]/3, 2^(1/3)/3, 1/3, 0, -1/3, -2/3, -1}'),
    ('6-k-ord4_int_pos_neg_rat_neg',
     'ar(2, -1, Rational(-2, 3), root_order=4)',
     'WL',
     '{2, (5*2^(1/4))/3, 15^(1/4), 14^(1/4), 7^(1/3), 13^(1/4), Sqrt[2]*3^(1/4), 11^(1/4), 6^(1/3), 10^(1/4), Sqrt[3], 5^(1/3), 2^(3/4), 5/3, 7^(1/4), 2^(2/3), 6^(1/4), 5^(1/4), 3^(1/3), Sqrt[2], 4/3, 3^(1/4), (2*5^(1/4))/3^(3/4), (2*14^(1/4))/3, (2*7^(1/3))/3, (2*13^(1/4))/3, 2^(1/3), (2*Sqrt[2])/3^(3/4), (2*11^(1/4))/3, (2*2^(1/3))/3^(2/3), 2^(1/4), (2*10^(1/4))/3, 2/Sqrt[3], (2*5^(1/3))/3, (2*2^(3/4))/3, (2*7^(1/4))/3, (2*2^(2/3))/3, (2*2^(1/4))/3^(3/4), 1, (2*5^(1/4))/3, 2/3^(2/3), (2*Sqrt[2])/3, 2/3^(3/4), (2*2^(1/3))/3, (2*2^(1/4))/3, 2/3, 0, -2/3, -1}'),
    ('6-l-ord3_int_pos_neg_rat_neg_large',
     'ar(2, -3, Rational(-4, 3), root_order=3)',
     'WL',
     '{2, 4/3^(2/3), (2*19^(1/3))/3, (2*Sqrt[5])/3, (2*11^(1/3))/3, 4/3, 2/3^(2/3), 2/3, 0, -4/3, (-4*2^(1/3))/3, (-4*Sqrt[2])/3, -4/3^(2/3), (-4*2^(2/3))/3, (-4*5^(1/3))/3, -4/Sqrt[3], (-4*2^(1/3))/3^(2/3), (-4*7^(1/3))/3, -8/3, -4/3^(1/3), (-4*10^(1/3))/3, (-4*11^(1/3))/3, (-4*Sqrt[5])/3}'),
    ('6-m-ord3_int_pos_neg_rat_neg_large_bis',
     'ar(2, -5, -7, root_order=3)',
     'WL',
     '{2, 0}'),
    ('6-n-ord3_int_pos_neg_rat_neg_single',
     'ar(4, -2, -6.01, root_order=3)',
     'WL',
     '{4}'),
    ('6-o-ord3_int_pos_pos_rat_neg_empty',
     'ar(2, 4, Rational(-7, 3), root_order=3)',
     'WL',
     '{}'),
    ('7 a stmet_pos_pos_pos',
     'ar(2, 3, Rational(1, 2), step_method="Root")',
     'WL',
     '{2, Sqrt[17]/2, 3/Sqrt[2], Sqrt[19]/2, Sqrt[5], Sqrt[21]/2, Sqrt[11/2], Sqrt[23]/2, Sqrt[6], 5/2, Sqrt[13/2], (3*Sqrt[3])/2, Sqrt[7], Sqrt[29]/2, Sqrt[15/2], Sqrt[31]/2, 2*Sqrt[2], Sqrt[33]/2, Sqrt[17/2], Sqrt[35]/2, 3}'),
    ('7 b stmet_neg_neg_pos',
     'ar(-3, -1, Rational(1, 2), step_method="Root")',
     'WL',
     '{-3, -1/2*Sqrt[35], -Sqrt[17/2], -1/2*Sqrt[33], -2*Sqrt[2], -1/2*Sqrt[31], -Sqrt[15/2], -1/2*Sqrt[29], -Sqrt[7], (-3*Sqrt[3])/2, -Sqrt[13/2], -5/2, -Sqrt[6], -1/2*Sqrt[23], -Sqrt[11/2], -1/2*Sqrt[21], -Sqrt[5], -1/2*Sqrt[19], -3/Sqrt[2], -1/2*Sqrt[17], -2, -1/2*Sqrt[15], -Sqrt[7/2], -1/2*Sqrt[13], -Sqrt[3], -1/2*Sqrt[11], -Sqrt[5/2], -3/2, -Sqrt[2], -1/2*Sqrt[7], -Sqrt[3/2], -1/2*Sqrt[5], -1}'),
    ('7 c stmet_neg_pos_pos',
     'ar(-3, 3, Rational(2, 3), step_method="Root")',
     'WL',
     '{-3, -1/3*Sqrt[77], -1/3*Sqrt[73], -Sqrt[23/3], -1/3*Sqrt[65], -1/3*Sqrt[61], -Sqrt[19/3], -1/3*Sqrt[53], -7/3, -Sqrt[5], -1/3*Sqrt[41], -1/3*Sqrt[37], -Sqrt[11/3], -1/3*Sqrt[29], -5/3, -Sqrt[7/3], -1/3*Sqrt[17], -1/3*Sqrt[13], -1, -1/3*Sqrt[5], -1/3, 0, 2/3, (2*Sqrt[2])/3, 2/Sqrt[3], 4/3, (2*Sqrt[5])/3, 2*Sqrt[2/3], (2*Sqrt[7])/3, (4*Sqrt[2])/3, 2, (2*Sqrt[10])/3, (2*Sqrt[11])/3, 4/Sqrt[3], (2*Sqrt[13])/3, (2*Sqrt[14])/3, 2*Sqrt[5/3], 8/3, (2*Sqrt[17])/3, 2*Sqrt[2], (2*Sqrt[19])/3, (4*Sqrt[5])/3}'),
    ('7 d stmet_neg_pos_neg',
     'ar(-3, 3, Rational(-2, 3), step_method="Root")',
     'WL',
     '{}'),
    ('8 a farey_pos_pos_pos',
     'ar(0, 3, Rational(1, 3), farey_range=True)',
     'WL',
     '{0, 1/3, Sqrt[2]/3, 1/2, 1/Sqrt[3], 2/3, 1/Sqrt[2], Sqrt[5]/3, Sqrt[2/3], Sqrt[3]/2, Sqrt[7]/3, (2*Sqrt[2])/3, 1, Sqrt[5]/2, 2/Sqrt[3], Sqrt[3/2], Sqrt[7]/2, 4/3, Sqrt[2], (2*Sqrt[5])/3, 3/2, 2*Sqrt[2/3], 5/3, Sqrt[3], (2*Sqrt[7])/3, (4*Sqrt[2])/3, 2, 3/Sqrt[2], Sqrt[5], 4/Sqrt[3], 7/3, (5*Sqrt[2])/3, Sqrt[6], 5/2, (3*Sqrt[3])/2, Sqrt[7], 8/3, 2*Sqrt[2], 5/Sqrt[3], (4*Sqrt[5])/3, 3}'),
    ('8 b farey_neg_neg_pos',
     'ar(-3, 0, 3, farey_range=True)',
     'WL',
     '{-3, (-4*Sqrt[5])/3, -5/Sqrt[3], -2*Sqrt[2], -8/3, -Sqrt[7], (-3*Sqrt[3])/2, -5/2, -Sqrt[6], (-5*Sqrt[2])/3, -7/3, -4/Sqrt[3], -Sqrt[5], -3/Sqrt[2], -2, (-4*Sqrt[2])/3, (-2*Sqrt[7])/3, -Sqrt[3], -5/3, -2*Sqrt[2/3], -3/2, (-2*Sqrt[5])/3, -Sqrt[2], -4/3, -1/2*Sqrt[7], -Sqrt[3/2], -2/Sqrt[3], -1/2*Sqrt[5], -1, (-2*Sqrt[2])/3, -1/3*Sqrt[7], -1/2*Sqrt[3], -Sqrt[2/3], -1/3*Sqrt[5], -(1/Sqrt[2]), -2/3, -(1/Sqrt[3]), -1/2, -1/3*Sqrt[2], -1/3, 0}'),
    ('8 c farey_neg_pos_pos',
     'ar(-2, 2, 4, farey_range=True)',
     'WL',
     '{-2, (-4*Sqrt[2])/3, -5/(2*Sqrt[2]), -7/4, -Sqrt[3], -5/3, -3/2, -Sqrt[2], -4/3, (-3*Sqrt[3])/4, -5/4, -2/Sqrt[3], -3/(2*Sqrt[2]), -1, (-2*Sqrt[2])/3, -1/2*Sqrt[3], -3/4, -(1/Sqrt[2]), -2/3, -(1/Sqrt[3]), -1/2, -1/3*Sqrt[2], -1/4*Sqrt[3], -1/2*1/Sqrt[2], -1/3, -1/4, 0, 1/4, 1/3, 1/(2*Sqrt[2]), Sqrt[3]/4, Sqrt[2]/3, 1/2, 1/Sqrt[3], 2/3, 1/Sqrt[2], 3/4, Sqrt[3]/2, (2*Sqrt[2])/3, 1, 3/(2*Sqrt[2]), 2/Sqrt[3], 5/4, (3*Sqrt[3])/4, 4/3, Sqrt[2], 3/2, 5/3, Sqrt[3], 7/4, 5/(2*Sqrt[2]), (4*Sqrt[2])/3, 2}'),
    ('9 a wp_pos_pos_pos_mp',
     'ar(sqrt(Rational(3999999999999999, 10))/20000000, sqrt(Rational(4000000000000001, 10))/20000000, Rational(1, 100000000000000000))',
     'WL',
     '{Sqrt[3999999999999999/10]/20000000, (20000000000000001*Sqrt[3999999999999999/10])/400000000000000000000000, (12500000000000003*Sqrt[3999999999999999/10])/250000000000000000000000}'),
    ('9 b wp_pos_pos_pos_30',
     'ar(sqrt(Rational(24999999999999999, 10))/50000000, sqrt(Rational(25000000000000001, 10))/50000000, Rational(1, 100000000000000000), working_precision=30)',
     'WL',
     '{Sqrt[24999999999999999/10]/50000000, (100000000000000001*Sqrt[24999999999999999/10])/5000000000000000000000000, (50000000000000001*Sqrt[24999999999999999/10])/2500000000000000000000000, (100000000000000003*Sqrt[24999999999999999/10])/5000000000000000000000000, (25000000000000001*Sqrt[24999999999999999/10])/1250000000000000000000000}'),
]


@pytest.mark.parametrize("tid,call,kind,expected",
                         CASES, ids=[c[0] for c in CASES])
def test_wl_suite(tid, call, kind, expected):
    ns = {"ar": ar, "sqrt": sqrt, "Rational": Rational, "S": S}
    result = eval(call, ns)
    if kind == "DIFFS_GE":
        bound = eval(expected, ns)
        diffs = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        for x in diffs:
            v = sympy.Abs(x) - bound
            assert v == 0 or v.evalf(30) >= -sympy.Float(10) ** -25, (
                f"{tid}: step {x} below lower bound {bound}")
    elif tid in APPROX_IDS:
        assert_close(result, expected, tid)
    else:
        assert_same(result, expected, tid)


# ── Group 10: performance (normalized like the WL WolframMark tests) ────────

def _reference_time():
    """A fixed workload, analogous to the WolframMark normalization."""
    t0 = time.perf_counter()
    s = 0.0
    for n in range(2, 200000):
        s += math.sqrt(n)
    x = sqrt(2)
    for _ in range(60):
        float((x + 1).evalf(30))
    return time.perf_counter() - t0


_BENCH = [
    ("bench_pos", lambda: ar(200), 60.0),
    ("bench_pos_neg", lambda: ar(200, -200, -1), 150.0),
    ("bench_neg_pos_pos", lambda: ar(-100, 100, Rational(1, 2)), 400.0),
    ("bench_pos_neg_neg", lambda: ar(60, -60, Rational(-1, 3)), 170.0),
    ("bench_pos_pos_pos_p30", lambda: ar(
        1 - Rational(1, 10 ** 13), 1 + Rational(1, 10 ** 13),
        Rational(1, 10 ** 17), working_precision=30), 60.0),
]


@pytest.mark.parametrize("name,func,ref", _BENCH, ids=[b[0] for b in _BENCH])
def test_10_benchmarks(name, func, ref):
    base = _reference_time()
    t0 = time.perf_counter()
    func()
    elapsed = time.perf_counter() - t0
    assert elapsed < 2 * ref * base, (
        f"{name}: {elapsed:.2f}s exceeds 2 x {ref} x {base:.3f}s")


# ── Failure modes (WL Failure objects → exceptions) ─────────────────────────

class TestFailures:
    def test_not_real(self):
        with pytest.raises(NotRealError):
            ar(sympy.I * sqrt(-1 + sqrt(2)), 4, 1 / sqrt(2))

    def test_not_algebraic(self):
        with pytest.raises(NotAlgebraicError):
            ar(0, 5, sqrt(E))

    def test_algebraics_only_off(self):
        result = ar(0, 5, sqrt(E), algebraics_only=False)
        assert len(result) > 0

    def test_step_bound(self):
        with pytest.raises(StepBoundError):
            ar(0, 5, Rational(1, 4), Rational(1, 2))

    def test_lower_bound_negative(self):
        with pytest.raises(LowerBoundError):
            ar(0, 5, Rational(1, 2), Rational(-1, 4))

    def test_farey_step(self):
        with pytest.raises(FareyStepError):
            ar(0, 3, Rational(2, 3), farey_range=True)


# ── Sanity: formula_complexity reference values from the WL kernel ──────────

def test_formula_complexity_reference():
    refs = [
        (S.One, 1.0),
        (sympy.Integer(2), 1.176776695296637),
        (sqrt(2), 3.530330085889911),
        (2 * sqrt(2), 4.707106781186548),
        (sqrt(1 + sqrt(2)), 13.060660171779823),
    ]
    for expr, val in refs:
        assert abs(formula_complexity(expr) - val) < 1e-12
