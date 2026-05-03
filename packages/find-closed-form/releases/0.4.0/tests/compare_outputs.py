"""Compare Python find_closed_form outputs with known WL results on ALL test cases."""

import signal
import sympy
from sympy import (
    Rational, pi, sqrt, log, exp, sin, asin, acos, atan, acot,
    gamma as spgamma, polygamma, zeta, sinh, cosh, sech, csch, asinh,
    Catalan, EulerGamma, GoldenRatio,
)
from find_closed_form import find_closed_form, formula_complexity
from fractions import Fraction
import time


class Timeout(Exception):
    pass


def _handler(signum, frame):
    raise Timeout()


def _neval(expr, n=18):
    return float(expr.evalf(n=n))


def run_case(label, call_fn, wl_result, timeout=120):
    t0 = time.time()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        result = call_fn()
        elapsed = time.time() - t0
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

        if result is None:
            return label, wl_result, "None", "FAIL", elapsed
        if isinstance(result, list):
            py_str = f"[{len(result)} results]"
            if len(result) > 0:
                py_str += f" first: {result[0]}"
        else:
            py_str = str(result)

        return label, wl_result, py_str, "OK", elapsed
    except Timeout:
        elapsed = time.time() - t0
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return label, wl_result, "TIMEOUT", "TIMEOUT", elapsed
    except Exception as e:
        elapsed = time.time() - t0
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return label, wl_result, f"ERROR: {e}", "ERROR", elapsed


CASES = []

# ── Basic Examples ──
CASES.append(("Basic: fcf[0.405465]",
    lambda: find_closed_form(0.405465),
    "Log[3/2]"))

CASES.append(("Basic: fcf[3.792277]",
    lambda: find_closed_form(3.792277),
    "1/6 + Gamma[1/4]"))

CASES.append(("Basic: fcf[3.311601]",
    lambda: find_closed_form(3.311601),
    "5/3 + Pi^2/6"))

CASES.append(("Basic: fcf[1.044866]",
    lambda: find_closed_form(1.044866),
    "1/Sqrt[Catalan]"))

CASES.append(("Basic: fcf[1.85653, 1/Zeta[#]^2]",
    lambda: find_closed_form(1.85653, functions=lambda x: 1/zeta(x)**2),
    "Zeta[1/5]^(-2)"))

# ── Scope ──
CASES.append(("Scope: fcf[0.405465, Log, 10]",
    lambda: find_closed_form(0.405465, functions=lambda x: log(x),
                             max_results=10, max_search_rounds=20, search_time_limit=120),
    "10 results, first=Log[3/2]"))

CASES.append(("Scope: fcf[-1.1857322, PolyGamma]",
    lambda: find_closed_form(-1.1857322, functions=lambda x: polygamma(0, x)),
    "7/9 + PolyGamma[0, 1/2]"))

CASES.append(("Scope: fcf[0.780653, ArcSinh]",
    lambda: find_closed_form(0.780653, functions=lambda x: asinh(x)),
    "Sqrt[5]*ArcSinh[4]/6"))

CASES.append(("Scope: fcf[7.443967, Log[1+Exp[#]]]",
    lambda: find_closed_form(7.443967, functions=lambda x: log(1+exp(x))),
    "10*Log[1+E^(1/10)]"))

CASES.append(("Scope: fcf[4.688231, Gamma ratio]",
    lambda: find_closed_form(4.688231, functions=lambda x, y: spgamma(x)/spgamma(y),
                             search_time_limit=120),
    "2*Sqrt[3]*Gamma[1/4]/Gamma[1/3]"))

CASES.append(("Scope: fcf[5.550045, {Sinh,...}]",
    lambda: find_closed_form(5.550045, functions=[
        lambda x: sinh(x), lambda x: cosh(x),
        lambda x: sech(x), lambda x: csch(x)]),
    "6*Sech[2/5]"))

CASES.append(("Scope: fcf[1.85653, 1/Zeta[#]^2] (dup)",
    lambda: find_closed_form(1.85653, functions=lambda x: 1/zeta(x)**2),
    "Zeta[1/5]^(-2)"))

CASES.append(("Scope: fcf[3.940443, {ArcTrig}]",
    lambda: find_closed_form(3.940443, functions=[
        lambda x: asin(x), lambda x: acos(x),
        lambda x: atan(x), lambda x: acot(x)]),
    "4*ArcSin[5/6]"))

# ── Algebraic Options ──
CASES.append(("AlgOpts: AlgebraicAdd=False",
    lambda: find_closed_form(0.1013578,
        functions=lambda x, y: 1/(spgamma(x)*spgamma(y)),
        algebraic_add=False, search_time_limit=60),
    "1/(Sqrt[Pi]*Gamma[1/6])"))

CASES.append(("AlgOpts: AlgebraicFactor=False",
    lambda: find_closed_form(-9.6530201,
        functions=lambda x, y: polygamma(0, x) + polygamma(0, y),
        algebraic_factor=False, search_time_limit=120),
    "3+PolyGamma[0,1/7]+PolyGamma[0,1/5]"))

CASES.append(("AlgOpts: both=False, Sin",
    lambda: find_closed_form(0.25,
        functions=lambda x: sin(pi*x),
        algebraic_factor=False, algebraic_add=False),
    "direct match"))

# ── Formula Complexity ──
CASES.append(("Complexity: Gamma default",
    lambda: find_closed_form(38.94017, functions=lambda x: spgamma(x)),
    "151/4 + Gamma[7/9]"))

CASES.append(("Complexity: Gamma FC<=15",
    lambda: find_closed_form(38.94017, functions=lambda x: spgamma(x),
                             formula_complexity_threshold=15),
    "2*Gamma[1/20]"))

# ── Search Rounds ──
CASES.append(("Rounds: Gamma[1/50] Plain",
    lambda: find_closed_form(49.44221, functions=lambda x: spgamma(x),
        algebraic_add=False, algebraic_factor=False, search_range="Plain"),
    "Gamma[1/50]"))

CASES.append(("Rounds: None beyond 50",
    lambda: find_closed_form(59.43902, functions=lambda x: spgamma(x),
        algebraic_add=False, algebraic_factor=False, search_range="Plain"),
    "None"))

CASES.append(("Rounds: Gamma[1/60] 100rds",
    lambda: find_closed_form(59.43902, functions=lambda x: spgamma(x),
        max_search_rounds=100, algebraic_add=False, algebraic_factor=False,
        search_range="Plain", search_time_limit=120),
    "Gamma[1/60]"))

CASES.append(("Rounds: Log*Log Integer",
    lambda: find_closed_form(6.263643,
        functions=lambda x, y: log(x)*log(y), search_range="Integer"),
    "2*Log[5]*Log[7]"))

CASES.append(("Rounds: Gamma*Gamma Plain",
    lambda: find_closed_form(14.911818,
        functions=lambda x, y: spgamma(x)*spgamma(y),
        search_range="Plain", search_time_limit=120),
    "Gamma product"))

CASES.append(("Rounds: custom range Log",
    lambda: find_closed_form(13.165149, functions=lambda x: log(x),
        search_range_fn=lambda cut: [Fraction(i) for i in range(0, 100*cut+1, 25)]),
    "custom range Log result"))

# ── Gamma Squared ──
CASES.append(("GammaSq: Gamma[#]^2",
    lambda: find_closed_form(20.0758, functions=lambda x: spgamma(x)**2),
    "-1 + Gamma[1/5]^2"))

# ── Rational Solutions ──
CASES.append(("RatSol: RS=True",
    lambda: find_closed_form(0.25, functions=lambda x: sin(pi*x),
        rational_solutions=True, algebraic_add=False),
    "1/4"))

CASES.append(("RatSol: Identity=1/4",
    lambda: find_closed_form(0.25, functions=lambda x: x),
    "1/4"))

CASES.append(("RatSol: sin(pi*#)=0.5",
    lambda: find_closed_form(0.5, functions=lambda x: sin(pi*x),
        algebraic_add=False, algebraic_factor=False),
    "1/2"))

# ── Search Arguments ──
CASES.append(("SearchArgs: Gamma {3,1,1/3}",
    lambda: find_closed_form(4.678938, functions=lambda x: spgamma(x),
        search_arguments=[Fraction(3), Fraction(1), Fraction(1,3)]),
    "2 + Gamma[1/3]"))

CASES.append(("SearchArgs: Gamma ratio",
    lambda: find_closed_form(1.32325,
        functions=lambda x, y: spgamma(x)/spgamma(y),
        search_arguments=[[Fraction(1), Fraction(1,2)],
                          [Fraction(3), Fraction(1), Fraction(1,3)]]),
    "Gamma ratio result"))

# ── Significant Digits ──
CASES.append(("SigDig: relaxed zeta",
    lambda: find_closed_form(0.81248057539,
        functions=lambda x: 1/zeta(x)**2, significant_digits=7),
    "1/Zeta[11/3]^2"))

CASES.append(("SigDig: Log[2] 6 digits",
    lambda: find_closed_form(0.693147, functions=lambda x: log(x)),
    "Log[2]"))

# ── Properties and Relations ──
CASES.append(("Props: 0.666 → 2/3",
    lambda: find_closed_form(0.666, functions=lambda x: x),
    "2/3"))

CASES.append(("Props: 4.243 → 3*Sqrt[2]",
    lambda: find_closed_form(4.243, functions=lambda x: x),
    "3*Sqrt[2]"))

CASES.append(("Props: 0.5848 → 5^(-1/3)",
    lambda: find_closed_form(0.5848, functions=lambda x: x),
    "5^(-1/3)"))


# ── Run all ──
print(f"{'#':<3} {'Test':<40} {'WL expected':<35} {'Python result':<40} {'Status':>7} {'Time':>7}")
print("=" * 135)

ok = diff = fail = timeout = 0

for i, (label, call_fn, wl_result) in enumerate(CASES, 1):
    _, _, py_str, status, elapsed = run_case(label, call_fn, wl_result, timeout=180)

    if status == "OK":
        ok += 1
    elif status == "FAIL":
        fail += 1
    elif status == "TIMEOUT":
        timeout += 1
    else:
        fail += 1

    py_display = py_str[:38] if len(py_str) > 38 else py_str
    wl_display = wl_result[:33] if len(wl_result) > 33 else wl_result
    print(f"{i:<3} {label:<40} {wl_display:<35} {py_display:<40} {status:>7} {elapsed:>6.1f}s")

print()
print(f"Total: {len(CASES)} | OK: {ok} | Failed: {fail} | Timeout: {timeout}")
