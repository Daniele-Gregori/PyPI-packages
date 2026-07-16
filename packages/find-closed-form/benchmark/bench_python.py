"""Benchmark find_closed_form (Python) — paired with bench_wolfram.wl.

Times representative closed-form searches over the same cases as the
Wolfram script (README/verification-suite searches with known results)
and prints the results as JSON.

Run from the package root:  python benchmark/bench_python.py
(with the sibling range packages on PYTHONPATH if not installed).
"""

import json
import platform
import sys
import time
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_pkg_root / "src"))
for _sibling in ("farey", "algebraic-range", "transcendental-range"):
    sys.path.insert(0, str(_pkg_root.parent / _sibling / "src"))

from sympy import asinh, exp, gamma, log, zeta, sinh, cosh, sech, csch  # noqa: E402

import find_closed_form as _pkg  # noqa: E402
from find_closed_form import find_closed_form  # noqa: E402

CASES = [
    ("default -> log(3/2)", (0.405465,), {}),
    ("default -> 1/6 + gamma(1/4)", (3.792277,), {}),
    ("default -> 1/sqrt(Catalan)", (1.044866,), {}),
    ("1/zeta(#)^2 -> zeta(1/5)^-2", (1.85653, lambda x: 1 / zeta(x) ** 2), {}),
    ("asinh -> sqrt(5)/6*asinh(4)", (0.780653, lambda x: asinh(x)), {}),
    ("log(1+exp(#)) -> 10*log(1+exp(1/10))",
     (7.443967, lambda x: log(1 + exp(x))), {}),
    ("{sinh,cosh,sech,csch} -> 6*sech(2/5)",
     (5.550045, [lambda x: sinh(x), lambda x: cosh(x),
                 lambda x: sech(x), lambda x: csch(x)]), {}),
    ("gamma(#1)/gamma(#2) (2-arg)",
     (4.688231, lambda x, y: gamma(x) / gamma(y)), {}),
    ("log, 10 results", (0.405465, lambda x: log(x), 10), {}),
    ("search_range=Algebraic -> exp(sqrt(2))",
     (4.1132503787829275,), {"search_range": "Algebraic"}),
]


def time_case(args, kwargs, budget=12.0, max_reps=5):
    """Mean wall-clock seconds over repetitions within a time budget
    (one warm-up call first)."""
    result = find_closed_form(*args, **kwargs)
    times = []
    start = time.perf_counter()
    while len(times) < max_reps and time.perf_counter() - start < budget:
        t0 = time.perf_counter()
        find_closed_form(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), result


def main():
    # Build the algebraic lookup table (~2 s, lazy) outside the timings,
    # like the WL kernel which constructs absLookupBase at load time.
    find_closed_form(0.5, lambda x: x, rational_solutions=True,
                     search_time_limit=10)

    out = {
        "python": sys.version.split()[0],
        "package": _pkg.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "cases": [],
    }
    for label, args, kwargs in CASES:
        mean, result = time_case(args, kwargs)
        out["cases"].append({
            "case": label,
            "mean_s": round(mean, 3),
            "result": str(result),
        })
        print(f"{label:42s} {mean:8.2f} s   {result}", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
