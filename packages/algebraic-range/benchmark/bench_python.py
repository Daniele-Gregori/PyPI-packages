"""
Benchmark algebraic-range (Python) on the five reference cases of the
Wolfram Language AlgebraicRange 2.0 test suite (group 10).

Run from the package root:

    python benchmark/bench_python.py

Prints results as JSON, including interpreter and library versions.
"""

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sympy
from sympy import Rational

from algebraic_range import algebraic_range, __version__

CASES = [
    ("algebraic_range(200)",
     lambda: algebraic_range(200)),
    ("algebraic_range(200, -200, -1)",
     lambda: algebraic_range(200, -200, -1)),
    ("algebraic_range(-100, 100, 1/2)",
     lambda: algebraic_range(-100, 100, Rational(1, 2))),
    ("algebraic_range(60, -60, -1/3)",
     lambda: algebraic_range(60, -60, Rational(-1, 3))),
    ("algebraic_range(1 - 10^-13, 1 + 10^-13, 10^-17, wp=30)",
     lambda: algebraic_range(1 - Rational(1, 10 ** 13),
                             1 + Rational(1, 10 ** 13),
                             Rational(1, 10 ** 17),
                             working_precision=30)),
]

BUDGET = 20.0  # seconds of repetitions per case


def bench(func):
    func()  # warm-up
    times = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < BUDGET and len(times) < 20:
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times), len(result)


def main():
    results = []
    for name, func in CASES:
        t_min, t_mean, length = bench(func)
        results.append({"case": name, "min_s": round(t_min, 4),
                        "mean_s": round(t_mean, 4), "length": length})
        print(f"{name}: mean {t_mean:.3f} s (min {t_min:.3f} s), "
              f"len={length}", file=sys.stderr)
    print(json.dumps({
        "package": f"algebraic-range {__version__}",
        "python": platform.python_version(),
        "sympy": sympy.__version__,
        "machine": platform.machine(),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
