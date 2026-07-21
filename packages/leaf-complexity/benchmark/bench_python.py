"""Benchmark the Python leaf_complexity against the Wolfram Language
LeafComplexity resource function.

Run from the package root (packages/leaf-complexity):

    python benchmark/bench_python.py

and compare with the Wolfram side:

    wolframscript -file benchmark/bench_wolfram.wl

Both scripts print their results as JSON over the same expression
corpus; see BENCHMARK.md for the collected results.
"""

import json
import math
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sympy
from sympy import I, Rational, expand, symbols

from leaf_complexity import leaf_complexity

x, y, z = symbols("x y z")

MAX_SECONDS = 2.0
MAX_REPS = 1000


def build_cases():
    small = (x + 2) / (y - 2)
    medium = (2 * x**Rational(1, 3) + I) / (x - 2 - 3 * I) - 5 / x**2
    poly2v = expand((x + y + 1)**15)
    poly3v = expand((x + y + z + 1)**12)
    nested = [[[i, j, Rational(i, j)] for j in range(1, 21)]
              for i in range(1, 21)]
    scaled_f = lambda v: math.log(1.0 + abs(v))
    return [
        ("small", lambda: leaf_complexity(small)),
        ("medium", lambda: leaf_complexity(medium)),
        ("poly-2var-15", lambda: leaf_complexity(poly2v)),
        ("poly-3var-12", lambda: leaf_complexity(poly3v)),
        ("nested-table", lambda: leaf_complexity(nested)),
        ("poly-2var-15-scaled", lambda: leaf_complexity(poly2v, scaled_f)),
        ("poly-2var-15-proper",
         lambda: leaf_complexity(poly2v, heads=False)),
    ]


def time_case(thunk):
    value = thunk()  # warm-up
    times = []
    start = time.perf_counter()
    while (time.perf_counter() - start < MAX_SECONDS
           and len(times) < MAX_REPS):
        t0 = time.perf_counter()
        thunk()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), len(times), value


def main():
    results = []
    for name, thunk in build_cases():
        mean, reps, value = time_case(thunk)
        results.append({
            "case": name,
            "ms": round(mean * 1000, 4),
            "reps": reps,
            "value": float(value),
        })
    print(json.dumps({
        "python": platform.python_version(),
        "sympy": sympy.__version__,
        "machine": platform.machine(),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
