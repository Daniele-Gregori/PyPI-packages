"""Benchmark transcendental_range (Python) — paired with bench_wolfram.wl.

Times the default implementation over the same cases as the Wolfram
script (the "tests of efficiency" ranges of the definition notebook) and
prints the results as JSON.

Run from the package root:  python benchmark/bench_python.py
"""

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sympy import Rational  # noqa: E402

from transcendental_range import transcendental_range  # noqa: E402
import transcendental_range as _pkg  # noqa: E402

CASES = [
    ("exp", (1, 100, 1), {}),
    ("exp", (100, -100, -1), {}),
    ("log", (1, 100, 1), {}),
    ("log", (100, -100, -1), {}),
    ("tanh", (1, 10, 1), {}),
    ("coth", (1, 30, 1), {}),
    ("sech", (1, 100, 1), {}),
    ("csch", (1, 100, 1), {}),
    ("asinh", (1, 100, 1), {}),
    ("acosh", (1, 100, 1), {}),
    ("atanh", (0, 100, Rational(1, 10)), {}),
    ("asin", (0, 100, Rational(1, 10)), {}),
    ("acos", (0, 10, Rational(1, 10)), {}),
    ("atan", (1, 100, 1), {}),
    ("asec", (1, 100, 1), {}),
    ("acsc", (1, 100, 1), {}),
    ("power", (1, 20, 1), {"generators_domain": "algebraics"}),
    ("all", (-2, 2, Rational(1, 3)), {}),
]


def time_case(method, args, kwargs, budget=5.0, max_reps=50):
    """Mean wall-clock seconds over repetitions within a time budget
    (one warm-up call first)."""
    result = transcendental_range(*args, method=method, **kwargs)
    times = []
    start = time.perf_counter()
    while len(times) < max_reps and time.perf_counter() - start < budget:
        t0 = time.perf_counter()
        transcendental_range(*args, method=method, **kwargs)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), len(result)


def main():
    out = {
        "python": sys.version.split()[0],
        "package": _pkg.core.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "cases": [],
    }
    for method, args, kwargs in CASES:
        mean, n = time_case(method, args, kwargs)
        out["cases"].append({
            "method": method,
            "args": [str(a) for a in args],
            "opts": {k: str(v) for k, v in kwargs.items()},
            "mean_ms": round(mean * 1000, 3),
            "length": n,
        })
        print(f"{method:8s} {str(args):28s} {mean*1000:10.2f} ms   "
              f"n={n}", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
