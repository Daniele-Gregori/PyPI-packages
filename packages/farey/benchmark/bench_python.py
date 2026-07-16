"""Benchmark farey_range (Python) — paired with bench_wolfram.wl.

Times the exact Farey-range implementation over the same cases as the
Wolfram script and prints the results as JSON.

Run from the package root:  python benchmark/bench_python.py
"""

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farey import farey_range  # noqa: E402
import farey as _pkg  # noqa: E402

# Cases the WL resource function supports directly (ascending, integer step).
CASES = [
    (0, 1, 3),
    (0, 10, 5),
    (-20, 20, 6),
    (0, 30, 7),
    (0, 50, 4),
    (0, 100, 3),
    (0, 200, 5),
    (0, 1000, 2),
]


def time_case(args, budget=5.0, max_reps=2000):
    """Mean wall-clock seconds over repetitions within a time budget
    (one warm-up call first)."""
    result = farey_range(*args)
    times = []
    start = time.perf_counter()
    while len(times) < max_reps and time.perf_counter() - start < budget:
        t0 = time.perf_counter()
        farey_range(*args)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), len(result)


def main():
    out = {
        "python": sys.version.split()[0],
        "package": _pkg.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "cases": [],
    }
    for args in CASES:
        mean, n = time_case(args)
        out["cases"].append({
            "args": list(args),
            "mean_ms": round(mean * 1000, 4),
            "length": n,
        })
        print(f"{str(args):18s} {mean*1000:10.4f} ms   n={n}", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
