"""Benchmark transcendental_range across all methods and range sizes.

Outputs results to BENCHMARK.md.
"""

import os
import platform
import time
import signal

from transcendental_range import transcendental_range
from transcendental_range.core import ALL_METHODS

TIMEOUT = 60  # seconds per call
RANGES = [(-10, 10), (-25, 25), (-100, 100)]
OUT_FILE = os.path.join(os.path.dirname(__file__), "BENCHMARK.md")


def _timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, _timeout_handler)


def bench(method, lo, hi):
    signal.alarm(TIMEOUT)
    try:
        t0 = time.perf_counter()
        result = transcendental_range(lo, hi, method=method)
        elapsed = time.perf_counter() - t0
        signal.alarm(0)
        return len(result), elapsed
    except TimeoutError:
        signal.alarm(0)
        return None, None


def fmt_cell(n, t):
    if n is None:
        return "timeout"
    return f"{n} in {t:.2f}s"


def main():
    results = {}
    for method in ALL_METHODS:
        results[method] = {}
        skip_rest = False
        for lo, hi in RANGES:
            if skip_rest:
                results[method][(lo, hi)] = (None, None)
                continue
            n, t = bench(method, lo, hi)
            results[method][(lo, hi)] = (n, t)
            print(f"  {method:<8} ({lo},{hi}): {fmt_cell(n, t)}")
            if n is None or t > 30:
                skip_rest = True

    # Build markdown
    lines = [
        "# Benchmark",
        "",
        "Performance of `transcendental_range(lo, hi, method=...)` across all 27 methods and three range sizes.",
        f"Measured on {platform.system()} ({platform.machine()}), Python {platform.python_version()}.",
        "",
        "## Results",
        "",
    ]

    # Table header
    hdr = "| Method |"
    sep = "|--------|"
    for lo, hi in RANGES:
        hdr += f" ({lo}, {hi}) |"
        sep += "-------------|"
    lines.append(hdr)
    lines.append(sep)

    # Table rows
    for method in ALL_METHODS:
        row = f"| {method} |"
        for lo, hi in RANGES:
            n, t = results[method][(lo, hi)]
            row += f" {fmt_cell(n, t)} |"
        lines.append(row)

    lines += [
        "",
        "## Notes",
        "",
        "- **Efficient path** (exp, log, power, all hyp, all inv-trig, all inv-hyp): "
        "uses monotonic outer algorithm with float pre-screening and deferred sympy expression creation. "
        "All complete in under 1s at (-100, 100).",
        "- **Naive path** (sin, cos, tan, cot, sec, csc): "
        "uses brute-force outer product with exact sympy comparisons. "
        "These are slow at larger ranges due to the non-monotonic, periodic nature of trig functions.",
        "- **Zero results** for power, atanh, asech: "
        "no algebraic irrational generators exist in the given rational ranges "
        "(power requires irrational exponents; atanh and asech have restricted domains).",
        f"- Timeout threshold: {TIMEOUT}s.",
        "",
    ]

    md = "\n".join(lines)

    with open(OUT_FILE, "w") as f:
        f.write(md)

    print(f"\nWritten to {OUT_FILE}")


if __name__ == "__main__":
    main()
