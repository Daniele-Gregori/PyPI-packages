"""Benchmark ``spreadsheet_trace`` (Python) on the example workbooks.

Run from anywhere:

    python benchmark/bench_python.py

Methodology: for each (file, cell) case the trace runs once as a warm-up
(which also fills the ``import_all`` cache, matching the Wolfram
``ImportOnce`` behavior), then repeatedly for ~5 seconds (at least 5, at
most 200 runs); the mean and minimum wall-clock times are reported.
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from spreadsheet_toolkit import spreadsheet_trace  # noqa: E402

DATA = PACKAGE_ROOT / "tests" / "data"

CASES = [
    ("example_01.xlsx", "D1"),
    ("example_01.xlsx", "E1"),
    ("example_02.xlsx", "C5"),
    ("example_02.xlsx", "D1"),
    ("example_03.xlsx", "B18"),
    ("example_03.xlsx", "F9"),
    ("example_04.xlsx", "Summary!B3"),
    ("example_05.xlsx", "Orders!D2"),
    ("example_05.xlsx", "Dashboard!B3"),
    ("example_07.xlsx", "Budget!B4"),
    ("example_08.xlsx", "Catalog!I2"),
]


def main() -> None:
    results = []
    for name, cell in CASES:
        path = str(DATA / name)
        spreadsheet_trace(path, cell)  # warm-up (also fills the import cache)
        times = []
        deadline = time.perf_counter() + 5.0
        while len(times) < 200 and (time.perf_counter() < deadline or len(times) < 5):
            t0 = time.perf_counter()
            spreadsheet_trace(path, cell)
            times.append(time.perf_counter() - t0)
        results.append({
            "file": name,
            "cell": cell,
            "runs": len(times),
            "mean_ms": round(statistics.mean(times) * 1000, 3),
            "min_ms": round(min(times) * 1000, 3),
        })

    print(json.dumps({
        "python": platform.python_version(),
        "machine": platform.machine(),
        "results": results,
    }, indent=1))


if __name__ == "__main__":
    main()
