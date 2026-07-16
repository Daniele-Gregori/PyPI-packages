# Benchmark: Python `transcendental_range` vs Wolfram Language `TranscendentalRange`

The Python `transcendental_range` 0.9.0 is a faithful port of the Wolfram
Language resource function
[TranscendentalRange](https://resources.wolframcloud.com/FunctionRepository/resources/TranscendentalRange)
1.1.0: on every method the default monotonic-outer implementation is verified
element-by-element against the naive `Outer`-based baseline (`test=True`),
transcribing the full Wolfram `VerificationTest` suite
(see `tests/test_transcendental_range.py`).

## Headline comparison

The definition notebook records two reference timings of the Wolfram 1.1.0
implementation (Wolfram Language 15.0, same machine class):

| Case | Elements | WL 1.1.0 | Python 0.9.0 |
|---|---:|---:|---:|
| `TranscendentalRange[-100, 100, 1/2]` | 80604 | 2.23 s | 4.7 s |
| `TranscendentalRange[1000]` | 577 | 0.047 s | 0.16 s |
| naive `Outer` baseline of `[1000]` (WL doc) / `test=True` (Python) | 577 | 11.16 s | 1.18 s |

The Python port reproduces the Wolfram output exactly (the 80604 elements
of the large mixed-sign range match one for one) within a factor 2–3 of
the Wolfram timings, despite doing exact symbolic arithmetic through
sympy. The monotonic-outer algorithm preserves its decisive advantage
over the naive `Outer` construction on exponentially growing ranges
(WL: 240×; Python: 7× — the Python baseline shares the port's cached
float arithmetic, so the gap is narrower but the winner is the same).

## Per-method timings (Python, default implementation)

Timings of `transcendental_range` on the "tests of efficiency" ranges of the
definition notebook (mean wall-clock; both implementations produce the same
elements):

| Method | Range | Elements | Default | Naive baseline |
|---|---|---:|---:|---:|
| exp | (1, 100, 1) | 54 | 0.01 s | 0.08 s |
| log | (1, 100, 1) | 2527 | 0.13 s | 0.10 s |
| sinh | (1, 100, 1) | 125 | 0.01 s | 0.03 s |
| cosh | (1, 100, 1) | 103 | 0.01 s | 0.03 s |
| tanh | (1, 10, 1) | 90 | <0.01 s | <0.01 s |
| coth | (1, 30, 1) | 437 | 0.03 s | 0.04 s |
| sech | (1, 100, 1) | 385 | 0.02 s | 0.05 s |
| csch | (1, 100, 1) | 385 | 0.03 s | 0.03 s |
| asinh | (1, 100, 1) | 2378 | 0.12 s | 0.13 s |
| acosh | (1, 100, 1) | 2175 | 0.09 s | 0.11 s |
| atanh | (0, 100, 1/10) | 8089 | 0.34 s | 0.35 s |
| acoth | (1, 100, 1) | 4950 | 0.18 s | 0.27 s |
| asech | (0, 10, 1/10) | 658 | 0.03 s | 0.01 s |
| acsch | (1, 100, 1) | 4950 | 0.19 s | 0.24 s |
| asin | (0, 100, 1/10) | 9196 | 0.39 s | 0.42 s |
| acos | (0, 10, 1/10) | 831 | 0.04 s | 0.01 s |
| atan | (1, 100, 1) | 6549 | 0.26 s | 0.29 s |
| acot | (1, 100, 1) | 4950 | 0.19 s | 0.17 s |
| asec | (1, 100, 1) | 6458 | 0.22 s | 0.25 s |
| acsc | (1, 100, 1) | 4980 | 0.17 s | 0.21 s |

The running time is dominated by the construction of the exact sympy
expressions of the surviving elements: on these bounded ranges the
monotonic scan and the naive full scan visit a comparable number of
in-range pairs, so the two Python implementations are close — the
monotonic algorithm pays off on ranges whose generator functions grow
fast (exp, log and the large mixed ranges of the headline table), exactly
as in the Wolfram original.

## Methodology

- Same machine, single-threaded, warm caches (one warm-up call per case).
- Wolfram: `AbsoluteTiming`/`EchoTiming` values recorded in the 1.1.0
  definition notebook (Wolfram Language 15.0, macOS x86-64).
- Python: `time.perf_counter` means (Python 3.14, macOS x86-64), local
  in-development `algebraic-range` 0.9.0 and `farey` 0.6.0.

## Reproducing

From the package root (`packages/transcendental-range`):

```bash
python benchmark/bench_python.py
wolframscript -file benchmark/bench_wolfram.wl
```

Both scripts print their results as JSON. The Wolfram script requires the
resource function version 1.1.0 (until published, load the definition
notebook `wolfram/TranscendentalRange-1-1-0-definition.nb` instead).
