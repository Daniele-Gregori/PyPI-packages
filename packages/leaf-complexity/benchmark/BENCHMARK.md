# Benchmark: Python `leaf_complexity` vs Wolfram Language `LeafComplexity`

The Python `leaf_complexity` is a faithful port of the Wolfram Language resource function [LeafComplexity](https://resources.wolframcloud.com/FunctionRepository/resources/LeafComplexity): both produce identical values on the whole test corpus (see `tests/test_leaf_complexity.py`, which transcribes all the examples of the original definition notebook, plus `Heads -> False` and wrapping variants computed with the original definitions). This benchmark compares their wall-clock speed on the same machine, over identical expression trees; in every case below the two implementations also returned exactly the same value.

## Results

| Case | Expression | WL (ms) | Python (ms) | WL speed-up |
|---|---|---:|---:|---:|
| small | `(x + 2)/(y - 2)` | 0.0175 | 0.0382 | 2.2× |
| medium | `(2 x^(1/3) + I)/(x - 2 - 3 I) - 5/x^2` | 0.0465 | 0.120 | 2.6× |
| poly-2var-15 | `Expand[(x + y + 1)^15]` (136 terms) | 1.44 | 3.64 | 2.5× |
| poly-3var-12 | `Expand[(x + y + z + 1)^12]` (455 terms) | 5.60 | 15.8 | 2.8× |
| nested-table | `Table[{i, j, i/j}, {i, 20}, {j, 20}]` | 3.27 | 6.05 | 1.8× |
| poly-2var-15-scaled | poly-2var-15 with `f = Log[1. + Abs[#]]&` | 2.70 | 8.89 | 3.3× |
| poly-2var-15-proper | poly-2var-15 with `Heads -> False` | 1.11 | 3.17 | 2.8× |

**The Wolfram kernel is roughly 2–3× faster (geometric mean ≈ 2.5×).** This is the same pattern as the `farey` package and the opposite of `spreadsheet-toolkit`: the work here is a pure recursive tree scan with exact arithmetic — there is no parsing hot-path for Python's compiled regex engine to exploit, so the compiled Wolfram evaluator's constant factors win.

## Why the difference

The port keeps the recursion (and the head-first `Scan` order) identical, so the gap is all constant factors:

- **Node dispatch.** The WL implementation dispatches each node through compiled pattern matching on atomic number types (`_Integer | _Real`, `_Symbol`, `_Complex`, `_Rational`), while the Python port walks an `isinstance` chain per node and pays a Python-level closure call for every accumulation step.
- **SymPy trees carry extra structural work.** SymPy has no `Complex` or `Power[E, z]` atoms, so the port must probe every `Add`/`Mul` for Gaussian numeric literals (`_gaussian_parts`) to reproduce the WL tree — pure overhead on expressions, like the expanded polynomials, that contain none.
- **Accumulator arithmetic.** `AddTo[s, Abs@e]` on machine/big integers is a single kernel operation; the Python side allocates a SymPy `Integer` result per leaf when the leaves are SymPy numbers.

The gap is smallest on the nested table (where SymPy `Rational` decomposition is comparatively cheap) and largest with a custom `f` (where WL applies a compiled `Function` while Python crosses the interpreter boundary per leaf).

## Methodology

- Same machine, single-threaded; expressions are built once outside the timed region, and one warm-up call precedes timing.
- Wolfram: trimmed-mean wall-clock time from `RepeatedTiming[..., 2]` per case.
- Python: mean wall-clock time over up to 1000 repetitions within ~2 s per case (`time.perf_counter`).

## Environment

- Intel Core i9-9980HK @ 2.40 GHz (x86_64), macOS
- Wolfram Language 14.3, `LeafComplexity` 1.0.0 (loaded from `wolfram/LeafComplexity-1-0-0-kernel.wl`)
- Python 3.14.2, leaf-complexity 0.7.0, SymPy 1.14.0

## Reproducing

From the package root (`packages/leaf-complexity`):

```bash
python benchmark/bench_python.py
wolframscript -file benchmark/bench_wolfram.wl
```

Both scripts print their results as JSON, including interpreter/kernel versions.
