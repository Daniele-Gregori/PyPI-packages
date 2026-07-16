# Benchmark: Python `algebraic_range` vs Wolfram Language `AlgebraicRange`

The Python `algebraic_range` 0.9 is a faithful port of the Wolfram Language
resource function [AlgebraicRange](https://resources.wolframcloud.com/FunctionRepository/resources/AlgebraicRange/)
version 2.0: both produce identical results on the whole WL verification
suite (see `tests/test_algebraic_range.py`, which transcribes it). This
benchmark compares wall-clock speed on the same machine, over the five
reference cases of the WL suite (its group-10 performance tests).

## Results

| Case | Elements | WL 2.0 (s) | Python 0.9 (s) | Python 0.8.2 (s) |
|---|---:|---:|---:|---:|
| `AlgebraicRange[200]` | 40 000 | 0.29 | 1.04 | 43.8 |
| `AlgebraicRange[200, -200, -1]` | 80 001 | 1.10 | 2.48 | > 300 (killed) |
| `AlgebraicRange[-100, 100, 1/2]` | 43 547 | 11.30 | 6.54 | not measurable |
| `AlgebraicRange[60, -60, -1/3]` | 24 329 | 3.44 | 2.80 | not measurable |
| `AlgebraicRange[1 ± 10^-13, 10^-17, wp → 30]` | 20 001 | 0.27 | 1.03 | unsupported |

Element counts agree exactly between WL 2.0 and Python 0.9 on all five cases.

**Python 0.9 runs the stepped ranges (the algorithmically hard cases) up to
~1.7× faster than WL 2.0, and the elementary/high-precision ranges within
~4× of it.** Against the previous Python release the rewrite is a ~40×–100×
speed-up where 0.8.2 could finish at all: 0.8.2 used a literal outer-product
construction with per-element numerical simplification, whose quadratic
blow-up made the reference cases essentially unreachable.

## Why the difference

The two implementations are bottlenecked in different places, which is why
the speed ratio inverts between the elementary and the stepped cases.

- **Elementary/high-precision cases: bulk object construction.** Building
  the 40 000 canonical radicals of `AlgebraicRange[200]` accounts for
  essentially the whole time on both sides — 1.05 s of sympy `Pow`/`Mul`
  construction (~26 µs per object) vs 0.28 s for WL's single vectorized
  `Range[1, 40000]^(1/2)` kernel operation. WL's C kernel mass-produces
  symbolic objects ~4× faster than the Python object system.
- **Stepped cases: the per-element factor window.** WL 2.0 calls
  `Nearest[fcrg -> "Index", x]` twice per elementary root, and `Nearest`
  re-processes the factor list on every call — measured ~169 µs per lookup
  on the case-3 grid, i.e. most of WL's 11.3 s across its ~10⁵ lookups.
  The port makes the identical selection by binary search over a
  precomputed float key array (~0.14 µs per lookup, ~1000× cheaper), so
  Python wins despite its slower object construction.
- **Windowed outer construction.** Like WL 2.0, the port never materializes
  the full outer product: for each elementary root it selects the admissible
  rational multipliers within its exact window, so the work is linear in
  the output size on both sides — only the constants above differ.
- **Exact arithmetic on fast types.** Rational grids are generated with
  Python `Fraction` arithmetic and converted to canonical sympy radicals
  only at the boundary, with square-free decompositions cached via a
  smallest-prime-factor sieve.
- **Numeric keys carried alongside expressions.** Sorting, deduplication and
  window decisions use floating-point keys seeded at construction time
  (exact comparisons resolve only genuine near-ties), the analogue of WL's
  machine-precision `N` in `cleanSort`.

### `Nearest` and its Python analogue

In `outerRange`, WL's `Nearest[fcrg -> "Index", x]` answers a simple
question — where does `x` fall in a *sorted* one-dimensional grid? — but
re-processes the grid on every call (~169 µs here). The Python analogue is
the standard-library `bisect`: a binary search over the factor keys,
computed once per call of the outer construction:

```python
first = bisect_left(fc_keys, flo)      # ≈ Nearest[fcrg -> "Index", flo]
last  = bisect_right(fc_keys, fhi) - 1 #   + the While adjustment loops
```

at ~0.14 µs per lookup. The same fix is available on the WL side: build a
`NearestFunction` once per `outerRange` call (`nf = Nearest[fcrg -> "Index"]`),
or binary-search the already-sorted grid directly — that would remove most
of WL's stepped-case cost.

## Methodology

- Same machine, single-threaded, warm caches: one warm-up call per case,
  then repetitions within a ~20 s budget per case.
- Wolfram: trimmed-mean wall-clock time from `RepeatedTiming[..., 20]`.
- Python: mean wall-clock time over the repetitions (`time.perf_counter`);
  means and minima differed by < 10 % in all cases.
- Python 0.8.2 was measured with a 300 s cap per case; cases 3 and 4 were
  skipped after case 2 exceeded it (their construction is strictly heavier
  for the 0.8.2 algorithm), and case 5 needs the `working_precision` option
  introduced in 0.9.

## Environment

- Intel Core i9-9980HK @ 2.40 GHz (x86_64), macOS
- Wolfram Language 15.0.0, `AlgebraicRange` 2.0.0 (WolframMark 3.783)
- Python 3.14.2, sympy 1.14.0, algebraic-range 0.9.0

## Reproducing

From the package root (`packages/algebraic-range`):

```bash
python benchmark/bench_python.py
wolframscript -file benchmark/bench_wolfram.wl
```

Both scripts print their results as JSON, including interpreter/kernel
versions.
