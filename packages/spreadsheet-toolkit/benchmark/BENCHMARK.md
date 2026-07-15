# Benchmark: Python `spreadsheet_trace` vs Wolfram Language `SpreadsheetTrace`

The Python `spreadsheet_trace` is a faithful port of the Wolfram Language resource function [SpreadsheetTrace](https://resources.wolframcloud.com/FunctionRepository/resources/SpreadsheetTrace): both produce identical traces on the whole test corpus (see `tests/test_spreadsheet_trace.py`, which transcribes the full Wolfram `VerificationTest` suite). This benchmark compares their wall-clock speed on the same machine, over the example workbooks in `tests/data`.

## Results

| Workbook | Cell | Case | WL (ms) | Python (ms) | Speed-up |
|---|---|---|---:|---:|---:|
| example_01 | `D1` | two-level chain | 4.28 | 0.104 | 41× |
| example_01 | `E1` | three-level chain | 6.19 | 0.113 | 55× |
| example_02 | `C5` | deep trace, duplicate branches | 41.6 | 0.335 | 124× |
| example_02 | `D1` | deepest converging trace (5 levels) | 52.5 | 0.410 | 128× |
| example_03 | `B18` | nested `IF` over two range aggregations | 44.0 | 0.377 | 117× |
| example_03 | `F9` | `SUM` over absolute-reference formulas | 27.9 | 0.265 | 105× |
| example_04 | `Summary!B3` | cross-sheet reference + `SUM` range | 7.65 | 0.163 | 47× |
| example_05 | `Orders!D2` | `VLOOKUP` with column range `Products!A:C` | 16.5 | 0.232 | 71× |
| example_05 | `Dashboard!B3` | cross-sheet chain over column ranges | 184.9 | 1.93 | 96× |
| example_07 | `Budget!B4` | heaviest case: repeated column-range expansions | 2351.3 | 19.8 | 119× |
| example_08 | `Catalog!I2` | `INDEX`/`MATCH` over two cross-sheet column ranges | 23.0 | 0.251 | 92× |

**The Python port is roughly 40–130× faster (geometric mean ≈ 85×).** The gap is smallest on shallow traces (where per-call overhead dominates) and largest on parsing-heavy traces with ranges and duplicate branches, where Wolfram's `StringCases`/`StringSplit` pattern machinery is replaced by compiled regular expressions.

For reference, the `Budget!B4` timing recorded in the original resource-function notebook (`CPUTimeUsed` ≈ 2.30 s) matches the 2.35 s measured here, so the Wolfram numbers are representative.

## Why the difference

The port keeps the algorithm (and even the copy-on-append list semantics) identical, so the gap is all constant factors, measured as follows:

- **Evaluator overhead per cell.** A minimal one-cell leaf trace costs ~700 µs in WL vs ~26 µs in Python: each cell visit pays for symbolic term-rewriting (pattern-matched definition dispatch, `Block`, `AppendTo` expression rebuilding) plus `ResourceFunction` wrapper calls — a single `SpreadsheetIndexToPosition` call is ~84 µs, and it runs for every visited cell. The Python equivalents are sub-microsecond C-level operations.
- **Interpreted string patterns vs compiled regexes.** The Wolfram parser rebuilds its `StringExpression` pattern lists on every formula parse (~17 µs before any matching) and `StringCases` interprets them per token, while the Python port scans with a cached compiled regex running in the C regex engine (~0.6 µs vs ~4.3 µs for the same scan with a quarter of the alternatives).
- **Column-range expansion multiplies both.** Expanding `Products!A:C` calls `PositionToSpreadsheetIndex` (~62 µs) once per generated reference; `Budget!B4` re-expands ranges throughout its recursion, which is why it is the slowest case in WL.

This is consistent with the pattern in the table: the smallest speed-ups occur where per-call overhead dominates (shallow chains), the largest where parsing and range expansion dominate.

## Methodology

- Same machine, single-threaded, warm caches: for each `(file, cell)` case one warm-up call is made first, which also fills the import cache (`ImportOnce` on the Wolfram side, `import_all` on the Python side). Timings therefore measure **tracing only**, not file import.
- Wolfram: trimmed-mean wall-clock time from `RepeatedTiming[..., 5]` (~5 s of repetitions per case).
- Python: mean wall-clock time over up to 200 repetitions within ~5 s per case (`time.perf_counter`); means and minima differed by <10% in all cases.

## Environment

- Intel Core i9-9980HK @ 2.40 GHz (x86_64), macOS
- Wolfram Language 15.0.0, `SpreadsheetTrace` 1.0.0 (Function Repository)
- Python 3.14.2, spreadsheet-toolkit 0.7.0, openpyxl 3.1.5

## Reproducing

From the package root (`packages/spreadsheet-toolkit`):

```bash
python benchmark/bench_python.py
wolframscript -file benchmark/bench_wolfram.wl
```

Both scripts print their results as JSON, including interpreter/kernel versions.
