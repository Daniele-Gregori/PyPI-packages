# Benchmark: Python `transform-tabular` vs Wolfram Language `TransformTabular`, `TransformColumns`, `TransformRows`

The Python functions `transform_tabular`, `transform_columns` and `transform_rows` are ports of the Wolfram Language resource functions [TransformTabular](https://resources.wolframcloud.com/FunctionRepository/resources/TransformTabular/) and TransformRows and of the built-in function [TransformColumns](https://reference.wolfram.com/language/ref/TransformColumns.html). The test suites transcribe the tests and documented examples of the originals, with expected results computed in Wolfram Language. This benchmark compares the wall-clock speed of the two sides on the same machine, over identical tables; in every case below the two implementations returned the same table (same total of all cells, same leading cells).

## Results

The tables hold integers in ten columns `a`–`j`: *small* has 10^5 rows, *large* has 10^6 rows. `rows` is a list of scattered row positions, 10% of the table.

Each case is run in Python in two styles. **Rowwise** is the literal translation of the Wolfram code, with a Python function called on every element or row. **Columnwise** writes the same transformation with whole-column operations inside a `ColumnwiseThread`, wherever that is possible.

| Case | Wolfram code | Table | WL (ms) | Python rowwise (ms) | Python columnwise (ms) |
|---|---|---|---:|---:|---:|
| tabular-elementwise | `TransformTabular[tab, # + 1 &]` | small | 2.18 | 412 | – |
| tabular-columnwise-value | `TransformTabular[tab, # - ColumnwiseValue[Min[#]] &]` | small | 2.38 | 630 | – |
| tabular-columnwise-thread | `TransformTabular[tab, ColumnwiseThread[Accumulate[#]] &]` | small | 2.26 | 658 | – |
| columns-arithmetic | `TransformColumns[tab, "a" -> Function[2 #a]]` | large | 0.64 | 692 | 28.0 |
| columns-new-column | `TransformColumns[tab, "k" -> Function[#a + #b]]` | large | 0.99 | 779 | 29.9 |
| columns-columnwise-value | `TransformColumns[tab, "a" -> Function[#a - ColumnwiseValue[Median[#b]]]]` | large | 4.22 | 897 | 45.6 |
| columns-strings | `TransformColumns[tab, "s" -> Function[ToString[#a] <> "-" <> ToString[#b]]]` | small | 22.4 | 80.3 | 89.9 |
| rows-list | `TransformRows[tab, rows, "a" -> Function[2 #a]]` | large | 291 | **174** | **112** |
| rows-list-unordered | the same with `"KeepOrder" -> False` | large | 60.3 | 222 | 151 |
| rows-criterion | `TransformRows[tab, EvenQ[#c] &, "a" -> Function[2 #a + 2 #b]]` | small | 34.7 | 110 | **20.2** |
| rows-all-cells | `TransformRows[tab, rows, #^2 &]` | small | 33.9 | 119 | – |

**For arithmetic on whole columns the Wolfram kernel is far faster: about 200–1000× against the rowwise Python style, and 10–45× against the columnwise one.** On transformations that are not plain arithmetic (strings) the gap narrows to about 4×. **`transform_rows` is the exception: with the default row order kept, Python is 1.7–2.6× faster** than `TransformRows`, and with a criterion written as a boolean column it is 1.7× faster.

## Why the difference

- **Wolfram vectorizes row functions; Python calls them.** `Tabular` is a typed column store, and `TransformColumns` turns a function like `2 #a &` into one operation on a packed column: 10^6 rows take about 1 ms. In Python the function is an opaque object that has to be called once per element or per row: about 0.4 µs per element in `transform_tabular` (412 ms for 10^6 cells) and 0.7 µs per row in `transform_columns` (692 ms for 10^6 rows). This cost is that of calling any Python function, and it is why the rowwise columns of the table are flat at 0.4–0.9 s per million calls.
- **The columnwise style removes the calls, and what remains is a copy.** With `ColumnwiseThread(lambda t: 2 * t["a"])` the same case takes 28 ms, of which 14–20 ms is spent copying the table: every function of this package returns an independent `DataFrame` (80 MB here), while a `Tabular` shares its unchanged columns with the result.
- **Strings are not vectorized on either side.** In `columns-strings` Wolfram evaluates the function row by row as well (0.22 µs per row), and the two sides come within a factor of 4.
- **`TransformRows` pays for restoring the row order.** It transforms the table, joins the untouched rows with the transformed ones, and permutes the result back with `FindPermutation` and `Permute`: this step alone takes 280 ms of the 291 ms of `rows-list` (as noted in the *Possible Issues* of its documentation, which is why it has the `"KeepOrder"` option). The Python port computes the new cells of the selected rows only and puts them in place by position (about 25 ms per million rows), so keeping the order costs nothing: `keep_order=False` is actually somewhat slower in Python, since it reorders the whole table.

## Methodology

- Same machine, single-threaded. The tables are built once, outside the timed region, from the same formula on both sides (`Mod[7919 i + 104729 j, 1000003]`), and one warm-up call precedes timing.
- Wolfram: trimmed-mean wall-clock time from `RepeatedTiming[..., 2]` per case.
- Python: mean wall-clock time over up to 200 repetitions within ~2 s per case (`time.perf_counter`).
- For every case both scripts report the total of all the cells of the result and its first three cells in column `a`; these coincide in all cases.

## Environment

- Intel Core i9-9980HK @ 2.40 GHz (x86_64), macOS
- Wolfram Language 15.0.1, `TransformTabular` 1.0.1 and `TransformRows` 0.6.1 (loaded from `wolfram/*-kernel.wl`), built-in `TransformColumns`
- Python 3.14.2, transform-tabular 0.9.0, pandas 3.0.0, NumPy 2.4.2

## Reproducing

From the package root (`packages/transform-tabular`):

```bash
python benchmark/bench_python.py
wolframscript -file benchmark/bench_wolfram.wl
```

Both scripts print their results as JSON, including interpreter/kernel versions.
