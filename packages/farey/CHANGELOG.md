# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-07-15

Aligns `farey_range` with the **improved** Wolfram Language
[`FareyRange`](https://resources.wolframcloud.com/FunctionRepository/resources/FareyRange/)
resource function used by the `DanieleGregori/GeneralizedRange` paclet, and
matches the port already embedded in `algebraic-range`.

### Changed

- **Exact semantics.** `farey_range(x, y, n)` now returns every rational with
  denominator ≤ `n` inside `[x, y]`, computed exactly with the Farey next-term
  recurrence (`Fraction` throughout). The previous float-based interval scaling
  gave different (and inexact) results for non-integer spans — e.g.
  `farey_range(0, 1.5, 2)` is now `[0, 1/2, 1, 3/2]` instead of
  `[0.0, 0.75, 1.5]`.
- **Directional, like `Range[a, b, step]`.** The range starts at the first
  bound and moves in the direction of the step: a positive step ascends
  (non-empty only when `start < end`), a negative step descends (non-empty only
  when `start > end`). Bounds that run against the step now yield `[]` rather
  than a reversed list. A degenerate `start == end` yields `[x]` when `x` lies
  on the order-`n` Farey grid (`den(x) <= n`), else `[]` — the canonical
  convention shared with `algebraic-range`'s internal port. (The published
  `FareyRange` resource function is internally inconsistent here, returning
  `{x}` only for interior on-grid points; this port follows the consistent
  rule.)
- **Step magnitude is the order only.** `n` and `1/n` are equivalent (order
  `n`); only the sign sets direction.
- Results are always `Fraction` objects (previously `float` for non-integer
  spans).
- Default `step` is now `1` (order 1) instead of `None`.

### Added

- Extension of the resource function to **negative** and **unit-fraction**
  steps (the underlying resource function accepts only positive integers).
- `benchmark/` folder (`bench_python.py`, `bench_wolfram.wl`, `BENCHMARK.md`)
  comparing the port against the WL resource function.
- Test suite rewritten with values verified against the WL `FareyRange`
  resource function, covering ascending/descending direction, empty and
  one-sided ranges, and bounds greater than 1.
