# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-07-16

Complete rewrite from scratch, closely following the Wolfram Language
resource function [TranscendentalRange](https://resources.wolframcloud.com/FunctionRepository/resources/TranscendentalRange)
version **1.1.0** (the previous 0.8.x port followed version 1.0.0).

### Added

- `multiplicity` option (WL "Multiplicity"): linear combinations of up to
  *n* generated terms — products for the `power` method — still provably
  transcendental by the Lindemann–Weierstrass and Baker theorems.
- `test` option (WL "Test", development only): naive `Outer`-based baseline
  implementation, used to validate the monotonic-outer algorithm.
- New test suite ported one-to-one from the WL 1.1.0 `VerificationTests`:
  every efficient method is checked against the naive baseline over eight
  sign/direction range configurations with randomized bounds and steps,
  plus fixed larger ranges and multiplicity-2/3 rounds. The WL
  `TimeConstrained` is translated (SIGALRM-based) and applied to every
  comparison, so pathological random draws abort at 30 s like the WL
  `$Aborted` instead of stalling the suite; the randomized rounds are two
  per method instead of the notebook's five.
- `FareyStepError` exception (WL `failureFareyStep`).
- Deterministic CI suite `tests/test_deterministic.py`
  (documentation examples with exact expected outputs, fixed
  default-vs-naive comparisons for every method): the randomized
  TimeConstrained suite is skipped on CI and run locally, and the GitHub
  workflow now runs the deterministic suite and triggers on package
  changes.
- `benchmark/` folder with paired Python/Wolfram scripts and BENCHMARK.md.

### Changed

- Method specifications ported from WL 1.1.0, fixing the `log` and `csch`
  monotonic-outer scan directions (minor bugs of WL 1.0.0 that produced
  incomplete ranges, e.g. for `log` over `[-10, 10]` with step `1/7`).
- The Farey generator range now delegates to the
  [farey](https://pypi.org/project/farey/) package (the port of the
  `FareyRange` resource function that WL calls), instead of an internal
  reimplementation. As in WL 1.1.0, a Farey step must be an integer >= 1
  or of the form `1/n` (or `-1/n`): other steps raise `FareyStepError`,
  including negative integers, which the 0.8.x port accepted.
- Both the default and the naive implementation now share the same cached
  float arithmetic and the same boundary decision (floats away from the
  bounds, exact sympy comparison in a narrow band around them), so their
  outputs agree element by element at machine precision.
- Duplicate elimination groups by 13 significant digits and, for
  multiplicity combinations, recomputes values from the canonical sympy
  form — matching the WL machine-precision `GroupBy[N]` behaviour for
  values reached through different representations, and merging pairs the
  WL splits by one machine ulp (e.g. `3 Log[8]` vs `9 Log[2]` in the
  documented `Method -> Log` example).
- Representative selection on numerically equal elements follows the WL
  minimal-argument rule (`|a|` for `b f(a)`; `|b| + |e|` for `b^e`),
  accumulated additively across multiplicity terms.
- `power` multiplicity products are combined with `powsimp`, mirroring the
  WL automatic simplification (e.g. products collapsing to an algebraic
  number are correctly discarded).
- Requires `algebraic-range >= 0.9.0` (the AlgebraicRange 2.0 port) and
  `farey >= 0.6.0`.

### Known limitations

- The `formula_complexity` heuristic is inherited from `algebraic-range`
  and will be rewritten in a future version; thresholded outputs can
  differ slightly from the WL 1.1.0 ones on transcendental expressions.
- Symbolic representatives may differ from the WL ones on numerically
  equal alternatives (the numerical content is identical).

## [0.8.2] - 2026-03-10

- Metadata and packaging fixes.

## [0.8.1] - 2026-03-04

- Readme and pyproject fixes; internal Farey range fix.

## [0.8.0] - 2026-03-04

- Initial release: port of TranscendentalRange 1.0.0 — 27 methods,
  efficient monotonic-outer path for 21 of them, Farey ranges,
  formula-complexity threshold, working precision; benchmark and test
  suite translated from the 1.0.0 notebook.
