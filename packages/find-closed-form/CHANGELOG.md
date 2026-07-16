# Changelog

All notable changes to the `find-closed-form` package.

## [0.5.0] — 2026-07-16

Closed-form search over algebraic and transcendental argument ranges,
integrating the sibling packages
[algebraic-range](https://pypi.org/project/algebraic-range/) and
[transcendental-range](https://pypi.org/project/transcendental-range/).

### Added

- **WL positional syntax**: all resource-function forms now work
  positionally — `find_closed_form(y)`, `find_closed_form(y, n)`
  (integer second argument = number of results),
  `find_closed_form(y, f)`, `find_closed_form(y, [f1, f2, ...])` and
  `find_closed_form(y, f, n)`. The keyword forms `functions=` and
  `max_results=` are unchanged (full backward compatibility);
  `find_closed_form(y, n1, n2)` raises `FindClosedFormError`.
- **`"Algebraic"` and `"Transcendental"` values for `search_range`**: for
  each search round the arguments span `algebraic_range(-cut, cut, 1/cut)`
  or `transcendental_range(-cut, cut, 1/cut)`, so functions are searched at
  exact roots and transcendental numbers — recovering formulae such as
  `exp(sqrt(2))` or `atan(log(2))` from their bare machine digits. The two
  generator packages are optional dependencies, installable together with
  the extra `pip install "find-closed-form[ranges]"`.
- **`search_range_options`**: a dict of keyword options forwarded to the
  range generator, e.g. `{"root_order": 3}` (algebraic-range) or
  `{"method": "log", "multiplicity": 2}` (transcendental-range).
- **Callable `search_range`** (WL parity: `"SearchRange" -> Function`):
  a function of the search round, equivalent to `search_range_fn`.
- **Identity among the default functions** when searching the
  `"Algebraic"` or `"Transcendental"` ranges with `functions=None`: the
  range elements are closed forms themselves, so they are also matched
  directly (up to algebraic factors and addends) — e.g. the
  Gelfond–Schneider constant `2**sqrt(2)` among the `'power'` elements.
- **Ground-truth round-trip tests** (`tests/test_search_ranges.py`):
  formulae of functions at algebraic or transcendental arguments are
  recovered from their float values alone, each search bounded by
  a `TimeConstrained` harness.

### WL parity (published 1.0.0 + bug-fixed kernels 1.0.0.1–1.0.0.4)

The Python core was reduced against the published WL resource-function
kernel 1.0.0 and the four bug-fixed working kernels
(`wolfram/Version 1.1.0/FindClosedFormV1.0.0.*.nb`). The bug fixes found
there and ported here:

- **`formula_complexity` follows kernel 1.0.0.4**: per-integer weight is
  now `(5*digits + digit_sum + Ω + sqrt(i))/8` (`Ω` = prime factors with
  multiplicity, `Ω(1) = 1`), non-positive integers `j` count as `-j + 1`,
  and a root of degree `m/n` duplicates its base `|m| + |n|` times
  (previously `(5*digits + digit_sum + sqrt(|i|))/3` and `|m*n|`).
  Integer powers now count the base once plus the exponent (previously
  the base was duplicated `|exp|` times).
- **Auto complexity threshold rescaled** (kernel 1.0.0.4): per round
  `(1 + sqrt(arity))/2 * (5 + fc(range))` — the constant dropped from 15
  to 5, paired with the new complexity weights — and `fc(range)` is now
  the *maximum* complexity over the range elements (WL
  `formulaComplexity[list]`), not the complexity of the largest element.
  With custom `search_arguments` the same formula applies over the given
  arguments (previously a flat 50).
- **Default function list** gains `erfc` and `Ei` (kernel 1.0.0.2 adds
  `Erfc` and `ExpIntegralEi`).
- **Algebraic lookup table completed to the WL `absLookupBase`**: integer
  ranges `10^4–10^5` (step 1), `10^5–10^6` (step 10), `10^6–10^8`
  (step 1000), `10^8–10^9` (step `10^5`) and `farey_range(0, 100, 100)`
  are now included (~800k entries, packed `array` storage, built lazily
  in ~2 s like the WL kernel).
- **`significant_digits` auto-detection**: exact input (`int`,
  `Fraction`, sympy `Rational`) now means 16 digits (WL `autoDigits`).
- **Sub-search early exit** (WL `searchRound`): the `Times`/`Plus`
  operations are skipped once a round already found enough results, and
  the search stops mid-round as soon as `max_results` is reached.
- **`search_arguments` as a dict now works** (WL
  `<|#1 -> list1, #2 -> list2|>`): per-slot lists with 1-based integer
  or `"#1"` keys (previously dicts fell through to a broken product).
- **`monitor_search` option added** (WL `"MonitorSearch"`): prints each
  result as it is found.
- Two previously `xfail`-marked tests (`test_polygamma`,
  `test_gamma_ratio`) now pass thanks to the threshold fixes.

### Not ported (deferred to a next version)

- **`functionChamber` argument chambers**: WL restricts the argument
  range per known function (e.g. `Sin[π x]` to `[0, 1/2]`) during the
  `Times`/`Plus` operations — a search-speed optimization requiring
  symbolic sub-function extraction (`functionCatch`).
- **`simplifyRational`**: WL rounds/continued-fraction-truncates the
  rational factors and addends of a result when precision allows.
- **WolframAlpha queries** (`"WolframAlphaQueries"`, `"SearchQueries"`):
  needs a WolframAlpha AppID; out of scope for an offline package.
- **`OutputArguments`** (returning the matched arguments), including the
  kernel-1.0.0.2 fix attributing arguments to the producing function in
  multi-function searches.
- **`SearchComplex`** (complex-argument search): needs a `ComplexRange`
  port and the complex phase lookup.
- **`RootApproximantMethod`**: the `"BuiltIn"` (PSLQ-style
  `RootApproximant`) method and custom lookup tables.
- **`Glaisher`/`Khinchin` constants and `BarnesG`, `DedekindEta`,
  `ModularLambda` functions**: no sympy equivalents (mpmath-only
  numerics, no symbolic representation).

### Changed

- **`search_time_limit` is now a hard `TimeConstrained`** (WL kernel
  parity): on Unix main threads the limit interrupts the search through
  `signal.setitimer`/`SIGALRM` — even inside a single long symbolic
  evaluation — returning the results found up to that point; elsewhere
  (Windows, non-main threads) the previous cooperative clock checks
  between functions and rounds remain.
- **Unknown `search_range` values now raise `FindClosedFormError`**
  instead of silently falling back to `"Farey"`.
- Multi-argument searches over oversized ranges now keep the
  smallest-magnitude arguments when truncating each slot (previously the
  most negative ones).

### Fixed

- Rounds whose generated argument range is empty (e.g. a transcendental
  `log` range at round 1) no longer crash the search; they are skipped.
- `float` values in `search_arguments` are exactified via
  `Fraction(x).limit_denominator(10**12)` instead of the exact binary
  expansion (`0.1` now means `1/10`).
- Results found in a round interrupted by the time limit are
  deduplicated before output.

## [0.4.0] and earlier

See the [git history](https://github.com/Daniele-Gregori/PyPI-packages/commits/main/packages/find-closed-form)
and `releases/`.
