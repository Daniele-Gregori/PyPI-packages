# find-closed-form 0.5.0 — reduction against the WL kernels

Report of the differences between the Python port (`find-closed-form`
0.5.0) and the Wolfram Language `FindClosedForm` sources: the published
resource-function kernel **1.0.0**
([FindClosedForm-1-0-0-kernel.wl](FindClosedForm-1-0-0-kernel.wl)) and the
four bug-fixed working kernels in
[Version 1.1.0/](Version%201.1.0/)
(`FindClosedFormV1.0.0.1.nb` … `FindClosedFormV1.0.0.4.nb`).

Date of the reduction: 2026-07-16.

## Method

The kernel code was extracted from the notebooks with

```wl
NotebookImport[nb, "Input" -> "HeldExpression"]
```

(`"InputText"` fails without a front end, since the notebooks contain box
notation) and compared statement-by-statement against the published 1.0.0
kernel. The raw `FindClosedForm-1-0-0-kernel.wl` itself contains
`\!\(\*TagBox[...]\)` box fragments that `ReadList` cannot parse, so the
notebook definitions were the source of truth for the diff.

## 1. What the bug-fixed kernels change vs published 1.0.0

Every definition is verbatim identical across the four notebooks except
four items:

| # | Change | Kernel | Ported |
|---|--------|--------|--------|
| 1 | `formulaComplexity` rewrite ("V4") | iterated in .1–.3, final in 1.0.0.4 | ✅ |
| 2 | Auto complexity threshold rescale, 15 → 6.5 → 5; WolframAlpha default 50 → 20 | 6.5 in 1.0.0.1, 5 in 1.0.0.4 | ✅ (threshold; WA queries not ported) |
| 3 | Default function list gains `Erfc` and `ExpIntegralEi` | 1.0.0.2 | ✅ |
| 4 | `OutputArguments` multi-function attribution fix | 1.0.0.2 | ❌ (no `output_arguments` yet) |

### 1.1 `formulaComplexity` V4

Final WL definition (kernel 1.0.0.4), now matched by
`formula_complexity` in `core.py`:

- Per-integer weight: `(5·IntegerLength + DigitSum + Ω + √i) / 8`,
  where `Ω(i)` counts prime factors with multiplicity (`Ω(1) = 1`).
  Previously `(5·IntegerLength + DigitSum + √|i|) / 3`.
- Non-positive integers `j` count as `−j + 1` (previously `|j|`).
- A root of degree `m/n` duplicates its base `|m| + |n|` times
  (previously `|m·n|`).
- Calibration check: `fc(1) = 1.0` in both WL and Python.

The Python port additionally fixed integer powers, which wrongly
duplicated the base `|exp|` times; they now count the base once plus the
exponent itself.

### 1.2 Auto complexity threshold

Per search round: `½ (1 + √arity) (C + fc)` with the constant `C`
dropping 15 → 6.5 (kernel 1.0.0.1) → 5 (kernel 1.0.0.4), paired with the
new complexity weights. Two further parity fixes on the Python side:

- `fc(range)` is the **maximum** complexity over the range elements
  (WL `formulaComplexity[list]` semantics), not the complexity of the
  largest element. This is what cured the formerly xfailed
  `test_polygamma`.
- Custom `search_arguments` now use the same formula over the given
  arguments (previously a flat 50).

The WolframAlpha default threshold 50 → 20 is moot until WA queries are
ported (see §3).

## 2. Other parity items ported in 0.5.0

Divergences of the Python core from the published 1.0.0 kernel found
during the reduction and fixed:

- **Algebraic lookup table completed to the full WL `absLookupBase`**:
  integer ranges `10^4–10^5` (step 1), `10^5–10^6` (step 10),
  `10^6–10^8` (step 1000), `10^8–10^9` (step `10^5`), plus
  `farey_range(0, 100, 100)` — ~801k entries, packed in `array`s,
  lazily built in ~2 s (the WL kernel comment claims the same build
  time).
- **`significant_digits` auto-detection**: exact input (`int`,
  `Fraction`, sympy `Rational`) means 16 digits (WL `autoDigits`).
- **`searchRound` early exits**: the `Times`/`Plus` sub-searches are
  skipped once a round has enough results, and the search stops
  mid-round when `max_results` is reached.
- **Dict `search_arguments`** (WL `<|#1 -> list1, #2 -> list2|>`):
  per-slot lists now work (previously dicts fell through to a broken
  product that iterated the keys).
- **`monitor_search`** (WL `"MonitorSearch"`): prints each result as it
  is found.
- Two formerly `xfail`-marked tests (`test_polygamma`,
  `test_gamma_ratio`) pass after the threshold fixes; markers removed.

## 2b. Options coverage

The WL `Options[FindClosedForm]` (15 entries) against the Python
signature:

| WL option (default) | Python | Status |
|---|---|---|
| `"SignificantDigits" -> Automatic` | `significant_digits=None` | ✅ incl. `autoDigits` (16 for exact input) |
| `"FormulaComplexity" -> Automatic` | `formula_complexity_threshold=None` | ✅ incl. auto per-round formula |
| `"AlgebraicFactor" -> True` | `algebraic_factor=True` | ✅ |
| `"AlgebraicAdd" -> True` | `algebraic_add=True` | ✅ |
| `"RationalSolutions" -> False` | `rational_solutions=False` | ✅ |
| `"SearchArguments" -> Automatic` | `search_arguments=None` | ✅ list + per-slot dict |
| `"SearchRange" -> "Farey"` | `search_range="Farey"` | ✅ Farey/Plain/Integer/callable; Python adds `"Algebraic"`/`"Transcendental"` |
| `"MaxSearchRounds" -> 50` | `max_search_rounds=50` | ✅ |
| `"SearchTimeLimit" -> 3600` | `search_time_limit=3600` | ✅ (SIGALRM hard limit) |
| `"MonitorSearch" -> False` | `monitor_search=False` | ✅ |
| `"WolframAlphaQueries" -> 3` | — | ❌ needs a WA AppID; offline package |
| `"SearchQueries" -> Automatic` | — | ❌ only meaningful with WA queries (results left to the search after WA) |
| `"RootApproximantMethod" -> Automatic` | — | ❌ deferred (`"BuiltIn"` PSLQ / custom tables; Python always uses the lookup table) |
| `"OutputArguments" -> False` | — | ❌ deferred |
| `"SearchComplex" -> False` | — | ❌ deferred (needs ComplexRange) |

Python-only additions: `search_range_options` (forwarded to the range
generators), `search_range_fn`, and `max_results` as the third
positional argument (the WL *n*).

## 3. Not ported — deferred to a next version

Difficulties found during the reduction, listed in decreasing order of
expected impact:

| Feature | Why deferred |
|---------|--------------|
| `functionChamber` argument chambers | WL restricts the argument range per known function (e.g. `Sin[π x]` to `[0, ½]`) during `Times`/`Plus` — a speed optimization requiring symbolic sub-function extraction (`functionCatch`). |
| `simplifyRational` | Continued-fraction truncation of the rational factors/addends of a result when precision allows. |
| WolframAlpha queries (`"WolframAlphaQueries"`, `"SearchQueries"`) | Needs a WolframAlpha AppID; out of scope for an offline package. |
| `OutputArguments` | Returning the matched arguments, including the kernel-1.0.0.2 attribution fix. |
| `SearchComplex` | Needs a `ComplexRange` port and the complex phase lookup. |
| `RootApproximantMethod` | The `"BuiltIn"` (PSLQ-style `RootApproximant`) method and custom lookup tables. |
| `Glaisher`, `Khinchin`, `BarnesG`, `DedekindEta`, `ModularLambda` | No sympy equivalents (mpmath-only numerics, no symbolic representation). |

One curiosity from the notebooks: kernel 1.0.0.4 already experiments
with `"SearchRange" -> AlgebraicRange`, so the 0.5.0
Algebraic/Transcendental range integration is ahead of the WL side
there.

## 4. Intentional Python-side deviations

Not WL bugs and not Python bugs — deliberate differences:

- **`search_time_limit` is a hard `TimeConstrained`** via
  `signal.setitimer`/`SIGALRM` on Unix main threads (interrupts even a
  single long symbolic evaluation); cooperative clock checks elsewhere.
- **Unknown `search_range` values raise `FindClosedFormError`** instead
  of silently falling back to `"Farey"`.
- **`float` values in `search_arguments`** are exactified via
  `Fraction(x).limit_denominator(10**12)` (so `0.1` means `1/10`), not
  the exact binary expansion.
- **Oversized multi-argument ranges** keep the smallest-magnitude
  arguments per slot when truncating (previously the most negative).

## 5. Related differences found in the range packages

Side findings from verifying the algebraic-/transcendental-range README
examples against WL:

- **Power canonicalization** (transcendental-range): sympy left
  `(2·√2)^(−2·√2)` where WL gives `2^(−3·√2)`. Fixed with a `_wl_power`
  helper (`as_powers_dict()` decomposition) at the two `make_expr`
  sites in `core.py`.
- **`formula_complexity_threshold=5` element count**: the README
  example `transcendental_range(0, 8, 1/2, generators_domain=
  'algebraics', formula_complexity_threshold=5)` yields **14** elements
  in Python against **16** in WL — the algebraic-range
  `formula_complexity` still predates the V4 rewrite (deferred, see
  the algebraic-range 0.9 follow-ups), so borderline scores differ.
  README updated to the actual Python count.

## 6. Verification status

- find-closed-form: **61/61 tests pass** (45 main + 14 search-ranges +
  the 2 formerly xfailed).
- transcendental-range: **103/103 tests pass** after the `_wl_power`
  fix (12.5 min run).
- README examples: **34/34 pass** across algebraic-range,
  transcendental-range and the find-closed-form cross-package examples
  (`exp(sqrt(2))`, `zeta(exp(1/2))`, Gelfond–Schneider `2**sqrt(2)`).
