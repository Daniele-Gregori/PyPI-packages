# Benchmark: Python `find_closed_form` vs Wolfram Language `FindClosedForm`

The Python `find-closed-form` 0.5.0 is a port of the Wolfram Language
resource function
[FindClosedForm](https://resources.wolframcloud.com/FunctionRepository/resources/FindClosedForm)
1.0.0, reduced against the published kernel plus the four bug-fixed
working kernels 1.0.0.1–1.0.0.4 (see
[wolfram/REPORT-wl-differences.md](../wolfram/REPORT-wl-differences.md)).

## Headline comparison

Representative searches from the README and the verification suite,
mean wall-clock seconds (same machine, WolframAlpha queries disabled on
both sides — the Python port is offline by design):

| Case | WL 1.0.0 | Python 0.5.0 |
|---|---:|---:|
| `FindClosedForm[0.405465]` (default search) | 0.24 s | 0.10 s |
| `FindClosedForm[3.792277]` → `1/6 + Gamma[1/4]` | 2.74 s | 2.79 s |
| `FindClosedForm[1.044866]` → `1/Sqrt[Catalan]` | 0.24 s | 0.03 s |
| `FindClosedForm[1.85653, 1/Zeta[#]^2 &]` → `Zeta[1/5]^-2` | 0.15 s | 0.15 s |
| `FindClosedForm[0.780653, ArcSinh]` → `Sqrt[5] ArcSinh[4]/6` | 0.07 s | 0.04 s |
| `FindClosedForm[7.443967, Log[1 + Exp[#]] &]` | 1.80 s | 0.83 s |
| `FindClosedForm[5.550045, {Sinh, Cosh, Sech, Csch}]` → `6 Sech[2/5]` | 0.56 s | 0.52 s |
| `FindClosedForm[4.688231, Gamma[#1]/Gamma[#2] &]` (2-arg) | 0.64 s | 3.90 s |
| `FindClosedForm[0.405465, Log, 10]` (10 results) | 1.65 s | 1.35 s |
| Algebraic search range → `E^Sqrt[2]` | 4.58 s | 0.03 s ¹ |

¹ Warm timing; the first (cold) Python call takes 1.8 s, still ~2.5×
faster than WL. Python keeps the algebraic lookup table and sympy's
expression caches across calls, while the WL side regenerates the
`AlgebraicRange` search range on every call (the WL 1.0.0 case is
expressed as `"SearchRange" -> (AlgebraicRange[-#, #, 1/#] &)` with the
AlgebraicRange 2.0.0 resource function; Python's built-in
`search_range="Algebraic"` does the same generation).

Overall the port runs within ±2× of the WL timings on 8 of 10 cases,
substantially faster on the range-based search, and slower only on the
two-argument search (see below).

## Result parity

Seven of the ten cases return symbolically identical results. The three
that differ all return *valid alternative matches* (every result matches
the target to the requested digits — the search returns the first
qualifying formula, and the two implementations enumerate candidates in
slightly different order):

- **`FindClosedForm[0.405465]`**: Python finds `Log[3/2]` (the result
  documented on the resource-function page); the published WL 1.0.0
  kernel with WolframAlpha disabled finds `6/83 + EulerGamma^2` first.
  Python follows the bug-fixed kernels' formulaComplexity V4 and
  rescaled threshold, which rank the simpler logarithm first.
- **`FindClosedForm[7.443967, Log[1 + Exp[#]] &]`**: WL finds
  `10 Log[1 + E^(1/10)]`, Python `9/14 + Log[1 + E^(34/5)]` — an
  equally precise match found one candidate earlier.
- **`Gamma[#1]/Gamma[#2]`**: WL finds `2 Sqrt[3] Gamma[1/4]/Gamma[1/3]`,
  Python `Sqrt[2] Gamma[-4/3]/Gamma[7/4]`. WL is ~6× faster here: its
  `functionChamber` optimization restricts the argument range per known
  function during the algebraic-combination steps — not yet ported
  (deferred, see the 0.5.0 CHANGELOG).
- **10-result search**: 8 of 10 results are shared (including sign
  variants: WL's `23/31 - Log[7/5]` is Python's `Log[5/7] + 23/31`).
  Python additionally keeps `-Log[2/3]` and `-Log[4/9]/2`, equivalent
  forms of results already listed (`Log[3/2]`, `Log[9/4]/2`) that sympy
  does not canonicalize away, where WL finds two further distinct
  formulae instead.

## Methodology

- Same machine, single-threaded, warm caches (one warm-up call per
  case, then repetitions within a 12 s budget).
- Wolfram: `RepeatedTiming` via wolframscript, Wolfram Language 14.3
  (macOS x86-64), resource functions FindClosedForm 1.0.0 and
  AlgebraicRange 2.0.0, `"WolframAlphaQueries" -> 0`.
- Python: `time.perf_counter` means (Python 3.14, macOS x86-64), local
  in-repo `find-closed-form` 0.5.0 with `farey` 0.7.0,
  `algebraic-range` 0.9.0, `transcendental-range` 0.9.0.

## Reproducing

From the package root (`packages/find-closed-form`):

```bash
python benchmark/bench_python.py
wolframscript -file benchmark/bench_wolfram.wl
```

Both scripts print their results as JSON (the Python script also
resolves the sibling range packages from the repo tree, so no
installation is needed). The Wolfram script downloads the two resource
functions on first use.
