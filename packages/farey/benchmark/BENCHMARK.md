# FareyRange — Python vs. Wolfram Language benchmark

This benchmark compares the exact `farey.farey_range` port against the original
Wolfram Language [`FareyRange`](https://resources.wolframcloud.com/FunctionRepository/resources/FareyRange/)
resource function it is ported from.

Both compute the **same set**: every rational with denominator ≤ *n* inside the
interval. The outputs are identical and the result lengths match case-for-case,
so the port is faithful. The `FareyRange` resource function is a pure
deterministic function — repeated calls return exactly the same rationals — so
only the timings vary between runs.

## Results

Timings are the mean wall-clock over 5 (WL, `RepeatedTiming`) / thousands
(Python, 5 s budget) repetitions after a warm-up call, on the same machine.

| `start, end, step` | length | WL ms | Python ms | speed-up (WL / Py) |
|--------------------|-------:|------:|----------:|-------------------:|
| `0, 1, 3`      |    5 | 0.080 | 0.018 | **4.5×** (Python faster) |
| `0, 10, 5`     |  101 | 0.115 | 0.142 | 0.81× |
| `-20, 20, 6`   |  481 | 0.207 | 0.624 | 0.33× |
| `0, 30, 7`     |  541 | 0.219 | 0.647 | 0.34× |
| `0, 50, 4`     |  301 | 0.164 | 0.376 | 0.44× |
| `0, 100, 3`    |  401 | 0.211 | 0.557 | 0.38× |
| `0, 200, 5`    | 2001 | 0.551 | 2.737 | 0.20× |
| `0, 1000, 2`   | 2001 | 0.858 | 2.866 | 0.30× |

Unlike the string-processing ports in this repo (where compiled Python `re`
beats interpreted WL `StringCases` by ~40–130×), **here the Wolfram Language is
faster** — roughly 3× on the larger ranges.

## Why the difference

`FareyRange` is nothing but exact rational arithmetic driven by the Farey
next-term recurrence. There is no parsing or pattern-matching hot-path for a
compiled Python library to exploit; the cost is dominated by building and
comparing rationals.

- **WL rationals are compiled.** Wolfram Language stores and manipulates exact
  rationals in kernel C code, so each mediant step (a couple of big-integer
  multiplies plus a GCD reduction) is a compiled operation.
- **Python `Fraction` is interpreted.** Every `Fraction(p, q)` runs Python-level
  `__new__`, normalisation and `math.gcd` bookkeeping; each comparison and add
  dispatches through Python. On a 2001-element range that per-element overhead
  adds up to the ~3× gap.
- **Python wins only when the output is tiny** (`0, 1, 3`), where WL's small
  fixed per-call cost outweighs five rationals of Python arithmetic.

The takeaway is the opposite of the string ports: the value of the Python port
here is **portability and dependency-free exactness** (`fractions.Fraction`,
no Wolfram kernel required), not raw speed.

## Methodology

- **Python** (`bench_python.py`): `time.perf_counter`, one warm-up call, then the
  mean over up to 2000 repetitions within a 5 s budget per case.
- **Wolfram** (`bench_wolfram.wl`): `RepeatedTiming[FR[x, y, z];, 5]` after a
  warm-up call, with `FR` bound to the resource function so lookup overhead is
  excluded from the measured region.
- Only the function call is timed; no I/O is inside the timed region.

## Environment

- Python 3.14, `farey` 0.7.0
- Wolfram Language 15.0
- Apple / Darwin, x86_64

## Reproduce

```bash
# from packages/farey
python benchmark/bench_python.py > /tmp/farey_py.json
wolframscript -file benchmark/bench_wolfram.wl > /tmp/farey_wl.json
```
