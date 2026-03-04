# Benchmark

Performance of `transcendental_range(lo, hi, method=...)` across all 27 methods and three range sizes.
Measured on Darwin (x86_64), Python 3.14.2.

## Results

| Method | (-10, 10) | (-25, 25) | (-100, 100) |
|--------|-------------|-------------|-------------|
| exp | 208 in 0.07s | 1276 in 0.05s | 20108 in 0.81s |
| log | 82 in 0.01s | 430 in 0.02s | 5064 in 0.19s |
| power | 0 in 0.00s | 0 in 0.00s | 0 in 0.00s |
| sin | 200 in 0.69s | 1250 in 4.48s | timeout |
| cos | 200 in 0.66s | 1250 in 4.31s | timeout |
| tan | 142 in 0.60s | 884 in 3.78s | 14304 in 59.74s |
| cot | 142 in 0.65s | 886 in 4.22s | timeout |
| sec | 120 in 2.30s | 768 in 15.42s | timeout |
| csc | 120 in 2.60s | 766 in 15.98s | timeout |
| sinh | 20 in 0.01s | 58 in 0.01s | 250 in 0.03s |
| cosh | 16 in 0.00s | 48 in 0.01s | 206 in 0.02s |
| tanh | 200 in 0.01s | 892 in 0.05s | 3592 in 0.22s |
| coth | 176 in 0.01s | 840 in 0.04s | 3498 in 0.23s |
| sech | 200 in 0.01s | 1250 in 0.05s | 20000 in 0.80s |
| csch | 200 in 0.01s | 1250 in 0.05s | 20000 in 0.75s |
| asin | 12 in 0.00s | 30 in 0.00s | 126 in 0.01s |
| acos | 12 in 0.00s | 30 in 0.00s | 126 in 0.01s |
| atan | 144 in 0.01s | 854 in 0.03s | 13100 in 0.53s |
| acot | 200 in 0.01s | 1250 in 0.05s | 20000 in 0.82s |
| asec | 206 in 0.02s | 1476 in 0.06s | 25032 in 0.90s |
| acsc | 186 in 0.01s | 1214 in 0.05s | 19862 in 0.79s |
| asinh | 84 in 0.01s | 420 in 0.02s | 4762 in 0.22s |
| acosh | 66 in 0.01s | 360 in 0.01s | 4356 in 0.16s |
| atanh | 0 in 0.00s | 0 in 0.00s | 0 in 0.00s |
| acoth | 180 in 0.01s | 1200 in 0.04s | 19800 in 0.76s |
| asech | 0 in 0.00s | 0 in 0.00s | 0 in 0.00s |
| acsch | 200 in 0.02s | 1250 in 0.05s | 20010 in 0.79s |

## Notes

- **Efficient path** (exp, log, power, all hyp, all inv-trig, all inv-hyp): uses monotonic outer algorithm with float pre-screening and deferred sympy expression creation. All complete in under 1s at (-100, 100).
- **Naive path** (sin, cos, tan, cot, sec, csc): uses brute-force outer product with exact sympy comparisons. These are slow at larger ranges due to the non-monotonic, periodic nature of trig functions.
- **Zero results** for power, atanh, asech: no algebraic irrational generators exist in the given rational ranges (power requires irrational exponents; atanh and asech have restricted domains).
- Timeout threshold: 60s.
