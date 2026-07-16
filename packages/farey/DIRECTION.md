# Direction semantics

Like the built-in `Range[a, b, step]`, a Farey range **starts at the first
bound and moves in the direction of the step**. A positive step ascends
(non-empty only when `start < end`); a negative step descends (non-empty only
when `start > end`). Bounds that run against the step — or a degenerate
`start == end` — give an empty list:

| `start, end` | `step = 3` (asc) | `step = -3` (desc) |
|--------------|------------------|--------------------|
| `0, 1` | `{0, 1/3, 1/2, 2/3, 1}` | `{}` |
| `1, 0` | `{}` | `{1, 2/3, 1/2, 1/3, 0}` |
| `2, 4` | `{2, 7/3, 5/2, 8/3, 3, 10/3, 7/2, 11/3, 4}` | `{}` |
| `4, 2` | `{}` | `{4, 11/3, 7/2, 10/3, 3, 8/3, 5/2, 7/3, 2}` |
| `0, 1/4` | `{0}` *(one-sided)* | `{}` |
| `1/4, 0` | `{}` | `{0}` *(one-sided desc)* |
| `5, 5` | `{5}` *(degenerate, on-grid)* | `{5}` |
| `2/5, 2/5` | `{}` *(degenerate, off-grid: den 5 > 3)* | `{}` |

`step = 3` and `step = 1/3` (or `-3` and `-1/3`) always produce the same result:

```python
farey_range(2, 4, 3) == farey_range(2, 4, Fraction(1, 3))   # True
farey_range(4, 2, -3) == farey_range(4, 2, Fraction(-1, 3))  # True
```
