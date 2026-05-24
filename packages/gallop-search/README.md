# gallop-search

[![Tests](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/gallop-search.yml/badge.svg)](https://github.com/Daniele-Gregori/PyPI-packages/actions/workflows/gallop-search.yml)
[![PyPI version](https://badge.fury.io/py/gallop-search.svg)](https://badge.fury.io/py/gallop-search)
[![Python](https://img.shields.io/pypi/pyversions/gallop-search.svg)](https://pypi.org/project/gallop-search/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Binary search from an arbitrary position in sorted sequences.

## Installation

```bash
pip install gallop-search
```

No dependencies — pure Python.

## Syntax

**binary_search_from**(*seq*, *value*, *start=0*) finds the index of *value* in a sorted sequence, galloping from *start*.

**binary_search_from_by**(*seq*, *value*, *\**, *key*, *start=0*) finds the index of an element whose *key* equals *value* in a key-sorted sequence.

| Parameter | Description |
|:---|:---|
| `seq` | A sorted sequence supporting `__getitem__` and `__len__` |
| `value` | The value to search for (or to match against `key(element)`) |
| `start` | Position to begin the gallop (default `0`). Negative indices count from the end |
| `key` | (*`_by` only*) Extracts the comparison value from each element |

Returns the index of the element, or `None` if not found.

## Usage

### `binary_search_from`

```python
from gallop_search import binary_search_from

binary_search_from([10, 20, 30, 40, 50], 30)
# 2

binary_search_from(list(range(1_000_000)), 500_010, start=500_000)
# 500010

binary_search_from(list(range(100)), 20, start=80)
# 20

binary_search_from(list(range(100)), 95, start=-10)
# 95

binary_search_from([10, 20, 30, 40, 50], 25)
# None
```

### `binary_search_from_by`

Search in sequences sorted by a key function. The `key` parameter extracts the comparison value from each element — similar to `sorted(..., key=...)`.

```python
from gallop_search import binary_search_from_by

events = [
    {"ts": 100, "msg": "start"},
    {"ts": 200, "msg": "running"},
    {"ts": 300, "msg": "done"},
]
binary_search_from_by(events, 200, key=lambda e: e["ts"])
# 1

points = [(0, 0.0), (1, 0.5), (2, 1.0), (3, 1.5), (4, 2.0)]
binary_search_from_by(points, 3, key=lambda p: p[0], start=1)
# 3

binary_search_from_by(events, 999, key=lambda e: e["ts"])
# None
```

## Details

The standard **binary search** algorithm is commonly used to efficiently find the index of an element of a sorted list, by splitting it initially in the middle and then in exponentially smaller and smaller intervals around the searched element.

`binary_search_from` is a variation of this standard algorithm in three respects:

1. By default the search does not start from the middle but from the **beginning** of the list.
2. The binary bisections first **increase** and then decrease if they reach elements greater than the searched one, proceeding with ordinary binary search until the desired element is found.
3. An optional `start` argument may be specified to begin the search from an **arbitrary position** index.

The search can be performed both forward and backward, as the **direction is adjusted automatically** depending on whether `seq[start]` is less than or greater than the target value.

Similar algorithms are sometimes known as "galloping search" or "exponential search", but they typically involve only the list and the searched element, with no starting position specification.

Ordinary binary search is likely to beat `binary_search_from` in most applications. However, `binary_search_from` may be more efficient when the caller has an educated guess about the target element's position. In that case the complexity is **O(log k)**, where *k* is the distance from the hint to the target, instead of the usual O(log n).

`binary_search_from_by` extends the same algorithm to sequences sorted by a key function rather than by element value. This is useful for searching in lists of records, objects, or tuples without extracting keys into a separate list.

### Possible issues

The result may be wrong or not found if the sequence is not sorted (or, for the `_by` variant, not sorted by the given key).

## When to use

| Scenario | `bisect` | `binary_search_from` |
|:---|:---:|:---:|
| No hint — search entire list | O(log n) | O(log n) |
| Hint near target (distance *k*) | O(log n) | **O(log k)** |
| Search by key function | O(log n)* | **O(log k)** |

\* `bisect` supports `key` since Python 3.10, but only for insertion points, and always searches from the middle.

Typical use cases: time-series queries with a cursor, streaming sorted data, iterative algorithms where successive lookups are nearby.

## References

- R. E. Maeder, *Computer Science with Mathematica* (Cambridge University Press, 2000) — standard binary search implementation.
- [Exponential search](https://en.wikipedia.org/wiki/Exponential_search) — Wikipedia.

## License

MIT
