"""Core binary-search-from algorithms.

A faithful translation of BinarySearchFrom (0.8.0) for the Wolfram Language,
contributed by Daniele Gregori.  The standard binary-search core follows
Roman E. Maeder, *Computer Science with Mathematica* (Cambridge UP, 2000).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_start(n: int, start: int) -> int:
    if start < 0:
        start += n
    if not (0 <= start < n):
        raise IndexError(
            f"start index out of range for sequence of length {n}"
        )
    return start


def _binary_search_core(value, b0, b1, get):
    while b0 <= b1:
        m = (b0 + b1) // 2
        v = get(m)
        if v == value:
            return m
        if v < value:
            b0 = m + 1
        else:
            b1 = m - 1
    return None


def _from_beginning(value, n, get):
    last = n - 1

    if get(0) == value:
        return 0
    if n < 2:
        return None
    if get(1) == value:
        return 1

    a = 2
    n0 = 1
    m = n1 = min((1 << a) - 1, last)

    while n0 <= n1:
        vm = get(m)
        if vm == value:
            return m
        if vm < value:
            if m >= last:
                return None
            n0 = min(n1, last)
            a += 1
            m = n1 = min((1 << a) - 1, last)
        else:
            return _binary_search_core(value, n0, min(n1, last), get)

    return None


def _from_index(value, n, s, get):
    last = n - 1
    vs = get(s)

    if vs <= value:
        # -- forward search --
        if vs == value:
            return s
        if s >= last:
            return None

        n1 = s + 1
        if get(n1) == value:
            return n1

        a = 2
        n0 = min(n1, last)
        m = n1 = min(s + (1 << a), last)

        while n0 <= n1:
            vm = get(m)
            if vm == value:
                return m
            if vm < value:
                if m >= last:
                    return None
                n0 = min(n1, last)
                a += 1
                m = n1 = min(s + (1 << a), last)
            else:
                return _binary_search_core(
                    value, min(n0, last), min(n1, last), get
                )
    else:
        # -- backward search --
        n0 = s - 1
        if n < 2 or s <= 0:
            return None

        a = 1
        n1 = max(n0, 0)
        m = n0 = max(s - 1 - (1 << a), 0)

        while n0 <= n1:
            vm = get(m)
            if vm == value:
                return m
            if vm > value:
                if m <= 0:
                    return None
                n1 = max(n0, 0)
                a += 1
                m = n0 = max(s - 1 - (1 << a), 0)
            else:
                return _binary_search_core(
                    value, max(n0, 0), max(n1, 0), get
                )

        return None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def binary_search_from(
    seq: Sequence,
    value: Any,
    start: int = 0,
) -> Optional[int]:
    """Find *value* in a sorted sequence, galloping from *start*.

    Uses exponential (galloping) search to locate the target range in
    O(log k) time, where *k* is the distance from *start* to the target,
    then narrows with binary search inside that range.

    Parameters
    ----------
    seq : Sequence
        A sorted sequence supporting ``__getitem__`` and ``__len__``.
        Elements must be comparable with ``<``, ``>``, and ``==``.
    value
        The value to search for.
    start : int, optional
        Position to begin the gallop (default ``0``).  Negative indices
        count from the end (``-1`` is the last element).

    Returns
    -------
    int or None
        Index of *value*, or ``None`` if not found.

    Examples
    --------
    >>> binary_search_from([10, 20, 30, 40, 50], 30)
    2
    >>> binary_search_from(list(range(10_000)), 9000, start=8990)
    9000
    """
    n = len(seq)
    if n == 0:
        return None

    s = _resolve_start(n, start)

    def get(m):
        return seq[m]

    if s == 0:
        return _from_beginning(value, n, get)
    return _from_index(value, n, s, get)


def binary_search_from_by(
    seq: Sequence,
    value: Any,
    *,
    key: Callable[[Any], Any],
    start: int = 0,
) -> Optional[int]:
    """Find an element in a key-sorted sequence, galloping from *start*.

    Searches for the first element whose *key* equals *value*, using the
    same exponential-then-binary algorithm as :func:`binary_search_from`.

    Parameters
    ----------
    seq : Sequence
        A sequence sorted by *key*: ``key(seq[0]) <= key(seq[1]) <= ...``.
    value
        The target value to match against ``key(element)``.
    key : callable
        Extracts the comparison value from each element.
    start : int, optional
        Position to begin the gallop (default ``0``).

    Returns
    -------
    int or None
        Index of the matching element, or ``None`` if not found.

    Examples
    --------
    >>> data = [{"t": 1}, {"t": 5}, {"t": 9}]
    >>> binary_search_from_by(data, 5, key=lambda d: d["t"])
    1
    >>> binary_search_from_by(data, 9, key=lambda d: d["t"], start=0)
    2
    """
    n = len(seq)
    if n == 0:
        return None

    s = _resolve_start(n, start)

    def get(m):
        return key(seq[m])

    if s == 0:
        return _from_beginning(value, n, get)
    return _from_index(value, n, s, get)
