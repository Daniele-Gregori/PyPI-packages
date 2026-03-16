"""
Core implementation of Farey sequences and Farey ranges.

The Farey sequence F_n is the sequence of completely reduced fractions,
between 0 and 1, which have denominators less than or equal to n,
arranged in order of increasing size.

A Farey range applies a Farey sequence over an arbitrary interval,
producing a set of evenly-distributed rational subdivision points.

Author: Daniele Gregori
"""

from fractions import Fraction
from typing import List, Union


class FareyError(Exception):
    """Raised when invalid arguments are passed to farey functions."""


def farey_sequence(n: int) -> List[Fraction]:
    """
    Generate the Farey sequence of order n.

    The Farey sequence F_n is the sequence of completely reduced fractions
    between 0 and 1, with denominators less than or equal to n, arranged
    in increasing order.

    Parameters
    ----------
    n : int
        The order of the Farey sequence. Must be a positive integer.

    Returns
    -------
    List[Fraction]
        A list of Fraction objects representing the Farey sequence.

    Raises
    ------
    ValueError
        If n is not a positive integer.

    Examples
    --------
    >>> farey_sequence(1)
    [Fraction(0, 1), Fraction(1, 1)]

    >>> farey_sequence(3)
    [Fraction(0, 1), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1, 1)]

    >>> farey_sequence(5)
    [Fraction(0, 1), Fraction(1, 5), Fraction(1, 4), Fraction(1, 3),
     Fraction(2, 5), Fraction(1, 2), Fraction(3, 5), Fraction(2, 3),
     Fraction(3, 4), Fraction(4, 5), Fraction(1, 1)]
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"Order n must be a positive integer, got {n}")

    fractions_set = set()
    for denom in range(1, n + 1):
        for numer in range(0, denom + 1):
            fractions_set.add(Fraction(numer, denom))

    return sorted(fractions_set)


def _resolve_step(step: Union[int, float, Fraction]) -> tuple:
    """Resolve a step argument into (order, reverse).

    Mirrors the WL FareyRange resource function behaviour:
    - Integer n>0:  order n, forward
    - Integer n<0:  order |n|, reversed
    - Fraction 1/n: order n, forward
    - Fraction -1/n: order n, reversed
    - Zero or other rationals: error

    Returns
    -------
    tuple of (int, bool)
        (order, reverse)
    """
    if isinstance(step, Fraction):
        if step == 0:
            raise FareyError("Step must be nonzero")
        reverse = step < 0
        astep = abs(step)
        if astep.denominator == 1:
            return int(astep), reverse
        elif astep.numerator == 1:
            return astep.denominator, reverse
        else:
            raise FareyError(
                f"Step must be a nonzero integer or 1/n, got {step}"
            )
    elif isinstance(step, int):
        if step == 0:
            raise FareyError("Step must be nonzero")
        return abs(step), step < 0
    elif isinstance(step, float):
        if step == 0:
            raise FareyError("Step must be nonzero")
        # Convert float to Fraction and delegate
        frac = Fraction(step).limit_denominator(10**9)
        if abs(float(frac) - step) > 1e-14:
            raise FareyError(
                f"Step must be a nonzero integer or 1/n, got {step}"
            )
        return _resolve_step(frac)
    else:
        raise FareyError(
            f"Step must be a nonzero integer or 1/n, got {step}"
        )


def farey_range(
    start: Union[int, float, Fraction],
    end: Union[int, float, Fraction],
    step: Union[int, float, Fraction, None] = None,
) -> List[Union[Fraction, float]]:
    """
    Generate a Farey range over the interval [start, end].

    Applies a Farey sequence to subdivide the interval into rational
    points. For integer-length intervals, the Farey pattern is applied
    to each unit sub-interval; for non-integer spans, it is scaled over
    the whole interval.

    Parameters
    ----------
    start : int, float, or Fraction
        Left endpoint of the interval.
    end : int, float, or Fraction
        Right endpoint of the interval.
    step : int, float, Fraction, or None
        Controls the Farey order and direction:
        - Positive integer n: Farey order n, ascending.
        - Negative integer -n: Farey order n, descending.
        - Fraction 1/n: Farey order n, ascending.
        - Fraction -1/n: Farey order n, descending.
        - None (default): Farey order 1.

    Returns
    -------
    List[Union[Fraction, float]]
        Sorted (or reverse-sorted) list of values in the range.
        Returns Fraction objects when start and end are integers;
        otherwise returns floats.

    Raises
    ------
    FareyError
        If step is zero or an unsupported rational.

    Examples
    --------
    >>> farey_range(0, 1, 3)
    [Fraction(0, 1), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1, 1)]

    >>> farey_range(0, 2, 2)  # doctest: +NORMALIZE_WHITESPACE
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 1), Fraction(3, 2), Fraction(2, 1)]

    >>> farey_range(0, 1, -3)
    [Fraction(1, 1), Fraction(2, 3), Fraction(1, 2), Fraction(1, 3), Fraction(0, 1)]
    """
    if step is None:
        order, reverse = 1, False
    else:
        order, reverse = _resolve_step(step)

    farey = farey_sequence(order)
    mn, mx = min(start, end), max(start, end)
    span = float(mx) - float(mn)
    num_units = int(span)

    # Non-integer span: scale Farey sequence over the whole interval
    if abs(span - num_units) > 1e-15:
        result = sorted(set(float(mn) + float(f) * span for f in farey))
        if reverse:
            result.reverse()
        return result

    # Integer span: apply Farey to each unit interval
    all_values = set()
    for unit in range(num_units):
        unit_start = float(mn) + unit
        for f in farey:
            all_values.add(unit_start + float(f))
    all_values.add(float(mx))

    result = sorted(all_values)

    # Keep as Fractions when possible
    if all(
        isinstance(x, (int, float, Fraction)) and float(x) == int(float(x))
        for x in [start, end]
    ):
        frac_result = []
        for val in result:
            try:
                frac = Fraction(val).limit_denominator(1000000)
                if abs(float(frac) - val) < 1e-14:
                    frac_result.append(frac)
                else:
                    frac_result.append(val)
            except (ValueError, OverflowError):
                frac_result.append(val)
        result = frac_result

    if reverse:
        result = list(reversed(result))
    return result
