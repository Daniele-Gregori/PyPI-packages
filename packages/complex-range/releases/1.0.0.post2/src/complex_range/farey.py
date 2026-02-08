"""
Farey sequence implementation for complex-range.

The Farey sequence F_n is the sequence of completely reduced fractions,
between 0 and 1, which have denominators less than or equal to n,
arranged in order of increasing size.

Author: Daniele Gregori
"""

from fractions import Fraction
from typing import List


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
    
    # Generate all fractions with denominator <= n
    fractions_set = set()
    for denom in range(1, n + 1):
        for numer in range(0, denom + 1):
            fractions_set.add(Fraction(numer, denom))
    
    # Sort and return
    return sorted(fractions_set)


def scaled_farey_sequence(n: int, start: float, end: float) -> List[Fraction]:
    """
    Generate a scaled Farey sequence between start and end.
    
    Parameters
    ----------
    n : int
        The order of the Farey sequence.
    start : float
        The starting value.
    end : float
        The ending value.
        
    Returns
    -------
    List[Fraction]
        Farey sequence scaled to [start, end] interval.
    """
    farey = farey_sequence(n)
    span = end - start
    return [Fraction(start) + f * Fraction(span) for f in farey]
