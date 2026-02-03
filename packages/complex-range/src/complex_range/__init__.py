"""
complex-range - Generate ranges of complex numbers.

A Python implementation of the Wolfram Language ComplexRange resource function,
originally created by Daniele Gregori.

Generates rectangular and linear ranges of complex numbers in the complex plane.

Example usage:
    >>> from complex_range import complex_range
    >>> list(complex_range(0, 2+2j))
    [0j, 1j, 2j, (1+0j), (1+1j), (1+2j), (2+0j), (2+1j), (2+2j)]
    
    >>> list(complex_range([0, 1+1j]))  # Linear range
    [0j, (1+1j)]

Author: Daniele Gregori
"""

from .core import complex_range, complex_range_iter, ComplexRangeError
from .farey import farey_sequence

__version__ = "1.0.0"
__author__ = "Daniele Gregori"
__all__ = ["complex_range", "complex_range_iter", "ComplexRangeError", "farey_sequence"]
