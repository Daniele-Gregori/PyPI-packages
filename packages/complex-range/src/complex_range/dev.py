"""
Development/testing utilities for complex-range.

This module exposes internal functions for interactive testing and debugging.
Import with: from complex_range.dev import *

Author: Daniele Gregori
"""

from .core import (
    complex_range,
    complex_range_iter,
    ComplexRangeError,
    _rectangular_range,
    _linear_range,
    _arange_inclusive,
    _make_complex,
    _farey_range_values,
    _handle_linear_range,
    _handle_rectangular_range,
    _to_number,
)
from .farey import farey_sequence, scaled_farey_sequence

__all__ = [
    # Public API
    "complex_range",
    "complex_range_iter",
    "ComplexRangeError",
    "farey_sequence",
    # Internal functions (for testing)
    "_rectangular_range",
    "_linear_range", 
    "_arange_inclusive",
    "_make_complex",
    "_farey_range_values",
    "_handle_linear_range",
    "_handle_rectangular_range",
    "_to_number",
    "scaled_farey_sequence",
]
