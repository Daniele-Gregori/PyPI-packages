"""
algebraic-range — Generate ranges of algebraic numbers.

A Python port of the Wolfram Language ResourceFunction ``AlgebraicRange``.
"""

from algebraic_range.core import (
    algebraic_range,
    formula_complexity,
    AlgebraicRangeError,
    NotRealError,
    NotAlgebraicError,
    StepBoundError,
    __version__,
)

__all__ = [
    "algebraic_range",
    "formula_complexity",
    "AlgebraicRangeError",
    "NotRealError",
    "NotAlgebraicError",
    "StepBoundError",
]
