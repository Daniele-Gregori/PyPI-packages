"""transcendental-range: generate transcendental numbers within a range."""

from .core import (
    transcendental_range,
    TranscendentalRangeError,
    NotAlgebraicError,
)

__all__ = [
    "transcendental_range",
    "TranscendentalRangeError",
    "NotAlgebraicError",
]

__version__ = "0.8.0a0"
