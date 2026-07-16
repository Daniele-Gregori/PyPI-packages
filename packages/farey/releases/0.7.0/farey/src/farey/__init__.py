"""farey — Generate Farey sequences and Farey ranges."""

__version__ = "0.7.0"
__author__ = "Daniele Gregori"

from farey.core import farey_sequence, farey_range, FareyError

__all__ = [
    "farey_sequence",
    "farey_range",
    "FareyError",
]
