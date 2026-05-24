"""gallop-search — Binary search from an arbitrary position in sorted sequences."""

__version__ = "0.9.0"
__author__ = "Daniele Gregori"

from gallop_search.core import (
    binary_search_from,
    binary_search_from_by,
)

__all__ = [
    "binary_search_from",
    "binary_search_from_by",
]
