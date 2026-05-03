"""find_closed_form — Find closed-form expressions for numerical values."""

__version__ = "0.0.2"
__author__ = "Daniele Gregori"

from find_closed_form.core import (
    find_closed_form,
    formula_complexity,
    farey_range,
    FindClosedFormError,
)

__all__ = [
    "find_closed_form",
    "formula_complexity",
    "farey_range",
    "FindClosedFormError",
]
