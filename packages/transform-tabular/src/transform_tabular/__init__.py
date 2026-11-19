"""transform_tabular: transform columns and rows of pandas DataFrames.

Python translation of the Wolfram Language resource functions TransformTabular and
TransformRows, and of the built-in function TransformColumns:

* ``transform_tabular`` applies one function to every element of all columns, or of a
  selection of them;
* ``transform_columns`` modifies existing columns or adds new ones, row by row;
* ``transform_rows`` transforms only a selection of rows.

All of them support column aggregates (``ColumnwiseValue``) and column-threaded
operations (``ColumnwiseThread``).
"""

__version__ = "0.9.0"

from .core import (
    ColumnwiseThread,
    ColumnwiseValue,
    transform_columns,
    transform_rows,
    transform_tabular,
)

__all__ = [
    "ColumnwiseThread",
    "ColumnwiseValue",
    "transform_columns",
    "transform_rows",
    "transform_tabular",
]
