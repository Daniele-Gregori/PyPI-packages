"""transform_tabular: Transform columns of pandas DataFrames.

Python translation of the Wolfram Language ResourceFunction TransformTabular.
Provides element-wise, column-aggregate (ColumnwiseValue), and column-threaded
(ColumnwiseThread) operations on DataFrame columns.
"""

__version__ = "0.8.0"

from .core import (
    ColumnwiseValue,
    ColumnwiseThread,
    transform_tabular,
)

__all__ = [
    "ColumnwiseValue",
    "ColumnwiseThread",
    "transform_tabular",
]
