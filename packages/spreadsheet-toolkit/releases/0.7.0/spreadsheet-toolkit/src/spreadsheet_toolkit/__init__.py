"""spreadsheet_toolkit — Tools for spreadsheet analysis.

Python port of the Wolfram Language package
``DanieleGregori`SpreadsheetToolkit`` and of the Wolfram resource function
``SpreadsheetTrace``.
"""

__version__ = "0.7.0"
__author__ = "Daniele Gregori"

from spreadsheet_toolkit.core import (
    import_all,
    import_cells,
    import_formulas,
    import_sheets,
    index_to_position,
    position_to_index,
    position_to_spreadsheet_index,
    spreadsheet_index_to_position,
)
from spreadsheet_toolkit.trace import spreadsheet_trace

__all__ = [
    "import_all",
    "index_to_position",
    "position_to_index",
    "spreadsheet_trace",
    # backward-compatible names (0.1.x), consistent with the API above
    "spreadsheet_index_to_position",
    "position_to_spreadsheet_index",
    "import_sheets",
    "import_cells",
    "import_formulas",
]
