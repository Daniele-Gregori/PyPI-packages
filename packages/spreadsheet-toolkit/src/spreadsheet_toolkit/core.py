"""Core spreadsheet utility functions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union

import openpyxl

_CELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _col_label_to_number(label: str) -> int:
    """Convert a column label like 'A', 'Z', 'AA' to a 1-based column number."""
    n = 0
    for ch in label.upper():
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def _number_to_col_label(n: int) -> str:
    """Convert a 1-based column number to a column label like 'A', 'Z', 'AA'."""
    if n < 1:
        raise ValueError(f"Column number must be >= 1, got {n}")
    letters = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def spreadsheet_index_to_position(index: str) -> tuple[int, int]:
    """Convert a spreadsheet cell reference to a (row, column) position.

    Parameters
    ----------
    index : str
        A cell reference such as ``"A1"``, ``"B3"``, or ``"AA12"``.

    Returns
    -------
    tuple[int, int]
        ``(row, column)`` with 1-based indices.

    Examples
    --------
    >>> spreadsheet_index_to_position("A1")
    (1, 1)
    >>> spreadsheet_index_to_position("C5")
    (5, 3)
    >>> spreadsheet_index_to_position("AA1")
    (1, 27)
    """
    m = _CELL_RE.match(index)
    if m is None:
        raise ValueError(f"Invalid cell reference: {index!r}")
    col_label, row_str = m.group(1), m.group(2)
    row = int(row_str)
    if row < 1:
        raise ValueError(f"Row number must be >= 1, got {row}")
    return (row, _col_label_to_number(col_label))


def position_to_spreadsheet_index(row: int, column: int) -> str:
    """Convert a (row, column) position to a spreadsheet cell reference.

    Parameters
    ----------
    row : int
        1-based row number.
    column : int
        1-based column number.

    Returns
    -------
    str
        A cell reference such as ``"A1"`` or ``"AA12"``.

    Examples
    --------
    >>> position_to_spreadsheet_index(1, 1)
    'A1'
    >>> position_to_spreadsheet_index(5, 3)
    'C5'
    >>> position_to_spreadsheet_index(1, 27)
    'AA1'
    """
    if row < 1:
        raise ValueError(f"Row must be >= 1, got {row}")
    if column < 1:
        raise ValueError(f"Column must be >= 1, got {column}")
    return f"{_number_to_col_label(column)}{row}"


def import_sheets(path: Union[str, Path]) -> list[str]:
    """Return the list of sheet names in a spreadsheet file.

    Parameters
    ----------
    path : str or Path
        Path to an ``.xlsx`` file.

    Returns
    -------
    list[str]
        Sheet names in workbook order.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def import_cells(
    path: Union[str, Path],
    sheet: Union[str, int, None] = None,
    rows: Union[tuple[int, int], None] = None,
    columns: Union[tuple[int, int], None] = None,
) -> list[list[Any]]:
    """Import cell values from a spreadsheet.

    Parameters
    ----------
    path : str or Path
        Path to an ``.xlsx`` file.
    sheet : str, int, or None
        Sheet name (str) or 1-based index (int). ``None`` uses the active sheet.
    rows : tuple[int, int] or None
        ``(first_row, last_row)`` inclusive, 1-based. ``None`` reads all rows.
    columns : tuple[int, int] or None
        ``(first_col, last_col)`` inclusive, 1-based. ``None`` reads all columns.

    Returns
    -------
    list[list[Any]]
        A 2D list of cell values (one inner list per row).
    """
    return _import_sheet(path, sheet, rows, columns, formulas=False)


def import_formulas(
    path: Union[str, Path],
    sheet: Union[str, int, None] = None,
    rows: Union[tuple[int, int], None] = None,
    columns: Union[tuple[int, int], None] = None,
) -> list[list[Any]]:
    """Import cell formulas from a spreadsheet.

    Cells that contain a formula return the formula string (e.g.
    ``"=A1+B1"``). Cells without a formula return their value.

    Parameters
    ----------
    path : str or Path
        Path to an ``.xlsx`` file.
    sheet : str, int, or None
        Sheet name (str) or 1-based index (int). ``None`` uses the active sheet.
    rows : tuple[int, int] or None
        ``(first_row, last_row)`` inclusive, 1-based. ``None`` reads all rows.
    columns : tuple[int, int] or None
        ``(first_col, last_col)`` inclusive, 1-based. ``None`` reads all columns.

    Returns
    -------
    list[list[Any]]
        A 2D list where formula cells contain their formula string.
    """
    return _import_sheet(path, sheet, rows, columns, formulas=True)


def _resolve_sheet(wb: openpyxl.Workbook, sheet: Union[str, int, None]):
    if sheet is None:
        return wb.active
    if isinstance(sheet, int):
        if sheet < 1 or sheet > len(wb.sheetnames):
            raise ValueError(
                f"Sheet index {sheet} out of range (workbook has {len(wb.sheetnames)} sheets)"
            )
        return wb[wb.sheetnames[sheet - 1]]
    if sheet not in wb.sheetnames:
        raise ValueError(f"Sheet {sheet!r} not found (available: {wb.sheetnames})")
    return wb[sheet]


def _import_sheet(
    path: Union[str, Path],
    sheet: Union[str, int, None],
    rows: Union[tuple[int, int], None],
    columns: Union[tuple[int, int], None],
    formulas: bool,
) -> list[list[Any]]:
    wb = openpyxl.load_workbook(path, read_only=False, data_only=not formulas)
    try:
        ws = _resolve_sheet(wb, sheet)

        min_row = rows[0] if rows else 1
        max_row = rows[1] if rows else ws.max_row
        min_col = columns[0] if columns else 1
        max_col = columns[1] if columns else ws.max_column

        result = []
        for row in ws.iter_rows(
            min_row=min_row, max_row=max_row,
            min_col=min_col, max_col=max_col,
        ):
            result.append([cell.value for cell in row])
        return result
    finally:
        wb.close()
