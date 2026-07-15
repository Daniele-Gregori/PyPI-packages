"""Trace dependencies among spreadsheet formulas.

Python port of the Wolfram Language resource function ``SpreadsheetTrace``
(1.0.0) by Daniele Gregori. The output closely mirrors that of the Wolfram
Language ``Trace``, but is cut before the actual spreadsheet evaluations
occur: each traced cell produces ``[cell, formula-or-value, subtraces...]``
and leaf cells produce ``[cell, value]``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union

from spreadsheet_toolkit.core import import_all, index_to_position, position_to_index

# ---------------------------------------------------------------------------
# Helpers: cell resolution
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+")


def _row_to_index(ref: str) -> int:
    """First number occurring in a cell reference (mirrors ``row2Ind``)."""
    return int(_NUMBER_RE.search(ref).group())


def _clean_cell_ref(cell: str) -> str:
    """Strip a leading ``Sheet!`` prefix (mirrors ``cleanCellRef``)."""
    return cell.split("!")[-1] if "!" in cell else cell


def _cell_sheet_index(cell: str, sheets: list, default: int = 1) -> int:
    """1-based sheet index of a cell reference (mirrors ``cellSheetIndex``)."""
    if "!" in cell:
        name = cell.split("!")[0]
        try:
            return sheets.index(name) + 1
        except ValueError:
            return default
    return default


# ---------------------------------------------------------------------------
# Parsing: main patterns
# ---------------------------------------------------------------------------

# cellPatt: plain and absolute ($) cell references
_CELL_PATTERNS = [
    r"[A-Za-z]+\d+",
    r"\$[A-Za-z]+\d+",
    r"[A-Za-z]+\$\d+",
    r"\$[A-Za-z]+\$\d+",
]

# colPatt
_COL_PATTERN = r"[A-Za-z]+"

# cellGenPatt: cell reference with an optional "Sheet!" prefix
_CELL_GEN_RE = re.compile(r"(?:.+!)?\$?[A-Za-z]+\$?\d+")


def _scan_regex(sheets: list, include_columns: bool):
    """Compile the reference-scanning pattern.

    Alternative order mirrors the Wolfram ``StringCases`` pattern lists:
    plain cell patterns first (``elpatt``), then sheet-prefixed cell patterns
    (``shpatt``), then — only when a column range occurred — sheet-prefixed
    column patterns (``cshpatt``).
    """
    alternatives = list(_CELL_PATTERNS)
    for pattern in _CELL_PATTERNS:
        for sheet in sheets:
            alternatives.append(re.escape(sheet) + "!" + pattern)
    if include_columns:
        for sheet in sheets:
            alternatives.append(re.escape(sheet) + "!" + _COL_PATTERN)
    return re.compile("|".join(f"(?:{a})" for a in alternatives))


# ---------------------------------------------------------------------------
# Parsing: formula parser
# ---------------------------------------------------------------------------


def _has_number(part: str) -> bool:
    return _NUMBER_RE.search(part) is not None


def _is_column_range(token: str, sheets: list) -> bool:
    """Whether the token ends with a sheet-prefixed column range like
    ``Products!A:C`` (mirrors ``colrgqF``)."""
    return any(
        re.fullmatch(rf".*{re.escape(sheet)}!{_COL_PATTERN}:{_COL_PATTERN}", token)
        for sheet in sheets
    )


def _leftmost_sheet(token: str, sheets: list) -> str:
    """First sheet name occurring in the token (mirrors
    ``StringCases[token, Alternatives @@ sheets]``)."""
    if not sheets:
        return ""
    match = re.search("|".join(re.escape(s) for s in sheets), token)
    return match.group(0) if match else ""


def _expand_cell_range(start: str, end: str) -> list:
    """Expand a cell range like ``A1:A3`` row by row (mirrors ``repcellrg``).

    The column part is taken from the range start, so row ranges such as
    ``B3:H3`` reduce to their first cell, as in the Wolfram original.
    """
    column = _NUMBER_RE.sub("", start)
    return [column + str(j) for j in range(_row_to_index(start), _row_to_index(end) + 1)]


def _expand_column_range(start: str, end: str, sheet: str, sheets: list, nrows: list) -> list:
    """Expand a column range like ``Products!A:C`` across all rows of the
    sheet, column-major (mirrors ``repcolrg``)."""
    start_col = index_to_position(_clean_cell_ref(start) + "1")[1]
    end_col = index_to_position(end + "1")[1]
    if nrows and sheet in sheets:
        n = nrows[sheets.index(sheet)]
    else:
        n = 0
    if n <= 0:
        return []
    refs = []
    for c in range(start_col, end_col + 1):
        for r in range(1, n + 1):
            ref = position_to_index((r, c))
            refs.append(sheet + "!" + ref if sheet != "" else ref)
    return refs


def _formula_parse(form: Any, sheets: list, nrows: list) -> list:
    """Extract the cell references a formula depends on, in order
    (mirrors ``spreadsheetFormulaParse``, flattened)."""
    if not isinstance(form, str) or form == "":
        return []

    sheet = ""
    column_range_seen = False

    # main parsing: split at parentheses and commas, expand ranges per token
    processed = []  # each entry: a string, or a list of strings
    for token in re.split(r"[(),]", form):
        if ":" not in token:
            processed.append(token)
            continue
        parts = token.split(":")
        expandable = len(parts) == 2
        if not _is_column_range(token, sheets):
            if expandable and _has_number(parts[0]) and _has_number(parts[1]):
                processed.append(_expand_cell_range(parts[0], parts[1]))
            else:
                processed.append(parts)
        else:
            sheet = _leftmost_sheet(token, sheets)
            column_range_seen = True
            if expandable and _has_number(parts[0]) and _has_number(parts[1]):
                processed.append(_expand_cell_range(parts[0], parts[1]))
            elif expandable and not _has_number(parts[0]) and not _has_number(parts[1]):
                processed.append(_expand_column_range(parts[0], parts[1], sheet, sheets, nrows))
            else:
                processed.append(parts)

    # scan for cell references; no column-only patterns unless a column range occurred
    pattern = _scan_regex(sheets, column_range_seen)
    refs = []
    for entry in processed:
        for item in entry if isinstance(entry, list) else [entry]:
            for match in pattern.finditer(item):
                # delete dollar
                refs.append(match.group(0).replace("$", ""))
    return refs


# ---------------------------------------------------------------------------
# Tracing: operations
# ---------------------------------------------------------------------------


def _formula_q(pos: tuple, si: int, formulas: list) -> bool:
    """Whether the cell at ``pos`` holds a formula (mirrors ``formulaQ``)."""
    return formulas[si - 1][pos[0] - 1][pos[1] - 1] != ""


def _get_formula(fq: bool, pos: tuple, si: int, data: list, formulas: list) -> Any:
    """The formula string, or the plain value for non-formula cells
    (mirrors ``getFormula``)."""
    grid = formulas[si - 1] if fq else data[si - 1]
    return grid[pos[0] - 1][pos[1] - 1]


def _get_parsing(fq: bool, form: Any, sheets: list, data: list, optdup: bool) -> list:
    """Parsed dependencies of a formula (mirrors ``getParsing``)."""
    nrows = [len(grid) for grid in data] if data else []
    if not fq:
        return []
    refs = _formula_parse(form, sheets, nrows)
    if optdup:
        return refs
    return list(dict.fromkeys(refs))


def _get_position(cell: str) -> tuple:
    """(row, column) of a cell reference (mirrors ``getPosition``)."""
    return index_to_position(_clean_cell_ref(cell))


# ---------------------------------------------------------------------------
# Tracing: cell pipeline
# ---------------------------------------------------------------------------


def _cell_pipeline(vec, cell, si, sheets, data, formulas, optdup):
    """Resolve one cell: append its formula/value to ``vec`` and return
    ``(vec, parsed dependencies, sheet index)`` (mirrors ``cellPipeline``).

    Out-of-range references reset the accumulated trace to ``[]``.
    """
    pos = _get_position(cell)
    nsi = _cell_sheet_index(cell, sheets, si)
    grid = data[nsi - 1]
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0
    if not (pos[0] <= n_rows and pos[1] <= n_cols):
        return [], [], si

    fq = _formula_q(pos, nsi, formulas)
    form = _get_formula(fq, pos, nsi, data, formulas)
    parse = _get_parsing(fq, form, sheets, data, optdup)

    return vec + [form], parse, nsi


# ---------------------------------------------------------------------------
# Tracing: core recursion
# ---------------------------------------------------------------------------


def _trace_cell(vec, cell, psi, sheets, data, formulas, optdup):
    """Mirror of ``iSpreadsheetTrace[v, cell, psi, ...]``."""
    vec = vec + [cell]
    vec, parse, si = _cell_pipeline(vec, cell, psi, sheets, data, formulas, optdup)

    if len(parse) == 1:
        nvec = list(parse)
        ncell = parse[0]
        nvec, nparse, nsi = _cell_pipeline(nvec, ncell, si, sheets, data, formulas, optdup)
        if nvec == [] and nparse == []:
            pass  # iSpreadsheetTrace[{}, {}, ...] := Nothing
        else:
            vec = vec + [_trace_parse(nvec, nparse, nsi, sheets, data, formulas, optdup)]
    elif len(parse) >= 2:
        for ncell in parse:
            vec = vec + [_trace_cell([], ncell, si, sheets, data, formulas, optdup)]

    return vec


def _trace_parse(vec, parse, si, sheets, data, formulas, optdup):
    """Mirror of ``iSpreadsheetTrace[v, parse, si, ...]``."""
    for ncell in parse:
        vec = vec + [_trace_cell([], ncell, si, sheets, data, formulas, optdup)]
    return vec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _validate_book(sheets, data, formulas) -> None:
    """Structural checks mirroring the Depth conditions of the Wolfram
    definition: flat sheet names, 2D data and formula grids per sheet."""
    if not isinstance(sheets, list) or not all(isinstance(s, str) for s in sheets):
        raise ValueError("sheets must be a flat list of sheet-name strings")
    for name, grids in (("data", data), ("formulas", formulas)):
        if not isinstance(grids, list) or not all(
            isinstance(grid, list) and all(isinstance(row, list) for row in grid)
            for grid in grids
        ):
            raise ValueError(f"{name} must be a list of 2D per-sheet grids")


def spreadsheet_trace(source, cell: str, trace_duplicates: bool = True) -> list:
    """Trace the dependencies of a spreadsheet cell.

    Mirrors ``SpreadsheetTrace[file, cell]`` and
    ``SpreadsheetTrace[{sheets, data, formulas}, cell]``.

    Parameters
    ----------
    source : str, Path, or tuple
        Path to an ``.xlsx`` file, or an ``(sheets, data, formulas)`` triple
        as returned by :func:`spreadsheet_toolkit.import_all`.
    cell : str
        A cell reference such as ``"D1"``, ``"$B$10"`` or ``"Summary!B3"``.
    trace_duplicates : bool
        Keep duplicate dependency branches (the ``"TraceDuplicates"``
        option; default ``True``).

    Returns
    -------
    list
        A nested trace ``[cell, formula, subtraces...]``, where each leaf
        cell contributes ``[cell, value]``.

    Examples
    --------
    >>> spreadsheet_trace("book.xlsx", "D1")  # doctest: +SKIP
    ['D1', 'C1*2', ['C1', 'A1+B1', ['A1', 10], ['B1', 20]]]
    """
    if not isinstance(cell, str):
        raise TypeError(f"Cell reference must be a string, got {type(cell).__name__}")
    if _CELL_GEN_RE.fullmatch(cell) is None:
        raise ValueError(f"Invalid cell reference: {cell!r}")

    if isinstance(source, (str, Path)):
        sheets, data, formulas = import_all(source)
    elif isinstance(source, (tuple, list)) and len(source) == 3:
        sheets, data, formulas = source
        _validate_book(sheets, data, formulas)
    else:
        raise TypeError(
            "source must be a spreadsheet file path or an "
            "(sheets, data, formulas) triple"
        )

    si = _cell_sheet_index(cell, sheets, 1)
    return _trace_cell([], cell, si, sheets, data, formulas, trace_duplicates)
