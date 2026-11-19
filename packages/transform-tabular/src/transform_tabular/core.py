"""Transformations of pandas DataFrames: all columns at once, by column, and by row.

Python translations of the Wolfram Language resource functions ``TransformTabular`` and
``TransformRows``, and of the built-in function ``TransformColumns``.
"""

import itertools
import numbers
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .columnwise import ColumnwiseThread, ColumnwiseValue, _Columnwise, _scope

__all__ = [
    "ColumnwiseThread",
    "ColumnwiseValue",
    "transform_columns",
    "transform_rows",
    "transform_tabular",
]


class _Columns(dict):
    """The cells of the columns of a table, as lists keyed by column label.

    A column is read from the table when it is first asked for, so that only the
    columns that the row functions actually use are converted.
    """

    __slots__ = ("numbers", "_table", "_positions")

    def __init__(self, table, positions=None):
        super().__init__()
        self.numbers = {label: number for number, label in enumerate(table.columns)}
        self._table = table
        self._positions = positions

    def __missing__(self, label):
        column = self._table.iloc[:, self.numbers[label]]  # KeyError: not a column
        if self._positions is not None:
            column = column.iloc[self._positions]
        cells = self[label] = column.tolist()
        return cells


class Row(Mapping):
    """A row of a table: a read-only mapping from column labels to values.

    A value can also be read as an attribute (``row.price`` for ``row["price"]``) when
    its label is an identifier that does not start with an underscore and is not the
    name of a mapping method.  ``dict(row)`` gives an independent copy of the row.
    """

    __slots__ = ("_columns", "_number")

    def __init__(self, columns, number):
        self._columns = columns
        self._number = number

    def __getitem__(self, label):
        return self._columns[label][self._number]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._columns[name][self._number]
        except KeyError:
            raise AttributeError(name) from None

    def __contains__(self, label):
        return label in self._columns.numbers

    def __iter__(self):
        return iter(self._columns.numbers)

    def __len__(self):
        return len(self._columns.numbers)

    def __repr__(self):
        return f"Row({dict(self)!r})"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def _check_table(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    if not df.columns.is_unique:
        raise ValueError("DataFrames with repeated column labels are not supported")


def _is_transformation(obj):
    return callable(obj) or isinstance(obj, _Columnwise)


def _is_integer(obj):
    return isinstance(obj, numbers.Integral) and not isinstance(obj, bool)


def _is_label(df, obj):
    try:
        return obj in df.columns
    except TypeError:  # unhashable
        return False


def _column_label(df, labels, item):
    """The label of one column, given by label or by position."""
    if _is_label(df, item):
        return item
    if _is_integer(item):
        try:
            return labels[item]
        except IndexError:
            raise IndexError(
                f"Column position {item} is out of range for a table with "
                f"{len(labels)} columns"
            ) from None
    raise KeyError(f"{item!r} is not among the columns {labels}")


def _resolve_columns(df, columns):
    """The labels of the columns selected by *columns*, without repetitions."""
    labels = list(df.columns)
    if columns is None:
        return labels
    if isinstance(columns, slice):
        return labels[columns]
    if _is_label(df, columns) or _is_integer(columns) or isinstance(columns, (str, bytes)):
        return [_column_label(df, labels, columns)]
    try:
        selection = list(columns)
    except TypeError:
        raise TypeError(f"Invalid column specification: {type(columns)}") from None
    return list(dict.fromkeys(_column_label(df, labels, item) for item in selection))


def _resolve_rows(table, rows):
    """The positions of the rows selected by *rows*, without repetitions."""
    length = len(table)
    if callable(rows):
        with _scope(lambda: table, length) as frame:
            flags = _apply(frame, rows, _row_source(table)())
        return [position for position, flag in enumerate(flags) if flag]
    if isinstance(rows, slice):
        return list(range(length)[rows])
    if _is_integer(rows):
        rows = [rows]
    elif isinstance(rows, (pd.Series, pd.Index)) and pd.api.types.is_bool_dtype(rows.dtype):
        rows = rows.to_numpy(dtype=bool, na_value=False)  # a missing entry selects nothing
    elif isinstance(rows, (str, bytes, Mapping)):
        raise TypeError(f"Invalid row specification: {type(rows)}")
    elif not isinstance(rows, (list, tuple, range, np.ndarray, pd.Series, pd.Index)):
        try:
            rows = list(rows)
        except TypeError:
            raise TypeError(f"Invalid row specification: {type(rows)}") from None
    selection = np.asarray(rows)
    if selection.ndim != 1:
        raise TypeError("Rows must be given as a one-dimensional list")
    if selection.size == 0:
        return []
    if selection.dtype == bool:
        if selection.size != length:
            raise ValueError(
                f"A boolean selection of rows must have one entry per row: got "
                f"{selection.size} entries for {length} rows"
            )
        return np.flatnonzero(selection).tolist()
    if selection.dtype.kind not in "iu":
        raise TypeError(
            "Rows must be given as integer positions, as booleans (one per row), "
            "or as a function of the row"
        )
    selection = selection.astype(np.intp, copy=False)
    positions = np.where(selection < 0, selection + length, selection)
    outside = (positions < 0) | (positions >= length)
    if outside.any():
        raise IndexError(
            f"Row position {selection[outside][0]} is out of range for a table with "
            f"{length} rows"
        )
    return pd.unique(positions).tolist()


def _items(transformations):
    """The items of a specification of column transformations."""
    if isinstance(transformations, Mapping):
        return [transformations]
    if isinstance(transformations, (list, tuple)):
        return list(transformations)
    raise TypeError(
        "Transformations must be given as a dictionary {column: function}, or as a "
        f"list of such dictionaries and of column labels, got {type(transformations).__name__}"
    )


def _position(order, label, role):
    try:
        return order.index(label)
    except ValueError:
        raise KeyError(f"{role} {label!r} is not among the columns {order}") from None


def _plan(df, transformations):
    """Read a specification of column transformations.

    Returns the transformation of every computed column, the order of the columns of
    the result, and the columns of *df* that are removed.
    """
    items = _items(transformations)
    order = list(df.columns)
    functions = {}
    # A new column is placed right after the column named last before it in the list;
    # before any such anchor it comes first, and without anchors it is appended.
    cursor = None if all(isinstance(item, Mapping) for item in items) else 0
    for item in items:
        if not isinstance(item, Mapping):
            cursor = _position(order, item, "The anchor column") + 1
            continue
        for label, function in item.items():
            if function is None:
                position = _position(order, label, "The column to remove")
                del order[position]
                functions.pop(label, None)
                if cursor is not None and position < cursor:
                    cursor -= 1
            elif not _is_transformation(function):
                raise TypeError(
                    f"The transformation of column {label!r} must be a function, a "
                    f"columnwise marker or None, got {type(function).__name__}"
                )
            else:
                if label not in order:
                    if cursor is None:
                        order.append(label)
                    else:
                        order.insert(cursor, label)
                        cursor += 1
                functions[label] = function
    removed = [label for label in df.columns if label not in order]
    return functions, order, removed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _apply(frame, function, arguments):
    """Call *function* on every ``(position, argument)`` pair, row by row."""
    values = []
    for position, argument in arguments:
        frame.row = position
        value = function(argument)
        if isinstance(value, _Columnwise):
            value = value.value
        values.append(value)
    return values


def _row_source(table, positions=None):
    """A function giving, at each call, an iterator over ``(position, row)`` pairs."""
    columns = _Columns(table, positions)
    where = range(len(table)) if positions is None else positions
    return lambda: zip(where, map(Row, itertools.repeat(columns), itertools.count()))


def _tabular_values(table, func, labels, positions=None):
    """The new cells of the columns *labels*, each one transformed element by element.

    Cells are computed at the row *positions* only (at all rows if ``None``); markers
    always see the whole column.
    """
    values = {}
    for label in labels:
        column = table[label]
        if isinstance(func, _Columnwise):
            with _scope(column.tolist, len(column)) as frame:
                values[label] = func._column(frame, positions)
            continue
        if positions is None:
            cells = column.tolist()
            arguments, load = enumerate(cells), cells.copy
        else:
            arguments, load = zip(positions, column.iloc[positions].tolist()), column.tolist
        with _scope(load, len(column)) as frame:
            values[label] = _apply(frame, func, arguments)
    return values


def _columns_values(table, functions, positions=None):
    """The new cells of the columns in *functions*, each one computed from the rows.

    Cells are computed at the row *positions* only (at all rows if ``None``); markers
    always see the whole table.
    """
    values = {}
    rows = None
    with _scope(lambda: table, len(table)) as frame:
        for label, function in functions.items():
            if isinstance(function, _Columnwise):
                values[label] = function._column(frame, positions)
            else:
                if rows is None:
                    rows = _row_source(table, positions)
                values[label] = _apply(frame, function, rows())
    return values


def _series(values, index):
    """A column with the given *index*, from a list of cells or a positional Series."""
    if isinstance(values, pd.Series):
        return pd.Series(values.array, index=index, copy=True)
    if len(index) == 0:
        return pd.Series([], index=index, dtype=object)
    return pd.Series(values, index=index)


def _set_column(table, label, values):
    """Set, in place, column *label* of *table* to the cells *values*."""
    if len(table.index) == 0 and label in table.columns and not isinstance(values, pd.Series):
        return  # no rows to transform: the column keeps its type
    table[label] = _series(values, table.index)


# ---------------------------------------------------------------------------
# TransformTabular
# ---------------------------------------------------------------------------

def transform_tabular(df_or_func, func_or_columns=None, columns=None):
    """Transform all columns of a DataFrame, or a selection of them, with one function.

    Direct forms::

        transform_tabular(df, func)            # all columns
        transform_tabular(df, func, columns)   # selected columns

    Operator forms (return a function of the DataFrame)::

        transform_tabular(func)(df)
        transform_tabular(func, columns)(df)

    Parameters
    ----------
    df : pandas.DataFrame
        The table to transform.  It is not modified.
    func : callable
        Function ``func(x) -> y`` applied to every element of the selected columns.
        It may use :class:`ColumnwiseValue` and :class:`ColumnwiseThread` markers,
        whose own functions receive the column as a list.
    columns : column specification, optional
        The columns to transform:

        * ``None`` -- all columns (default)
        * a column label
        * an ``int`` -- one column by position (0-based; negative from the end)
        * a list of labels or positions
        * a ``slice`` of positions

    Returns
    -------
    pandas.DataFrame or callable
        The transformed table, with the same rows, columns and index, or, in the
        operator forms, a function that transforms a table.
    """
    if not isinstance(df_or_func, pd.DataFrame):
        if not _is_transformation(df_or_func):
            raise TypeError(
                f"First argument must be a DataFrame or callable, "
                f"got {type(df_or_func)}"
            )
        func = df_or_func
        selection = columns if func_or_columns is None else func_or_columns
        return lambda df: transform_tabular(df, func, selection)

    df, func = df_or_func, func_or_columns
    if func is None:
        raise ValueError("Function argument is required")
    if not _is_transformation(func):
        raise TypeError(f"Second argument must be callable, got {type(func)}")
    _check_table(df)

    labels = _resolve_columns(df, columns)
    result = df.copy()
    for label, values in _tabular_values(df, func, labels).items():
        _set_column(result, label, values)
    return result


# ---------------------------------------------------------------------------
# TransformColumns
# ---------------------------------------------------------------------------

def transform_columns(df_or_transformations, transformations=None):
    """Modify existing columns of a DataFrame, or add new ones, row by row.

    Direct form::

        transform_columns(df, {"column": func, ...})

    Operator form (returns a function of the DataFrame)::

        transform_columns({"column": func, ...})(df)

    Parameters
    ----------
    df : pandas.DataFrame
        The table to transform.  It is not modified.
    transformations : dict or list
        A dictionary ``{column: transformation}``.  Each transformation is one of:

        * a function ``func(row) -> value``, applied to every row.  The row is a
          read-only mapping from column labels to values, which can also be read
          as attributes (``row["a"]`` or ``row.a``).  The function may use
          :class:`ColumnwiseValue` and :class:`ColumnwiseThread` markers, whose own
          functions receive the whole table (the DataFrame);
        * a marker itself, when the new column does not depend on the single row;
        * ``None``, to remove the column.

        An existing column is replaced in place; a new column is appended after the
        existing ones.  All transformations are computed from the original table: none
        of them sees the result of another.

        To place new columns elsewhere, give a list that mixes dictionaries and labels
        of existing columns: a new column goes right after the column named last
        before it in the list (``["a", {"new": func}]`` puts ``new`` after ``a``), and
        first if no column is named before it (``[{"new": func}, "a"]``).

    Returns
    -------
    pandas.DataFrame or callable
        The transformed table, with the same rows and index, or, in the operator form,
        a function that transforms a table.
    """
    if not isinstance(df_or_transformations, pd.DataFrame):
        if transformations is not None:
            raise TypeError(
                f"First argument must be a DataFrame, "
                f"got {type(df_or_transformations).__name__}"
            )
        specification = df_or_transformations
        _items(specification)
        return lambda df: transform_columns(df, specification)

    df = df_or_transformations
    if transformations is None:
        raise ValueError("Transformations argument is required")
    _check_table(df)

    functions, order, _ = _plan(df, transformations)
    result = df.copy()
    # All the new cells are computed before any of them is stored.
    for label, values in _columns_values(result, functions).items():
        _set_column(result, label, values)
    if list(result.columns) != order:
        result = result[order]
    return result


# ---------------------------------------------------------------------------
# TransformRows
# ---------------------------------------------------------------------------

def _scatter(pieces, length, index):
    """Assemble a column from pieces labelled by row position; other rows are missing."""
    column = pieces[0] if len(pieces) == 1 else pd.concat(pieces)
    column = column.reindex(pd.RangeIndex(length))
    column.index = index
    return column


def transform_rows(df, rows, transformation, *, keep_order=True):
    """Transform only selected rows of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The table to transform.  It is not modified.
    rows : row specification
        The rows to transform:

        * an ``int`` -- one row by position (0-based; negative from the end)
        * a list of positions
        * a function ``crit(row) -> bool`` -- all the rows for which it is true, the
          row being passed as in :func:`transform_columns`
        * a ``slice`` of positions, or a list or ``Series`` of booleans, one per row
    transformation : dict, list or callable
        What to do with the selected rows:

        * column transformations, given as in :func:`transform_columns`
          (``{"column": func(row)}``);
        * or one function ``func(x) -> y`` applied to every cell of the selected
          rows, as in :func:`transform_tabular`.

        :class:`ColumnwiseValue` and :class:`ColumnwiseThread` markers are computed
        on whole columns, including the rows that are not selected.
    keep_order : bool, default True
        Whether to keep the original order of the rows.  If false, the transformed
        rows come first, in the order in which they were selected, followed by the
        others.

    Returns
    -------
    pandas.DataFrame
        The table with the selected rows transformed.  The other rows are left as
        they are: in a column that is new they are missing.  A column to remove
        (``None``) loses its cells in the selected rows only.  If no row is selected,
        the table is returned unchanged.
    """
    _check_table(df)
    if isinstance(transformation, (Mapping, list, tuple)):
        functions, order, removed = _plan(df, transformation)
    elif _is_transformation(transformation):
        functions, order, removed = None, list(df.columns), []
    else:
        raise TypeError(
            "The transformation must be a function, or column transformations as in "
            f"transform_columns, got {type(transformation).__name__}"
        )

    result = df.copy()
    positions = _resolve_rows(result, rows)
    if not positions:
        return result
    if functions is None:
        values = _tabular_values(result, transformation, order, positions)
    else:
        values = _columns_values(result, functions, positions)

    length = len(df)
    chosen = np.asarray(positions, dtype=np.intp)
    untouched = np.ones(length, dtype=bool)
    untouched[chosen] = False
    rest = np.flatnonzero(untouched)

    def kept(label):
        """The cells of an original column in the rows that are not selected."""
        column = df[label].iloc[rest]
        column.index = rest
        return column

    for label, cells in values.items():
        pieces = [_series(cells, chosen)]
        if label in df.columns and rest.size:
            pieces.insert(0, kept(label))
        result[label] = _scatter(pieces, length, df.index)
    for label in removed:
        result[label] = _scatter([kept(label)], length, df.index)

    for label in removed:
        order.insert(min(df.columns.get_loc(label), len(order)), label)
    if list(result.columns) != order:
        result = result[order]
    if not keep_order:
        result = result.iloc[np.concatenate([chosen, rest])]
    return result
