"""Core implementation of transform_tabular."""

import pandas as pd

# Module-level row index for ColumnwiseThread synchronisation
_row_index = 0


def _get_value(obj):
    """Extract current value from a marker, or return the object as-is."""
    if isinstance(obj, (ColumnwiseValue, ColumnwiseThread)):
        return obj._v
    return obj


# ---------------------------------------------------------------------------
# Arithmetic mixin — shared by both marker classes
# ---------------------------------------------------------------------------

class _ArithmeticMixin:
    """Provides arithmetic and comparison operators using ``self._v``."""

    def __add__(self, other): return self._v + _get_value(other)
    def __radd__(self, other): return _get_value(other) + self._v
    def __sub__(self, other): return self._v - _get_value(other)
    def __rsub__(self, other): return _get_value(other) - self._v
    def __mul__(self, other): return self._v * _get_value(other)
    def __rmul__(self, other): return _get_value(other) * self._v
    def __truediv__(self, other): return self._v / _get_value(other)
    def __rtruediv__(self, other): return _get_value(other) / self._v
    def __floordiv__(self, other): return self._v // _get_value(other)
    def __rfloordiv__(self, other): return _get_value(other) // self._v
    def __mod__(self, other): return self._v % _get_value(other)
    def __rmod__(self, other): return _get_value(other) % self._v
    def __pow__(self, other): return self._v ** _get_value(other)
    def __rpow__(self, other): return _get_value(other) ** self._v
    def __neg__(self): return -self._v
    def __pos__(self): return +self._v
    def __abs__(self): return abs(self._v)
    def __lt__(self, other): return self._v < _get_value(other)
    def __le__(self, other): return self._v <= _get_value(other)
    def __gt__(self, other): return self._v > _get_value(other)
    def __ge__(self, other): return self._v >= _get_value(other)
    def __float__(self): return float(self._v)
    def __int__(self): return int(self._v)
    def __bool__(self): return bool(self._v)


# ---------------------------------------------------------------------------
# ColumnwiseValue
# ---------------------------------------------------------------------------

class ColumnwiseValue(_ArithmeticMixin):
    """Column-level aggregation marker.

    Wraps a function ``func(column_as_list) -> scalar``.  When used inside a
    ``transform_tabular`` call the aggregate is pre-computed for each target
    column and the resulting scalar participates in per-element arithmetic.

    Example::

        cv_mean = ColumnwiseValue(lambda col: sum(col) / len(col))
        result  = transform_tabular(df, lambda x: x - cv_mean)
    """

    def __init__(self, func):
        self.func = func
        self._value = None

    def _bind(self, values):
        result = self.func(values)
        if isinstance(result, (ColumnwiseValue, ColumnwiseThread)):
            raise ValueError(
                "Mixed compositions of ColumnwiseValue and "
                "ColumnwiseThread are not allowed"
            )
        self._value = result

    @property
    def _v(self):
        return self._value

    def __repr__(self):
        return f"ColumnwiseValue({self._value!r})"


# ---------------------------------------------------------------------------
# ColumnwiseThread
# ---------------------------------------------------------------------------

class ColumnwiseThread(_ArithmeticMixin):
    """Column-level transformation marker.

    Wraps a function ``func(column_as_list) -> list_of_same_length``.  When
    used inside a ``transform_tabular`` call the transformation is pre-computed
    for each target column and each row receives its corresponding element.

    Example::

        ct_sorted = ColumnwiseThread(lambda col: sorted(col))
        result    = transform_tabular(df, lambda x: ct_sorted)
    """

    def __init__(self, func):
        self.func = func
        self._values = None

    def _bind(self, values):
        result = self.func(values)
        if isinstance(result, (ColumnwiseValue, ColumnwiseThread)):
            raise ValueError(
                "Mixed compositions of ColumnwiseValue and "
                "ColumnwiseThread are not allowed"
            )
        self._values = list(result)

    @property
    def _v(self):
        return self._values[_row_index]

    def __repr__(self):
        return f"ColumnwiseThread({self._values!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_markers(func):
    """Return (cvs, cts) lists of markers captured in *func*'s closure."""
    cvs, cts = [], []
    seen = set()
    if hasattr(func, "__closure__") and func.__closure__:
        for cell in func.__closure__:
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            if isinstance(obj, ColumnwiseValue):
                cvs.append(obj)
            elif isinstance(obj, ColumnwiseThread):
                cts.append(obj)
    return cvs, cts


def _resolve_columns(df, columns):
    """Resolve a column specification to a list of column names."""
    all_cols = list(df.columns)
    if columns is None:
        return list(all_cols)
    if isinstance(columns, str):
        return [columns]
    if isinstance(columns, int):
        return [all_cols[columns]]
    if isinstance(columns, slice):
        return all_cols[columns]
    if isinstance(columns, list):
        return [all_cols[c] if isinstance(c, int) else c for c in columns]
    raise TypeError(f"Invalid column specification: {type(columns)}")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def transform_tabular(df_or_func, func_or_columns=None, columns=None):
    """Transform columns of a DataFrame by applying a function.

    Direct forms::

        transform_tabular(df, func)            # all columns
        transform_tabular(df, func, columns)   # selected columns

    Operator forms (return a callable)::

        transform_tabular(func)(df)
        transform_tabular(func, columns)(df)

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame (direct form).
    func : callable
        ``func(x) -> y`` applied element-wise.  May reference
        :class:`ColumnwiseValue` and :class:`ColumnwiseThread` markers.
    columns : column specification, optional
        Which columns to transform.  Accepts:

        * ``None`` -- all columns (default)
        * ``str`` -- single column name
        * ``int`` -- single column by 0-based index
        * ``list[str | int]`` -- multiple columns
        * ``slice`` -- column range

    Returns
    -------
    pandas.DataFrame or callable
    """
    global _row_index

    # ---- operator form: first arg is callable, not a DataFrame ----
    if not isinstance(df_or_func, pd.DataFrame):
        if callable(df_or_func):
            _func, _cols = df_or_func, func_or_columns
            return lambda df: transform_tabular(df, _func, _cols)
        raise TypeError(
            f"First argument must be a DataFrame or callable, "
            f"got {type(df_or_func)}"
        )

    # ---- direct form ----
    df = df_or_func
    func = func_or_columns

    if func is None:
        raise ValueError("Function argument is required")
    if not callable(func):
        raise TypeError(f"Second argument must be callable, got {type(func)}")

    result = df.copy()
    cols = _resolve_columns(df, columns)
    cvs, cts = _find_markers(func)
    has_ct = len(cts) > 0

    for col in cols:
        col_values = df[col].tolist()

        # bind markers to this column
        for cv in cvs:
            cv._bind(col_values)
        for ct in cts:
            ct._bind(col_values)

        # apply func element-wise
        new_values = []
        for i, val in enumerate(col_values):
            if has_ct:
                _row_index = i
            raw = func(val)
            # if func returns a marker object, extract its current value
            if isinstance(raw, (ColumnwiseValue, ColumnwiseThread)):
                raw = raw._v
            new_values.append(raw)

        result[col] = new_values

    return result
