"""Columnwise markers: values computed from whole columns, once per transformation.

``ColumnwiseValue`` and ``ColumnwiseThread`` are the counterparts of the Wolfram
Language wrappers of the same name.  A marker wraps a function of a whole column (or
of the whole table); while a transformation runs it stands for the result of that
function, and it can be used in arithmetic and comparisons like an ordinary value.
"""

import contextvars
import math
import operator
import types
from collections.abc import Mapping, Set
from contextlib import contextmanager

import numpy as np
import pandas as pd

__all__ = ["ColumnwiseThread", "ColumnwiseValue"]

_MIXED = "Mixed compositions of ColumnwiseValue and ColumnwiseThread are not allowed"
_UNSET = object()

# The scope of the transformation that is currently running, if any.  A context
# variable keeps nested and concurrent transformations apart.
_FRAME = contextvars.ContextVar("transform_tabular_frame", default=None)


class _Frame:
    """The scope markers are evaluated in: one column, or one table."""

    __slots__ = ("length", "row", "evaluating", "_load", "_data", "_results", "_threads")

    def __init__(self, load, length):
        self.length = length    # number of rows
        self.row = None         # position of the row being transformed
        self.evaluating = 0     # > 0 while the function of a marker is running
        self._load = load
        self._data = _UNSET
        self._results = {}
        self._threads = {}

    @property
    def data(self):
        """What the marker functions receive, loaded when the first marker needs it."""
        if self._data is _UNSET:
            self._data = self._load()
        return self._data

    def resolve(self, marker):
        """The columnwise result of *marker*, evaluated once in this scope."""
        entry = self._results.get(marker._key)
        if entry is None:
            self.evaluating += 1
            try:
                result = marker._evaluate(self)
            finally:
                self.evaluating -= 1
            # Keeping the marker alive keeps the ids its key is made of from being reused.
            entry = self._results[marker._key] = (marker, result)
        return entry[1]

    def thread(self, marker):
        """The column of a ``ColumnwiseThread`` as a list, for row-by-row access."""
        items = self._threads.get(marker._key)
        if items is None:
            column = self.resolve(marker)
            items = column.tolist() if _is_array(column) else list(column)
            self._threads[marker._key] = items
        return items


@contextmanager
def _scope(load, length):
    """Run a transformation whose markers are evaluated on the data given by ``load()``."""
    frame = _Frame(load, length)
    token = _FRAME.set(frame)
    try:
        yield frame
    finally:
        _FRAME.reset(token)


def _is_array(column):
    """Whether *column* is a one-dimensional pandas or NumPy container."""
    return isinstance(column, (pd.Series, pd.Index)) or (
        isinstance(column, np.ndarray) and column.ndim == 1
    )


def _function_key(func):
    """Identify *func* up to re-creation of the same function object.

    Two functions made by evaluating the same ``lambda`` or ``def`` in the same scope
    get the same key, although they are distinct objects.
    """
    if isinstance(func, types.FunctionType):
        return (
            id(func.__code__),
            id(func.__globals__),
            tuple(map(id, func.__closure__ or ())),
            tuple(map(id, func.__defaults__ or ())),
            tuple(sorted((name, id(value)) for name, value in (func.__kwdefaults__ or {}).items())),
        )
    return id(func)


def _value(obj):
    """The current value of a marker; any other object as it is."""
    return obj.value if isinstance(obj, _Columnwise) else obj


class _Columnwise:
    """Base of the markers: evaluation in the running scope, and operators."""

    __slots__ = ("func", "_key")

    # NumPy scalars and arrays leave binary operations to the reflected methods below.
    __array_ufunc__ = None

    def __init__(self, func):
        if not callable(func):
            raise TypeError(
                f"{type(self).__name__} expects a function, got {type(func).__name__}"
            )
        self.func = func
        # A marker written inside a row function is created again for every row.  Such
        # markers are identified by their function, so that it is still evaluated once.
        inline = _FRAME.get() is not None
        self._key = (type(self), _function_key(func) if inline else id(self))

    @property
    def value(self):
        """The value the marker stands for (only while a transformation is running)."""
        frame = _FRAME.get()
        if frame is None:
            raise RuntimeError(
                f"{type(self).__name__} can only be evaluated inside transform_tabular, "
                "transform_columns or transform_rows"
            )
        return self._current(frame)

    def __repr__(self):
        return f"{type(self).__name__}({self.func!r})"

    # ---- arithmetic
    def __add__(self, other): return self.value + _value(other)
    def __radd__(self, other): return other + self.value
    def __sub__(self, other): return self.value - _value(other)
    def __rsub__(self, other): return other - self.value
    def __mul__(self, other): return self.value * _value(other)
    def __rmul__(self, other): return other * self.value
    def __truediv__(self, other): return self.value / _value(other)
    def __rtruediv__(self, other): return other / self.value
    def __floordiv__(self, other): return self.value // _value(other)
    def __rfloordiv__(self, other): return other // self.value
    def __mod__(self, other): return self.value % _value(other)
    def __rmod__(self, other): return other % self.value
    def __divmod__(self, other): return divmod(self.value, _value(other))
    def __rdivmod__(self, other): return divmod(other, self.value)
    def __pow__(self, other): return self.value ** _value(other)
    def __rpow__(self, other): return other ** self.value
    def __neg__(self): return -self.value
    def __pos__(self): return +self.value
    def __abs__(self): return abs(self.value)

    # ---- logic
    def __and__(self, other): return self.value & _value(other)
    def __rand__(self, other): return other & self.value
    def __or__(self, other): return self.value | _value(other)
    def __ror__(self, other): return other | self.value
    def __xor__(self, other): return self.value ^ _value(other)
    def __rxor__(self, other): return other ^ self.value
    def __invert__(self): return ~self.value

    # ---- comparisons
    def __lt__(self, other): return self.value < _value(other)
    def __le__(self, other): return self.value <= _value(other)
    def __gt__(self, other): return self.value > _value(other)
    def __ge__(self, other): return self.value >= _value(other)

    def __eq__(self, other):
        # Outside a transformation a marker is an ordinary object, equal only to itself.
        if _FRAME.get() is None:
            return NotImplemented
        return self.value == _value(other)

    def __ne__(self, other):
        if _FRAME.get() is None:
            return NotImplemented
        return self.value != _value(other)

    __hash__ = object.__hash__

    # ---- conversions
    def __bool__(self): return _FRAME.get() is None or bool(self.value)
    def __int__(self): return int(self.value)
    def __float__(self): return float(self.value)
    def __complex__(self): return complex(self.value)
    def __index__(self): return operator.index(self.value)
    def __round__(self, *ndigits): return round(self.value, *ndigits)
    def __trunc__(self): return math.trunc(self.value)
    def __floor__(self): return math.floor(self.value)
    def __ceil__(self): return math.ceil(self.value)


class ColumnwiseValue(_Columnwise):
    """A value computed once from a whole column, the same for every row.

    ``ColumnwiseValue(func)`` wraps a function that reduces a column to a single value,
    such as a mean or a maximum.  Inside a transformation the marker stands for that
    value, which is computed once and not once per row.

    The function receives the columnwise counterpart of what the row function receives:

    * in :func:`transform_tabular`, where the row function gets one element of a
      column, it receives that column as a list;
    * in :func:`transform_columns`, where the row function gets a row, it receives the
      table (the ``DataFrame``, whose columns span all the rows).

    Examples
    --------
    Subtract from every element the mean of its column::

        mean = ColumnwiseValue(lambda col: sum(col) / len(col))
        transform_tabular(df, lambda x: x - mean)

    Subtract from column ``"a"`` the median of column ``"b"``::

        median_b = ColumnwiseValue(lambda t: t["b"].median())
        transform_columns(df, {"a": lambda r: r["a"] - median_b})

    Markers follow their value in arithmetic and comparisons.  Where a plain value is
    needed, for instance as the argument of another function, use ``marker.value``.
    """

    __slots__ = ()

    def _evaluate(self, frame):
        result = self.func(frame.data)
        if isinstance(result, _Columnwise):
            raise ValueError(_MIXED)
        return result

    def _current(self, frame):
        return frame.resolve(self)

    def _column(self, frame, positions):
        """The value repeated for the rows at *positions* (all rows if ``None``)."""
        count = frame.length if positions is None else len(positions)
        return [frame.resolve(self)] * count


class ColumnwiseThread(_Columnwise):
    """A column computed once from a whole column, and threaded over the rows.

    ``ColumnwiseThread(func)`` wraps a function that turns a column into another column
    of the same length, such as a sort or a cumulative sum.  Inside a transformation
    the marker stands, in each row, for the element of that column in the same
    position.

    The function receives the same argument as the function of a
    :class:`ColumnwiseValue`, and must return one value per row (a list, tuple, NumPy
    array, ``Series``, ...); values are matched to rows by position.

    Examples
    --------
    Replace every column with its cumulative sum::

        from itertools import accumulate

        running = ColumnwiseThread(lambda col: list(accumulate(col)))
        transform_tabular(df, lambda x: running)

    A transformation that consists of the marker alone can be given as the marker
    itself; no function is then called row by row::

        transform_columns(df, {"total": ColumnwiseThread(lambda t: t["a"].cumsum())})
    """

    __slots__ = ()

    def _evaluate(self, frame):
        result = self.func(frame.data)
        if isinstance(result, _Columnwise):
            raise ValueError(_MIXED)
        try:
            if isinstance(result, (str, bytes, Mapping, Set, pd.DataFrame)):
                raise TypeError
            if not isinstance(result, (list, tuple, pd.Series, pd.Index, np.ndarray)):
                result = list(result)
            size = len(result)
        except TypeError:
            raise ValueError(
                f"The function of {self!r} must return one value per row, "
                f"got {type(result).__name__}"
            ) from None
        if size != frame.length:
            raise ValueError(
                f"The result of {self!r} has length {size}, inconsistent with the "
                f"number of rows, {frame.length}"
            )
        return result

    def _current(self, frame):
        if frame.evaluating:
            # Inside the function of another marker there is no current row.
            raise ValueError(_MIXED)
        return frame.thread(self)[frame.row]

    def _column(self, frame, positions):
        """The column at *positions* (all rows if ``None``), without a loop over rows."""
        column = frame.resolve(self)
        if _is_array(column):
            # Positional: an index carried by the result plays no role.
            column = pd.Series(column if isinstance(column, np.ndarray) else column.array)
            return column if positions is None else column.iloc[positions]
        items = frame.thread(self)
        return items if positions is None else [items[position] for position in positions]
