# Changelog

## 0.9.0

Two new functions, and an update of `transform_tabular` to the Wolfram Language resource function TransformTabular 1.0.1.

### Added

- `transform_columns`, a port of the built-in Wolfram Language function `TransformColumns`: it modifies existing columns or adds new ones, computing each cell from its row. It supports the removal of columns (`None`), the placement of new columns next to existing ones, and an operator form.
- `transform_rows`, a port of the Wolfram Language resource function `TransformRows` (0.6.1): it transforms only the rows given by position or selected by a condition, with column transformations or with one function for all the cells, and has the option `keep_order`.
- A marker can be given as the transformation itself, as in `transform_columns(df, {"total": ColumnwiseThread(lambda t: t["a"].cumsum())})`. The column is then computed without calling a function row by row.
- Markers have a `value` property, for use where a plain value is needed.
- Markers support `==`, `!=`, `divmod`, `round` and the logical operators.
- A benchmark against the Wolfram Language originals, in `benchmark/`.

### Changed

- `transform_tabular` follows TransformTabular 1.0.1: the function can be any callable, such as `math.sqrt` or a composition, in all its forms.
- In the column selection of `transform_tabular`, an integer that is a column label now selects that column; other integers are positions, as before.
- The result of a `ColumnwiseThread` function must have one value per row: a result of a different length, or one that is not a column, raises a `ValueError`.
- A column label or position that does not exist raises a `KeyError` or an `IndexError` with an explicit message.
- The package now depends explicitly on NumPy.

### Fixed

- `ColumnwiseValue` and `ColumnwiseThread` were recognised only when the transformation function held them in its closure, so the examples of the README failed when run at the top level of a script or of a notebook (`TypeError`). Markers now work wherever they are defined, including inside the transformation function itself.
- Transformations with markers can be nested and can run in several threads at once.

## 0.8.0

First release: `transform_tabular`, `ColumnwiseValue` and `ColumnwiseThread`.
