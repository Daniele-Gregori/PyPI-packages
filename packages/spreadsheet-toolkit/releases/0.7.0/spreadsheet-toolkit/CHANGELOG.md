# Changelog

## 0.7.0 — 2026-07-15

Complete rewrite following the Wolfram Language sources (`SpreadsheetToolkit` package and the [SpreadsheetTrace](https://resources.wolframcloud.com/FunctionRepository/resources/SpreadsheetTrace) resource function).

### Added

- `spreadsheet_trace` — recursive formula dependency tracing (cell ranges, sheet-prefixed column ranges, cross-sheet and absolute references, `trace_duplicates` option), a faithful port of the Wolfram `SpreadsheetTrace` 1.0.0 kernel, verified against its full `VerificationTest` suite.
- `import_all` — one-call cached import of `(sheets, data, formulas)`, following the Wolfram `Import` conventions: `""` for empty cells, formula strings without the leading `=`, `""` for non-formula cells in the formulas grid.
- `index_to_position` / `position_to_index` — cell reference conversion, mirroring the Wolfram `IndexToPosition` / `PositionToIndex` (absolute references like `$B$10` supported).

### Backward-compatible helpers

The 0.1.x names remain available, consistent with the new API:

- `spreadsheet_index_to_position` — alias of `index_to_position`
- `position_to_spreadsheet_index` — alias of `position_to_index` (takes a `(row, column)` pair)
- `import_sheets(file)` — the sheet names, equivalent to `import_all(file)[0]`
- `import_cells(file, sheet=None, rows=None, columns=None)` — cell values of one sheet (by name or 1-based index; optional inclusive 1-based ranges), with `""` for empty cells
- `import_formulas(file, sheet=None, rows=None, columns=None)` — likewise for formulas, without the leading `=` and with `""` for non-formula cells

### Changed (vs 0.1.0 behavior)

- Empty cells import as `""` instead of `None`.
- `import_formulas` returns formula strings without the leading `=`, and `""` (not the cell value) for non-formula cells.
- `position_to_spreadsheet_index` takes a single `(row, column)` pair instead of two arguments.
- Absolute references (`$A$1`) are accepted by the index conversion functions instead of raising `ValueError`.

## 0.1.0 — 2025

Initial release: `import_sheets`, `import_cells`, `import_formulas`, `spreadsheet_index_to_position`, `position_to_spreadsheet_index`.
