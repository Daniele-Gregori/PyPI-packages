"""
Core implementation of associate_columns.

Translates the Wolfram Language ResourceFunction "AssociateColumns" into Python.
Operates on pandas DataFrames and lists of dicts.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ColSpec = Union[str, List[str]]  # single column name or list of column names
MergeFunc = Callable[[list], Any]
TabularInput = Union[pd.DataFrame, List[Dict[str, Any]]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dataframe(tab: TabularInput) -> pd.DataFrame:
    """Normalise input to a DataFrame."""
    if isinstance(tab, pd.DataFrame):
        return tab
    if isinstance(tab, list) and all(isinstance(r, dict) for r in tab):
        return pd.DataFrame(tab)
    raise TypeError(
        f"Expected a pandas DataFrame or a list of dicts, got {type(tab).__name__}"
    )


def _extract_column(df: pd.DataFrame, col: ColSpec) -> list:
    """Extract a single column or zip multiple columns into tuples."""
    if isinstance(col, str):
        return df[col].tolist()
    # multiple columns → list of tuples
    return list(df[col].itertuples(index=False, name=None))


def _extract_values(df: pd.DataFrame, col: ColSpec) -> list:
    """Extract value column(s). Single col → scalars, multi col → lists."""
    if isinstance(col, str):
        return df[col].tolist()
    return [list(row) for row in df[col].itertuples(index=False, name=None)]


def _hashable(x):
    """Make a value usable as a dict key (tuples are fine, lists are not)."""
    if isinstance(x, list):
        return tuple(x)
    return x


# ---------------------------------------------------------------------------
# Core: two-level association (keys → values)
# ---------------------------------------------------------------------------

def _associate_core(
    keys: list,
    values: list,
    merge: Optional[MergeFunc] = None,
    duplicates_warning: bool = True,
) -> dict:
    """Build a dict from parallel key/value lists, optionally merging duplicates."""
    seen = defaultdict(list)
    order = []
    for k, v in zip(keys, values):
        hk = _hashable(k)
        if hk not in seen:
            order.append(hk)
        seen[hk].append(v)

    has_dups = any(len(v) > 1 for v in seen.values())

    if not has_dups:
        # no duplicates → flat dict
        return {k: seen[k][0] for k in order}

    if merge is None:
        if duplicates_warning:
            dup_keys = [k for k in order if len(seen[k]) > 1]
            warnings.warn(
                f"Duplicate keys found: {dup_keys}. "
                "Values are collected into lists (Identity merge). "
                "Pass a merge function or set duplicates_warning=False to silence this.",
                stacklevel=3,
            )
        merge = list  # identity merge: keep the list as-is

    return {k: merge(seen[k]) for k in order}


# ---------------------------------------------------------------------------
# Recursive nesting (3+ column levels)
# ---------------------------------------------------------------------------

def _nest_associate(
    df: pd.DataFrame,
    cols: List[ColSpec],
    merges: List[Optional[MergeFunc]],
    duplicates_warning: bool,
) -> dict:
    """Recursively build nested dicts from a chain of column specs."""
    if len(cols) == 2:
        # base case: key → value
        keys = _extract_column(df, cols[0])
        values = _extract_values(df, cols[1])
        return _associate_core(keys, values, merges[0], duplicates_warning)

    # recursive case: group by first col, then recurse on the rest
    key_col = cols[0]
    rest_cols = cols[1:]
    rest_merges = merges[1:]
    merge_this_level = merges[0]

    keys = _extract_column(df, key_col)
    groups = defaultdict(list)
    order = []
    for i, k in enumerate(keys):
        hk = _hashable(k)
        if hk not in groups:
            order.append(hk)
        groups[hk].append(i)

    result = {}
    for hk in order:
        indices = groups[hk]
        sub_df = df.iloc[indices].reset_index(drop=True)
        inner = _nest_associate(sub_df, rest_cols, rest_merges, duplicates_warning)
        if merge_this_level is not None:
            # apply merge to the inner dict (e.g. sort keys)
            inner = _apply_dict_merge(inner, merge_this_level)
        result[hk] = inner

    return result


def _apply_dict_merge(d: dict, merge: MergeFunc) -> dict:
    """Apply a merge function that operates on the inner dict.

    `sorted` → sort the dict by keys.
    Other callables are applied to the dict directly.
    """
    if merge is sorted:
        return dict(sorted(d.items()))
    # For general callables, apply to the dict
    try:
        return merge(d)
    except TypeError:
        return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_cols_arg(cols) -> List[ColSpec]:
    """Parse the cols argument into a flat list of ColSpec.

    Accepts:
        - ("A", "B")             → ["A", "B"]
        - ("A", "B", "C")       → ["A", "B", "C"]
        - (["A","B"], "C")      → [["A","B"], "C"]
        - (["A","B"], ["C","D"]) → [["A","B"], ["C","D"]]
    """
    if isinstance(cols, (list, tuple)):
        return list(cols)
    raise TypeError(f"cols must be a list or tuple, got {type(cols).__name__}")


def associate_columns(
    tab: TabularInput,
    cols: Union[Sequence[ColSpec], Tuple[ColSpec, ...]],
    merge: Union[MergeFunc, List[Optional[MergeFunc]], None] = None,
    *,
    duplicates_warning: bool = True,
) -> dict:
    """Create nested dictionaries from columns of a tabular object.

    Parameters
    ----------
    tab : DataFrame or list of dicts
        The input tabular data.
    cols : sequence of column specs
        Each element is either a column name (str) or a list of column names.
        With 2 elements: ``cols[0]`` → keys, ``cols[1]`` → values.
        With 3+ elements: creates nested dicts, one level per element.
    merge : callable, list of callables, or None
        Merging function(s) for duplicate keys.
        - ``None``: auto-merge with ``list`` (identity), warns if duplicates exist.
        - A single callable: applied at every nesting level where duplicates occur.
        - A list of callables: one per nesting transition (length = len(cols) - 1).
          For the innermost (leaf) level, the callable merges duplicate values.
          For intermediate levels, it transforms the inner dict (e.g. ``sorted``
          to sort by keys).
    duplicates_warning : bool
        If True (default), warns when duplicate keys are found and no merge
        function is provided.

    Returns
    -------
    dict
        A (possibly nested) dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> from col2dict import associate_columns
    >>> df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})
    >>> associate_columns(df, ("Name", "Age"))
    {'Alice': 30, 'Bob': 25}

    >>> df = pd.DataFrame({"Year": [2020, 2020, 2021], "Name": ["A", "B", "C"], "Score": [90, 85, 92]})
    >>> associate_columns(df, ("Year", "Name", "Score"))
    {2020: {'A': 90, 'B': 85}, 2021: {'C': 92}}

    >>> associate_columns(df, ("Year", "Name"), merge=sorted)
    {2020: ['A', 'B'], 2021: ['C']}
    """
    df = _to_dataframe(tab)
    col_list = _parse_cols_arg(cols)

    if len(col_list) < 2:
        raise ValueError("cols must contain at least 2 elements (keys and values).")

    # Normalise merge into a list of length (len(cols) - 1)
    n_levels = len(col_list) - 1
    if merge is None:
        merge_list: List[Optional[MergeFunc]] = [None] * n_levels
    elif callable(merge) and not isinstance(merge, list):
        merge_list = [merge] * n_levels
    elif isinstance(merge, (list, tuple)):
        merge_list = list(merge)
        if len(merge_list) < n_levels:
            merge_list.extend([None] * (n_levels - len(merge_list)))
    else:
        raise TypeError(f"merge must be callable, list of callables, or None; got {type(merge).__name__}")

    if len(col_list) == 2:
        keys = _extract_column(df, col_list[0])
        values = _extract_values(df, col_list[1])
        return _associate_core(keys, values, merge_list[0], duplicates_warning)

    return _nest_associate(df, col_list, merge_list, duplicates_warning)
