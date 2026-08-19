from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _deterministic_mode(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return np.nan
    modes = values.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else np.nan


def aggregate_history(
    frame: pd.DataFrame,
    *,
    id_col: str = "case_id",
    time_col: str | None = None,
    tie_breaker_col: str | None = None,
    categorical_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create AMEX-style customer aggregates from a history table.

    Order-free aggregates are always available. ``first``, ``last``,
    ``last_minus_first`` and ``last_minus_mean`` are generated only when a
    real time column is explicitly supplied; parquet row order is never used
    as a time proxy.
    """
    if id_col not in frame.columns:
        raise ValueError(f"Missing id column: {id_col}")
    if time_col is not None and time_col not in frame.columns:
        raise ValueError(f"Missing explicit time column: {time_col}")
    if tie_breaker_col is not None and tie_breaker_col not in frame.columns:
        raise ValueError(f"Missing explicit tie-breaker column: {tie_breaker_col}")
    if tie_breaker_col is not None and time_col is None:
        raise ValueError("tie_breaker_col requires an explicit time_col")
    excluded = {id_col, time_col, tie_breaker_col}
    value_cols = [column for column in frame.columns if column not in excluded]
    if not value_cols:
        return frame[[id_col]].drop_duplicates().reset_index(drop=True)

    explicit_categorical = set(categorical_cols or [])
    unknown_categorical = explicit_categorical.difference(value_cols)
    if unknown_categorical:
        raise ValueError(f"Categorical columns are missing: {sorted(unknown_categorical)}")
    numeric_cols = [
        column
        for column in value_cols
        if is_numeric_dtype(frame[column].dtype) and column not in explicit_categorical
    ]
    resolved_categorical_cols = [column for column in value_cols if column not in numeric_cols]
    grouped = frame.groupby(id_col, sort=False, dropna=False)
    result = grouped.size().rename("row_count").to_frame()

    for column in numeric_cols:
        aggregate = grouped[column].agg(["count", "mean", "std", "min", "max", "sum", "nunique"])
        aggregate = aggregate.rename(columns={name: f"{name}_{column}" for name in aggregate.columns})
        result = result.join(aggregate, how="left")
    for column in resolved_categorical_cols:
        aggregate = grouped[column].agg(
            count="count",
            nunique="nunique",
            mode=_deterministic_mode,
        )
        aggregate = aggregate.rename(columns={name: f"{name}_{column}" for name in aggregate.columns})
        result = result.join(aggregate, how="left")

    if time_col is not None:
        sort_cols = [id_col, time_col]
        duplicate_time = frame.duplicated(sort_cols, keep=False)
        if duplicate_time.any() and tie_breaker_col is None:
            raise ValueError(
                "Duplicate id/time rows require tie_breaker_col; input row order is not a time order"
            )
        if tie_breaker_col is not None:
            sort_cols.append(tie_breaker_col)
            if frame.duplicated(sort_cols, keep=False).any():
                raise ValueError("id/time/tie-breaker values must uniquely order history rows")
        ordered = frame.sort_values(sort_cols, kind="mergesort", na_position="last")
        ordered_grouped = ordered.groupby(id_col, sort=False, dropna=False)
        for column in value_cols:
            first = ordered_grouped[column].first().rename(f"first_{column}")
            last = ordered_grouped[column].last().rename(f"last_{column}")
            result = result.join(first, how="left").join(last, how="left")
            if column in numeric_cols:
                result[f"last_minus_first_{column}"] = (
                    result[f"last_{column}"] - result[f"first_{column}"]
                )
                result[f"last_minus_mean_{column}"] = (
                    result[f"last_{column}"] - result[f"mean_{column}"]
                )

    return result.reset_index()
