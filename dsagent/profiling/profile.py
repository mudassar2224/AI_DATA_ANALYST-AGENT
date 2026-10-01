"""Deterministic dataset and column profiling.

This is the profiler specialist: it infers column types and computes
summary statistics using pandas only. Nothing here calls an LLM — that
keeps every number traceable back to a computation instead of a
generated guess.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_type: str  # "numeric" | "categorical" | "datetime" | "text" | "boolean"
    non_null_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[str]

    # populated only when inferred_type == "numeric"
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_row_count: int
    numeric_columns: int
    categorical_columns: int
    datetime_columns: int
    columns: list[ColumnProfile]


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Anything left (legacy "object" dtype, or pandas' newer dedicated
    # string dtype) is a candidate for text / categorical / a date stored
    # as text. Don't gate on a specific dtype name here — pandas has
    # shipped more than one string representation, and this branch should
    # catch all of them.
    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical"

    sample = non_null.astype(str).head(50)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.8:
        return "datetime"
    if non_null.nunique() / len(non_null) < 0.5:
        return "categorical"
    return "text"


def profile_column(series: pd.Series) -> ColumnProfile:
    non_null = series.dropna()
    inferred = _infer_type(series)

    profile = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        inferred_type=inferred,
        non_null_count=int(non_null.shape[0]),
        missing_count=int(series.isna().sum()),
        missing_pct=round(float(series.isna().mean() * 100), 2),
        unique_count=int(series.nunique(dropna=True)),
        sample_values=[str(v) for v in non_null.head(3).tolist()],
    )

    if inferred == "numeric" and len(non_null) > 0:
        profile.mean = round(float(non_null.mean()), 4)
        profile.std = round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0
        profile.min = round(float(non_null.min()), 4)
        profile.max = round(float(non_null.max()), 4)
        profile.median = round(float(non_null.median()), 4)

    return profile


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    columns = [profile_column(df[col]) for col in df.columns]

    type_counts = {"numeric": 0, "categorical": 0, "datetime": 0}
    for c in columns:
        if c.inferred_type in type_counts:
            type_counts[c.inferred_type] += 1

    return DatasetProfile(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_row_count=int(df.duplicated().sum()),
        numeric_columns=type_counts["numeric"],
        categorical_columns=type_counts["categorical"],
        datetime_columns=type_counts["datetime"],
        columns=columns,
    )
