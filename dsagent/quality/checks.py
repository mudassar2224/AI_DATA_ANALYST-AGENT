"""Deterministic data-quality checks.

This is the quality-auditor specialist: missingness (including disguised
tokens like "N/A" that pandas doesn't treat as null by default), exact
duplicates, and a simple IQR-based outlier flag for numeric columns.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

DISGUISED_MISSING_TOKENS = {"na", "n/a", "unknown", "?", "-", "none", "null", "not available", ""}


class MissingReport(BaseModel):
    total_missing_cells: int
    overall_missing_pct: float
    rows_with_any_missing: int
    by_column_pct: dict[str, float]
    columns_with_disguised_missing: list[str]


class DuplicateReport(BaseModel):
    exact_duplicate_rows: int
    exact_duplicate_pct: float


class OutlierFlag(BaseModel):
    column: str
    outlier_count: int
    outlier_pct: float
    lower_bound: float
    upper_bound: float


def check_missingness(df: pd.DataFrame) -> MissingReport:
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    by_column = (df.isna().mean() * 100).round(2).to_dict()

    disguised: list[str] = []
    for col in df.select_dtypes(include=["object", "str"]).columns:
        lowered = df[col].dropna().astype(str).str.strip().str.lower()
        if lowered.isin(DISGUISED_MISSING_TOKENS).any():
            disguised.append(col)

    return MissingReport(
        total_missing_cells=total_missing,
        overall_missing_pct=round((total_missing / total_cells * 100) if total_cells else 0.0, 2),
        rows_with_any_missing=int(df.isna().any(axis=1).sum()),
        by_column_pct=by_column,
        columns_with_disguised_missing=disguised,
    )


def check_duplicates(df: pd.DataFrame) -> DuplicateReport:
    dup_count = int(df.duplicated().sum())
    return DuplicateReport(
        exact_duplicate_rows=dup_count,
        exact_duplicate_pct=round((dup_count / len(df) * 100) if len(df) else 0.0, 2),
    )


def check_outliers(df: pd.DataFrame) -> list[OutlierFlag]:
    flags: list[OutlierFlag] = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) > 0:
            flags.append(
                OutlierFlag(
                    column=col,
                    outlier_count=len(outliers),
                    outlier_pct=round(len(outliers) / len(series) * 100, 2),
                    lower_bound=round(float(lower), 4),
                    upper_bound=round(float(upper), 4),
                )
            )
    return flags
