"""Tests for dsagent.profiling.profile."""

from __future__ import annotations

from dsagent.profiling.profile import profile_column, profile_dataset


def test_profile_dataset_counts(messy_df):
    profile = profile_dataset(messy_df)
    assert profile.row_count == 9
    assert profile.column_count == 4
    assert profile.duplicate_row_count == 1


def test_numeric_column_gets_stats(messy_df):
    profile = profile_column(messy_df["income"])
    assert profile.inferred_type == "numeric"
    assert profile.mean is not None
    assert profile.missing_count == 0


def test_categorical_column_type(messy_df):
    profile = profile_column(messy_df["city"])
    assert profile.inferred_type == "categorical"


def test_missing_count_detected(messy_df):
    profile = profile_column(messy_df["age"])
    assert profile.missing_count == 1
    assert profile.missing_pct > 0


def test_datetime_column_detected():
    import pandas as pd

    series = pd.Series(["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10"], name="signup")
    profile = profile_column(series)
    assert profile.inferred_type == "datetime"
