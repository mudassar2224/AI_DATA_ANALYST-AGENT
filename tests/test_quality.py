"""Tests for dsagent.quality.checks."""

from __future__ import annotations

from dsagent.quality.checks import check_duplicates, check_missingness, check_outliers


def test_duplicate_detection(messy_df):
    report = check_duplicates(messy_df)
    assert report.exact_duplicate_rows == 1


def test_missingness_report(messy_df):
    report = check_missingness(messy_df)
    assert report.total_missing_cells == 1
    assert report.rows_with_any_missing == 1


def test_disguised_missing_detected(messy_df):
    report = check_missingness(messy_df)
    assert "city" in report.columns_with_disguised_missing


def test_no_false_positive_on_clean_column(messy_df):
    report = check_missingness(messy_df)
    assert "id" not in report.columns_with_disguised_missing


def test_outlier_flagged_on_income(messy_df):
    flags = check_outliers(messy_df)
    flagged_columns = [f.column for f in flags]
    assert "income" in flagged_columns
    income_flag = next(f for f in flags if f.column == "income")
    assert income_flag.outlier_count == 2


def test_no_outliers_on_small_sample():
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3]})
    assert check_outliers(df) == []
