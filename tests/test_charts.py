"""Tests for dsagent.viz.charts."""

from __future__ import annotations

import pandas as pd
import pytest

from dsagent.viz.charts import (
    _MAX_CATEGORIES_SHOWN,
    categorical_distribution_chart,
    column_chart,
    missing_values_chart,
    numeric_distribution_chart,
    relationship_chart,
)


def test_missing_values_chart_none_when_nothing_missing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert missing_values_chart(df) is None


def test_missing_values_chart_present_when_missing():
    df = pd.DataFrame({"a": [1, None, 3]})
    fig = missing_values_chart(df)
    assert fig is not None


def test_numeric_distribution_chart_basic():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], name="x")
    fig = numeric_distribution_chart(series, "x")
    assert fig is not None
    assert fig.data[0].type == "histogram"


def test_numeric_distribution_chart_none_for_too_few_values():
    assert numeric_distribution_chart(pd.Series([1.0]), "x") is None


def test_categorical_distribution_chart_caps_at_top_n():
    values = [f"cat_{i}" for i in range(30)]  # 30 distinct, one row each
    series = pd.Series(values)
    fig = categorical_distribution_chart(series, "x", top_n=10)
    assert fig is not None
    # 10 top categories + 1 "Other" bucket
    assert len(fig.data[0].y) == 11


def test_categorical_distribution_chart_no_other_bucket_when_under_top_n():
    series = pd.Series(["a", "a", "b", "c"])
    fig = categorical_distribution_chart(series, "x", top_n=10)
    assert "Other" not in list(fig.data[0].y)


def test_column_chart_dispatches_numeric():
    series = pd.Series([1, 2, 3, 4, 5])
    fig = column_chart(series, "x", "numeric")
    assert fig.data[0].type == "histogram"


def test_column_chart_dispatches_categorical():
    series = pd.Series(["a", "b", "a"])
    fig = column_chart(series, "x", "categorical")
    assert fig.data[0].type == "bar"


def test_column_chart_none_for_text_and_datetime():
    series = pd.Series(["a", "b"])
    assert column_chart(series, "x", "text") is None
    assert column_chart(series, "x", "datetime") is None


@pytest.fixture
def rel_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_a": [1, 2, 3, 4, 5, 6],
            "num_b": [2, 4, 5, 4, 5, 7],
            "cat_a": ["x", "y", "x", "y", "x", "y"],
            "cat_b": ["p", "p", "q", "q", "p", "q"],
        }
    )


def test_relationship_chart_numeric_numeric(rel_df):
    fig = relationship_chart(rel_df, "num_a", "num_b", "numeric_numeric")
    assert fig.data[0].type == "scatter"


def test_relationship_chart_categorical_numeric(rel_df):
    fig = relationship_chart(rel_df, "num_a", "cat_a", "categorical_numeric")
    assert fig.data[0].type == "box"
    assert len(fig.data) == 2  # one box per category level


def test_relationship_chart_categorical_categorical(rel_df):
    fig = relationship_chart(rel_df, "cat_a", "cat_b", "categorical_categorical")
    assert fig.data[0].type == "bar"


def test_relationship_chart_none_for_empty_pair():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    assert relationship_chart(df, "a", "b", "numeric_numeric") is None


def test_relationship_chart_caps_high_cardinality_categorical_categorical():
    # 200 rows, 150 distinct cities and 150 distinct countries -- exactly
    # the shape that produced an unreadable 125-color legend in practice
    n = 200
    df = pd.DataFrame(
        {
            "country": [f"country_{i % 150}" for i in range(n)],
            "city": [f"city_{i % 150}" for i in range(n)],
        }
    )
    fig = relationship_chart(df, "country", "city", "categorical_categorical")
    assert len(fig.data) <= _MAX_CATEGORIES_SHOWN + 1  # + the "Other" bucket
    assert len(fig.data[0].x) <= _MAX_CATEGORIES_SHOWN + 1


def test_relationship_chart_caps_high_cardinality_categorical_numeric():
    n = 200
    df = pd.DataFrame(
        {
            "value": list(range(n)),
            "city": [f"city_{i % 150}" for i in range(n)],
        }
    )
    fig = relationship_chart(df, "value", "city", "categorical_numeric")
    assert len(fig.data) <= _MAX_CATEGORIES_SHOWN + 1
