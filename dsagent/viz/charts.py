"""Deterministic chart selection.

The dispatcher (column_chart) is the adaptive part: it looks at a
column's inferred type and cardinality and picks a chart rule, rather
than drawing the same chart type for every column regardless of what's
actually in it. A high-cardinality categorical column gets top-N + an
"Other" bucket instead of a bar per unique value; a numeric column gets
a histogram sized to its own value count.

relationship_chart does the same for a column pair, keyed off the
relationship_type dsagent.analysis.relationships already computed —
scatter for numeric-numeric, box plot for categorical-numeric, and a
stacked bar for categorical-categorical.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_MAX_CATEGORIES_SHOWN = 10


def missing_values_chart(df: pd.DataFrame) -> go.Figure | None:
    pct = (df.isna().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > 0]
    if pct.empty:
        return None

    fig = go.Figure(go.Bar(x=pct.values, y=pct.index, orientation="h"))
    fig.update_layout(
        title="Missing values by column",
        xaxis_title="% missing",
        yaxis_title=None,
        height=max(240, 28 * len(pct)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def numeric_distribution_chart(series: pd.Series, name: str) -> go.Figure | None:
    values = series.dropna()
    if len(values) < 2:
        return None

    fig = go.Figure(go.Histogram(x=values, nbinsx=min(40, max(10, values.nunique()))))
    fig.update_layout(
        title=f"Distribution of {name}",
        xaxis_title=name,
        yaxis_title="count",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.02,
    )
    return fig


def categorical_distribution_chart(
    series: pd.Series, name: str, top_n: int = _MAX_CATEGORIES_SHOWN
) -> go.Figure | None:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return None

    if len(counts) > top_n:
        top = counts.iloc[:top_n]
        other = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other})])

    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h"))
    fig.update_layout(
        title=f"Top categories in {name}",
        xaxis_title="count",
        yaxis_title=None,
        height=max(240, 28 * len(counts)),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def column_chart(series: pd.Series, name: str, inferred_type: str) -> go.Figure | None:
    """The adaptive rule: pick a chart by type, not a fixed checklist."""
    if inferred_type == "numeric":
        return numeric_distribution_chart(series, name)
    if inferred_type in ("categorical", "boolean"):
        return categorical_distribution_chart(series, name)
    return None  # text/datetime: no chart rule yet, see docs/architecture.md


def _collapse_rare(series: pd.Series, top_n: int = _MAX_CATEGORIES_SHOWN) -> pd.Series:
    """Replace anything outside the top_n most frequent values with "Other",
    so a chart's legend or axis can't explode past what's actually
    readable regardless of the column's real cardinality.
    """
    counts = series.value_counts()
    if len(counts) <= top_n:
        return series
    keep = set(counts.iloc[:top_n].index)
    return series.where(series.isin(keep), other="Other")


def relationship_chart(
    df: pd.DataFrame, column_a: str, column_b: str, relationship_type: str
) -> go.Figure | None:
    paired = df[[column_a, column_b]].dropna()
    if paired.empty:
        return None

    if relationship_type == "numeric_numeric":
        fig = go.Figure(
            go.Scatter(x=paired[column_a], y=paired[column_b], mode="markers", opacity=0.6)
        )
        fig.update_layout(xaxis_title=column_a, yaxis_title=column_b)
    elif relationship_type == "categorical_numeric":
        # by relationships.py's convention, column_a is numeric, column_b categorical
        groups = _collapse_rare(paired[column_b].astype(str))
        fig = go.Figure()
        for level in groups.unique():
            fig.add_trace(go.Box(y=paired.loc[groups == level, column_a], name=level))
        fig.update_layout(yaxis_title=column_a, xaxis_title=column_b, showlegend=False)
    elif relationship_type == "categorical_categorical":
        rows = _collapse_rare(paired[column_a].astype(str))
        cols = _collapse_rare(paired[column_b].astype(str))
        table = pd.crosstab(rows, cols)
        fig = go.Figure()
        for col in table.columns:
            fig.add_trace(go.Bar(name=str(col), x=table.index, y=table[col]))
        fig.update_layout(barmode="stack", xaxis_title=column_a, yaxis_title="count")
    else:
        return None

    fig.update_layout(
        title=f"{column_a} vs {column_b}", height=320, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
