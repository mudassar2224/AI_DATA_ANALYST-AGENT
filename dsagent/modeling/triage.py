"""Deterministic dataset triage for choosing an analysis path."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "unsupervised_exploration",
]

_TARGET_HINTS = (
    "target",
    "label",
    "outcome",
    "churn",
    "default",
    "fraud",
    "risk",
    "stress",
    "response",
)


class DatasetTriage(BaseModel):
    """A deterministic recommendation for the next analysis family."""

    task_type: TaskType
    target_column: str | None = None
    identifier_columns: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    target_classes: int | None = None
    target_positive_rate: float | None = None
    recommended_metrics: list[str] = Field(default_factory=list)
    rationale: str


def infer_target_column(df: pd.DataFrame) -> str | None:
    """Infer a likely target only from a small-cardinality, meaningful name."""
    candidates: list[tuple[int, str]] = []
    for column in df.columns:
        name = str(column)
        lowered = name.lower()
        if lowered == "id" or lowered.endswith("_id"):
            continue
        series = df[column].dropna()
        if series.empty or series.nunique() < 2 or series.nunique() > 20:
            continue
        score = sum(2 for hint in _TARGET_HINTS if hint in lowered)
        if score:
            candidates.append((score, name))
    return max(candidates, default=(0, ""))[1] or None


def _identifier_columns(df: pd.DataFrame) -> list[str]:
    identifiers: list[str] = []
    for column in df.columns:
        name = str(column)
        lowered = name.lower()
        if lowered == "id" or lowered.endswith("_id"):
            identifiers.append(name)
    return identifiers


def triage_dataset(df: pd.DataFrame, target_column: str | None = None) -> DatasetTriage:
    """Classify a dataframe and recommend metrics without using an LLM."""
    target = target_column if target_column in df.columns else infer_target_column(df)
    identifiers = _identifier_columns(df)
    feature_columns = [str(c) for c in df.columns if c != target and c not in identifiers]

    if target is None:
        return DatasetTriage(
            task_type="unsupervised_exploration",
            identifier_columns=identifiers,
            feature_columns=feature_columns,
            recommended_metrics=[
                "silhouette score",
                "cluster stability",
                "reconstruction error",
            ],
            rationale=(
                "No reliable target column was detected; begin with clustering, "
                "anomaly detection, or dimensionality reduction."
            ),
        )

    values = df[target].dropna()
    classes = int(values.nunique())
    if classes <= 2:
        task_type: TaskType = "binary_classification"
        positive_rate = float((values == values.value_counts().index[-1]).mean())
        metrics = ["PR-AUC", "ROC-AUC", "balanced accuracy", "precision", "recall", "F1"]
        rationale = (
            "The target has two classes; use stratified validation and imbalance-aware metrics."
        )
    elif classes <= 20 and (not pd.api.types.is_numeric_dtype(values) or classes < 10):
        task_type = "multiclass_classification"
        positive_rate = None
        metrics = ["macro F1", "balanced accuracy", "per-class recall", "confusion matrix"]
        rationale = (
            "The target has a small number of classes; use stratified multiclass evaluation."
        )
    elif pd.api.types.is_numeric_dtype(values):
        task_type = "regression"
        positive_rate = None
        metrics = ["MAE", "RMSE", "R²", "residual diagnostics"]
        rationale = (
            "The target is numeric with many distinct values; treat it as a regression target."
        )
    else:
        task_type = "unsupervised_exploration"
        positive_rate = None
        metrics = ["cluster stability", "silhouette score", "human validation"]
        rationale = (
            "The target is not suitable for the current supervised baseline; "
            "inspect the data structure first."
        )

    return DatasetTriage(
        task_type=task_type,
        target_column=target,
        identifier_columns=identifiers,
        feature_columns=feature_columns,
        target_classes=classes,
        target_positive_rate=round(positive_rate, 4) if positive_rate is not None else None,
        recommended_metrics=metrics,
        rationale=rationale,
    )
