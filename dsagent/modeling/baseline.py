"""Small, deterministic, leakage-aware baseline classification models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class BaselineModelResult(BaseModel):
    """Metrics and metadata from one baseline model."""

    model_name: str
    metrics: dict[str, float | None]
    train_rows: int
    test_rows: int
    feature_count: int
    warnings: list[str] = Field(default_factory=list)


def run_baseline_classification(
    df: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[BaselineModelResult]:
    """Train a majority baseline and logistic regression for a binary target.

    This is deliberately a baseline, not an automatic production model. It
    drops obvious identifiers, uses a stratified holdout, and reports metrics
    that remain meaningful when the positive class is imbalanced.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    clean = df.dropna(subset=[target_column]).copy()
    target_values = clean[target_column]
    if target_values.nunique() != 2:
        raise ValueError("The baseline classifier requires exactly two target classes.")

    y, labels = pd.factorize(target_values, sort=True)
    if len(labels) != 2:
        raise ValueError("The target must contain two usable classes.")

    identifier_columns = [
        column
        for column in clean.columns
        if str(column).lower() == "id" or str(column).lower().endswith("_id")
    ]
    feature_frame = clean.drop(columns=[target_column, *identifier_columns], errors="ignore")
    supported = feature_frame.select_dtypes(
        include=["number", "bool", "object", "string", "category"]
    )
    if supported.shape[1] == 0:
        raise ValueError("No supported numeric or categorical feature columns were found.")

    numeric_columns = supported.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in supported.columns if column not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    x_train, x_test, y_train, y_test = train_test_split(
        supported,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = ColumnTransformer(transformers)
    prevalence = float(y.mean())
    warnings: list[str] = []
    if min(prevalence, 1 - prevalence) < 0.2:
        warnings.append("The target is imbalanced; accuracy alone is not sufficient.")
    if identifier_columns:
        warnings.append("Identifier-like columns were excluded to reduce leakage risk.")

    def metrics_for(model: Any) -> dict[str, float | None]:
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        return {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        }

    results: list[BaselineModelResult] = []
    for name, classifier in (
        ("majority baseline", DummyClassifier(strategy="most_frequent")),
        (
            "logistic regression",
            LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear"),
        ),
    ):
        model = Pipeline([("preprocess", preprocessor), ("model", classifier)])
        model.fit(x_train, y_train)
        results.append(
            BaselineModelResult(
                model_name=name,
                metrics=metrics_for(model),
                train_rows=len(x_train),
                test_rows=len(x_test),
                feature_count=supported.shape[1],
                warnings=warnings,
            )
        )
    return results
