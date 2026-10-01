from __future__ import annotations

import numpy as np
import pandas as pd

from dsagent.modeling.baseline import run_baseline_classification
from dsagent.modeling.triage import triage_dataset


def _classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 120
    signal = rng.normal(size=rows)
    target = (signal > 1.0).astype(int)
    return pd.DataFrame(
        {
            "signal": signal,
            "segment": np.where(signal > 0, "high", "low"),
            "ID": [f"ID_{i}" for i in range(rows)],
            "liquidity_stress_next_30d": target,
        }
    )


def test_triage_detects_binary_target_and_identifier():
    triage = triage_dataset(_classification_df())

    assert triage.task_type == "binary_classification"
    assert triage.target_column == "liquidity_stress_next_30d"
    assert triage.identifier_columns == ["ID"]
    assert triage.target_classes == 2
    assert triage.target_positive_rate is not None
    assert "PR-AUC" in triage.recommended_metrics


def test_triage_without_target_recommends_unsupervised_path():
    df = pd.DataFrame({"x": [1, 2, 3], "category": ["a", "b", "a"]})

    triage = triage_dataset(df)

    assert triage.task_type == "unsupervised_exploration"
    assert triage.target_column is None


def test_baseline_classification_excludes_id_and_reports_metrics():
    results = run_baseline_classification(_classification_df(), "liquidity_stress_next_30d")

    assert [result.model_name for result in results] == [
        "majority baseline",
        "logistic regression",
    ]
    for result in results:
        assert result.feature_count == 2
        assert result.train_rows == 96
        assert result.test_rows == 24
        assert set(result.metrics) == {
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        }
        assert all(value is not None for value in result.metrics.values())
        assert any("identifier" in warning.lower() for warning in result.warnings)
