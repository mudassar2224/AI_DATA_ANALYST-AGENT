"""Tests for dsagent.analysis.relationships.

Uses a synthetic dataset with planted signal (columns that really are
related) and pure noise (columns that aren't), so the tests check that
the FDR-corrected scan finds the real relationship and does NOT flag the
noise pair — not just that it runs without crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsagent.analysis.relationships import find_relationships


@pytest.fixture
def signal_and_noise_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300

    x = rng.normal(0, 1, n)
    y_related = x * 2.0 + rng.normal(0, 0.5, n)  # real relationship with x
    y_noise = rng.normal(0, 1, n)  # unrelated to everything

    group = rng.choice(["A", "B", "C"], size=n)
    # numeric column whose mean genuinely differs by group
    group_offset = {"A": 0.0, "B": 5.0, "C": -5.0}
    numeric_by_group = np.array([group_offset[g] for g in group]) + rng.normal(0, 1, n)

    cat_noise = rng.choice(["X", "Y", "Z"], size=n)  # unrelated to `group`

    return pd.DataFrame(
        {
            "x": x,
            "y_related": y_related,
            "y_noise": y_noise,
            "group": group,
            "numeric_by_group": numeric_by_group,
            "cat_noise": cat_noise,
        }
    )


@pytest.fixture
def column_types() -> dict[str, str]:
    return {
        "x": "numeric",
        "y_related": "numeric",
        "y_noise": "numeric",
        "group": "categorical",
        "numeric_by_group": "numeric",
        "cat_noise": "categorical",
    }


def _find(report, a, b):
    for r in report.results:
        if {r.column_a, r.column_b} == {a, b}:
            return r
    return None


def test_detects_real_numeric_relationship(signal_and_noise_df, column_types):
    report = find_relationships(signal_and_noise_df, column_types)
    result = _find(report, "x", "y_related")
    assert result is not None
    assert result.is_significant is True
    assert result.effect_label in ("moderate", "strong")


def test_does_not_flag_numeric_noise(signal_and_noise_df, column_types):
    report = find_relationships(signal_and_noise_df, column_types)
    result = _find(report, "x", "y_noise")
    assert result is not None
    assert result.is_significant is False


def test_detects_real_group_difference(signal_and_noise_df, column_types):
    report = find_relationships(signal_and_noise_df, column_types)
    result = _find(report, "numeric_by_group", "group")
    assert result is not None
    assert result.is_significant is True


def test_does_not_flag_categorical_noise(signal_and_noise_df, column_types):
    report = find_relationships(signal_and_noise_df, column_types)
    result = _find(report, "group", "cat_noise")
    assert result is not None
    assert result.is_significant is False


def test_target_mode_only_compares_against_target(signal_and_noise_df, column_types):
    report = find_relationships(signal_and_noise_df, column_types, target_column="y_related")
    pairs = {(r.column_a, r.column_b) for r in report.results}
    assert all("y_related" in pair for pair in pairs)
    # target mode is linear in columns, not quadratic
    assert report.pairs_tested == len(column_types) - 1


def test_caps_wide_datasets():
    rng = np.random.default_rng(1)
    n = 50
    wide_df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, n) for i in range(30)})
    types = {c: "numeric" for c in wide_df.columns}
    report = find_relationships(wide_df, types, max_columns=10)
    assert report.was_capped is True
    assert len(report.columns_considered) == 10


def test_no_cap_by_default_scans_every_pair():
    # the old default silently capped at 20 columns; this is the direct
    # regression test that nothing is capped unless explicitly asked for
    rng = np.random.default_rng(2)
    n = 100
    wide_df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, n) for i in range(30)})
    types = {c: "numeric" for c in wide_df.columns}
    report = find_relationships(wide_df, types)
    assert report.was_capped is False
    assert len(report.columns_considered) == 30
    assert report.pairs_tested == 30 * 29 // 2  # C(30, 2), nothing skipped


def test_vectorized_batch_matches_pairwise_computation():
    # the vectorized numeric-numeric path must agree with computing each
    # pair independently, not just "run without crashing"
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, n) for i in range(6)})
    types = {c: "numeric" for c in df.columns}

    batch_report = find_relationships(df, types)

    from scipy import stats as scipy_stats

    for r in batch_report.results:
        expected_r, expected_p = scipy_stats.spearmanr(df[r.column_a], df[r.column_b])
        assert abs(r.metric_value - round(float(expected_r), 4)) < 1e-6
        assert abs(r.p_value - round(float(expected_p), 6)) < 1e-6


def test_missing_values_only_affect_pairs_involving_that_column():
    # a NaN in one column must not poison correlations between two OTHER
    # columns that have no missing data at all -- this is the specific
    # failure mode nan_policy="omit" has to avoid in the batch path
    rng = np.random.default_rng(4)
    n = 300
    df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, n) for i in range(4)})
    df.loc[0:20, "col_0"] = None  # only col_0 has missing data

    types = {c: "numeric" for c in df.columns}
    report = find_relationships(df, types)

    clean_pair = next(r for r in report.results if {r.column_a, r.column_b} == {"col_2", "col_3"})
    from scipy import stats as scipy_stats

    expected_r, _ = scipy_stats.spearmanr(df["col_2"], df["col_3"])
    assert abs(clean_pair.metric_value - round(float(expected_r), 4)) < 1e-6

    affected_pair = next(
        r for r in report.results if {r.column_a, r.column_b} == {"col_0", "col_1"}
    )
    assert affected_pair.metric_value != 0.0  # still computed, not dropped or zeroed out


def test_two_numeric_columns_edge_case():
    # scipy.stats.spearmanr returns bare scalars (not matrices) for
    # exactly 2 variables -- a real API quirk that needs explicit handling
    rng = np.random.default_rng(5)
    n = 100
    x = rng.normal(0, 1, n)
    df = pd.DataFrame({"a": x, "b": x * 2 + rng.normal(0, 0.1, n)})
    types = {"a": "numeric", "b": "numeric"}
    report = find_relationships(df, types)
    assert len(report.results) == 1
    assert report.results[0].effect_label == "strong"


def test_constant_column_does_not_crash():
    rng = np.random.default_rng(6)
    n = 100
    df = pd.DataFrame({"const": [5.0] * n, "normal": rng.normal(0, 1, n)})
    types = {"const": "numeric", "normal": "numeric"}
    report = find_relationships(df, types)  # must not raise
    assert report.pairs_tested >= 0
