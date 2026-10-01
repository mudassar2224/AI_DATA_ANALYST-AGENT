"""Deterministic relationship analysis: correlation, association, and
group-difference tests between column pairs, with multiple-testing
correction so scanning many pairs doesn't manufacture false "insights."

Numeric vs numeric (Spearman), categorical vs categorical (Cramer's V),
and categorical vs numeric (eta-squared from a one-way ANOVA). Every
p-value across the whole scan goes through Benjamini-Hochberg FDR
correction before anything is reported as significant.

No column count is capped by default — every pair gets tested. That's
only practical because the numeric-numeric case (usually the overwhelming
majority of pairs in a wide dataset) is computed as a single vectorized
scipy call across the whole numeric matrix at once, not a Python loop
over individual pairs: ~200x faster in testing (178 columns, 15,753
pairs: 10.3s looped vs 0.05s vectorized), which is what makes "just
compute all of it" a reasonable default instead of a performance trap.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Effect-size labels use standard, well-established conventions (Cohen 1988
# for eta-squared and correlation; the common Cramer's V interpretation
# table). They're rules of thumb, not universal cutoffs.
_CORR_BANDS = [(0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"), (1.01, "strong")]
_CRAMERS_V_BANDS = [(0.1, "negligible"), (0.2, "weak"), (0.4, "moderate"), (1.01, "strong")]
_ETA_SQ_BANDS = [(0.01, "negligible"), (0.06, "weak"), (0.14, "moderate"), (1.01, "strong")]
_MIN_PAIR_SAMPLE = 5


def _label(value: float, bands: list[tuple[float, str]]) -> str:
    abs_value = abs(value)
    for threshold, label in bands:
        if abs_value < threshold:
            return label
    return bands[-1][1]


class RelationshipResult(BaseModel):
    column_a: str
    column_b: str
    relationship_type: str  # "numeric_numeric" | "categorical_categorical" | "categorical_numeric"
    metric_name: str  # "spearman_r" | "cramers_v" | "eta_squared"
    metric_value: float
    p_value: float
    adjusted_p_value: float
    effect_label: str
    is_significant: bool


class RelationshipReport(BaseModel):
    results: list[RelationshipResult]
    columns_considered: list[str]
    pairs_tested: int
    was_capped: bool
    alpha: float


def _spearman_pair(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """One pair at a time — used only in target-column mode, where the
    cost is already linear in column count and a full matrix would waste
    time computing every other-column-to-other-column pair nobody asked
    for.
    """
    combined = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(combined) < _MIN_PAIR_SAMPLE:
        return 0.0, 1.0
    r, p = stats.spearmanr(combined["a"], combined["b"])
    return float(r), float(p)


def _spearman_all_pairs(
    df: pd.DataFrame, numeric_cols: list[str]
) -> dict[tuple[str, str], tuple[float, float]]:
    """Every numeric-numeric pair's (r, p), computed as one vectorized
    scipy call across the whole numeric matrix instead of a Python loop.
    nan_policy="omit" makes this per-pair correct despite being computed
    together: a missing value in one column only affects pairs involving
    that column, not the whole matrix (verified against direct pairwise
    computation — see tests/test_relationships.py).
    """
    if len(numeric_cols) < 2:
        return {}

    matrix = df[numeric_cols].to_numpy(dtype=float)
    valid = ~np.isnan(matrix)
    # pairwise shared-valid-row counts via one matrix multiply, so a
    # sparse pair (few rows where BOTH columns are present) can still be
    # filtered out below without a second per-pair loop
    valid_counts = valid.T.astype(int) @ valid.astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
        result = stats.spearmanr(matrix, nan_policy="omit")

    if len(numeric_cols) == 2:
        # scipy returns bare scalars, not matrices, for exactly 2 variables
        corr = np.array([[1.0, result.correlation], [result.correlation, 1.0]])
        pval = np.array([[0.0, result.pvalue], [result.pvalue, 0.0]])
    else:
        corr, pval = result.correlation, result.pvalue

    pairs: dict[tuple[str, str], tuple[float, float]] = {}
    for i, j in combinations(range(len(numeric_cols)), 2):
        if valid_counts[i, j] < _MIN_PAIR_SAMPLE:
            continue
        r, p = corr[i, j], pval[i, j]
        if np.isnan(r) or np.isnan(p):
            continue
        pairs[(numeric_cols[i], numeric_cols[j])] = (float(r), float(p))
    return pairs


def _cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    combined = pd.DataFrame({"a": a, "b": b}).dropna()
    if combined.empty:
        return 0.0, 1.0
    table = pd.crosstab(combined["a"], combined["b"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    denom = min(k - 1, r - 1)
    v = float((phi2 / denom) ** 0.5) if denom > 0 else 0.0
    return v, float(p)


def _eta_squared(numeric: pd.Series, groups: pd.Series) -> tuple[float, float]:
    combined = pd.DataFrame({"y": numeric, "g": groups}).dropna()
    grouped = combined.groupby("g", observed=True)["y"]
    sizes = grouped.size()
    if len(sizes) < 2 or (sizes < 2).any():
        return 0.0, 1.0
    samples = [s.values for _, s in grouped]
    _, p = stats.f_oneway(*samples)
    grand_mean = combined["y"].mean()
    ss_between = float((sizes * (grouped.mean() - grand_mean) ** 2).sum())
    ss_total = float(((combined["y"] - grand_mean) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    return eta_sq, float(p)


def find_relationships(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target_column: str | None = None,
    alpha: float = 0.05,
    max_columns: int | None = None,
) -> RelationshipReport:
    """Scan column pairs for associations.

    If target_column is given, only that column is compared against every
    other column (linear cost, computed pair by pair). Otherwise every
    pair among all usable columns is tested by default (max_columns=None) —
    numeric-numeric pairs via one vectorized call regardless of count,
    categorical pairs via a loop (cheap in practice: real datasets rarely
    have enough categorical columns for O(cols^2) categorical pairs to be
    the bottleneck; see tests/test_relationships.py for the scale this was
    checked against). Pass max_columns to cap the columns considered if a
    dataset is wide enough on the categorical side specifically to matter.
    """
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    cat_cols = [c for c, t in column_types.items() if t in ("categorical", "boolean")]
    usable = [c for c in df.columns if c in numeric_cols or c in cat_cols]

    is_target_mode = bool(target_column and target_column in usable)
    was_capped = False

    if is_target_mode:
        others = [c for c in usable if c != target_column]
        pairs = [(target_column, c) for c in others]
    else:
        if max_columns is not None and len(usable) > max_columns:
            usable = usable[:max_columns]
            was_capped = True
        pairs = list(combinations(usable, 2))

    raw: list[tuple[str, str, str, str, float, float]] = []

    if is_target_mode:
        for a, b in pairs:
            type_a = "numeric" if a in numeric_cols else "categorical"
            type_b = "numeric" if b in numeric_cols else "categorical"
            if type_a == "numeric" and type_b == "numeric":
                value, p = _spearman_pair(df[a], df[b])
                raw.append((a, b, "numeric_numeric", "spearman_r", value, p))
            elif type_a == "categorical" and type_b == "categorical":
                value, p = _cramers_v(df[a], df[b])
                raw.append((a, b, "categorical_categorical", "cramers_v", value, p))
            else:
                numeric_col, cat_col = (a, b) if type_a == "numeric" else (b, a)
                value, p = _eta_squared(df[numeric_col], df[cat_col])
                raw.append((numeric_col, cat_col, "categorical_numeric", "eta_squared", value, p))
    else:
        numeric_in_scope = [c for c in usable if c in numeric_cols]
        cat_in_scope = [c for c in usable if c in cat_cols]

        numeric_results = _spearman_all_pairs(df, numeric_in_scope)
        for (a, b), (value, p) in numeric_results.items():
            raw.append((a, b, "numeric_numeric", "spearman_r", value, p))

        for a, b in combinations(cat_in_scope, 2):
            value, p = _cramers_v(df[a], df[b])
            raw.append((a, b, "categorical_categorical", "cramers_v", value, p))

        for num_col in numeric_in_scope:
            for cat_col in cat_in_scope:
                value, p = _eta_squared(df[num_col], df[cat_col])
                raw.append((num_col, cat_col, "categorical_numeric", "eta_squared", value, p))

    if not raw:
        return RelationshipReport(
            results=[],
            columns_considered=usable,
            pairs_tested=0,
            was_capped=was_capped,
            alpha=alpha,
        )

    p_values = [r[5] for r in raw]
    rejected, adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    band_map = {
        "spearman_r": _CORR_BANDS,
        "cramers_v": _CRAMERS_V_BANDS,
        "eta_squared": _ETA_SQ_BANDS,
    }

    results = []
    for (a, b, rel_type, metric_name, value, p), adj_p, sig in zip(
        raw, adjusted, rejected, strict=True
    ):
        label = _label(value, band_map[metric_name])
        results.append(
            RelationshipResult(
                column_a=a,
                column_b=b,
                relationship_type=rel_type,
                metric_name=metric_name,
                metric_value=round(value, 4),
                p_value=round(float(p), 6),
                adjusted_p_value=round(float(adj_p), 6),
                effect_label=label,
                is_significant=bool(sig) and label != "negligible",
            )
        )

    results.sort(key=lambda r: (-r.is_significant, -abs(r.metric_value)))

    return RelationshipReport(
        results=results,
        columns_considered=usable,
        pairs_tested=len(raw),
        was_capped=was_capped,
        alpha=alpha,
    )
