"""Tests for dsagent.evidence.store — the dotted-path resolver that the
verifier depends on to check LLM-cited references against real data.
"""

from __future__ import annotations

import pytest

from dsagent.analysis.relationships import find_relationships
from dsagent.evidence.store import EvidenceResolutionError, EvidenceStore
from dsagent.profiling.profile import profile_dataset
from dsagent.quality.checks import check_duplicates, check_missingness, check_outliers


@pytest.fixture
def store(messy_df) -> EvidenceStore:
    profile = profile_dataset(messy_df)
    column_types = {c.name: c.inferred_type for c in profile.columns}
    return EvidenceStore(
        profile=profile,
        missing=check_missingness(messy_df),
        duplicates=check_duplicates(messy_df),
        outliers=check_outliers(messy_df),
        relationships=find_relationships(messy_df, column_types),
    )


def test_resolves_simple_attribute(store):
    assert store.resolve("missing.total_missing_cells") == 1


def test_resolves_nested_dict(store):
    by_col = store.resolve("missing.by_column_pct")
    assert isinstance(by_col, dict)
    assert "age" in by_col


def test_resolves_dict_key(store):
    pct = store.resolve("missing.by_column_pct.age")
    assert isinstance(pct, float)
    assert pct > 0


def test_resolves_list_item_by_bracket(store):
    flag = store.resolve("outliers[income]")
    assert flag.column == "income"
    assert flag.outlier_count == 2


def test_resolves_duplicate_report_field(store):
    assert store.resolve("duplicates.exact_duplicate_rows") == 1


def test_unknown_field_raises(store):
    with pytest.raises(EvidenceResolutionError):
        store.resolve("missing.nonexistent_field")


def test_unknown_bracket_target_raises(store):
    with pytest.raises(EvidenceResolutionError):
        store.resolve("outliers[not_a_real_column]")


def test_bracket_on_non_list_raises(store):
    with pytest.raises(EvidenceResolutionError):
        store.resolve("duplicates[income]")


def test_resolves_column_field_via_chained_bracket(store):
    # profile.columns[<name>].<field> — bracket lookup followed by a field
    # access on the result, which is exactly the pattern the interpreter's
    # prompt tells the LLM to use for citing a specific column stat.
    pct = store.resolve("profile.columns[age].missing_pct")
    assert isinstance(pct, float)
    assert pct > 0


def test_resolves_relationship_by_two_column_names(store):
    # relationships.results[<col_a>,<col_b>] — the two-value bracket form.
    first = store.relationships.results[0]
    ref = f"relationships.results[{first.column_a},{first.column_b}]"
    result = store.resolve(ref)
    assert {result.column_a, result.column_b} == {first.column_a, first.column_b}
