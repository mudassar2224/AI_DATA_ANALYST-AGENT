"""Tests for dsagent.agents.verifier.

No LLM involved anywhere here — the verifier is pure Python, so these
tests construct Finding objects by hand and check them against a real
EvidenceStore built from the messy_df fixture.
"""

from __future__ import annotations

import pytest

from dsagent.agents.schemas import Finding
from dsagent.agents.verifier import rejection_feedback, verify_findings
from dsagent.analysis.relationships import find_relationships
from dsagent.evidence.store import EvidenceStore
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


def _finding(id_, refs, headline="test headline"):
    return Finding(
        id=id_,
        headline=headline,
        evidence_refs=refs,
        severity="info",
        meaning="test meaning",
        next_step="test next step",
    )


def test_grounded_finding_is_verified(store):
    finding = _finding("f1", ["missing.total_missing_cells"])
    result = verify_findings([finding], store)
    assert len(result.verified) == 1
    assert result.rejected_finding_ids == []
    assert result.verified[0].resolved_values["missing.total_missing_cells"] == "1"


def test_finding_with_no_refs_is_rejected(store):
    finding = _finding("f2", [])
    result = verify_findings([finding], store)
    assert result.verified == []
    assert result.rejected_finding_ids == ["f2"]


def test_finding_with_invented_ref_is_rejected(store):
    finding = _finding("f3", ["missing.this_field_does_not_exist"])
    result = verify_findings([finding], store)
    assert result.verified == []
    assert result.rejected_finding_ids == ["f3"]


def test_finding_with_one_bad_ref_among_good_ones_is_fully_rejected(store):
    # fail-closed: one bad ref rejects the whole finding, not a partial pass
    finding = _finding("f4", ["missing.total_missing_cells", "missing.fake_field"])
    result = verify_findings([finding], store)
    assert result.verified == []
    assert result.rejected_finding_ids == ["f4"]


def test_mixed_batch_keeps_good_drops_bad(store):
    good = _finding("good", ["duplicates.exact_duplicate_rows"])
    bad = _finding("bad", ["nonexistent.field"])
    result = verify_findings([good, bad], store)
    assert [v.finding.id for v in result.verified] == ["good"]
    assert result.rejected_finding_ids == ["bad"]


def test_rejection_feedback_empty_when_nothing_rejected(store):
    finding = _finding("f1", ["missing.total_missing_cells"])
    result = verify_findings([finding], store)
    assert rejection_feedback([finding], result) == ""


def test_rejection_feedback_mentions_bad_finding(store):
    finding = _finding("f3", ["missing.fake_field"], headline="Bogus claim")
    result = verify_findings([finding], store)
    feedback = rejection_feedback([finding], result)
    assert "Bogus claim" in feedback
