"""Tests for dsagent.agents.graph — the interpret/verify LangGraph loop.

All four scenarios that matter for trustworthiness are covered without
ever touching Groq: a clean run, a self-correcting retry, exhausting the
retry budget without an infinite loop, and an LLM-call exception being
caught rather than crashing the app.
"""

from __future__ import annotations

import pytest

from dsagent.agents.graph import run_reasoning_layer
from dsagent.agents.schemas import Finding, FindingList
from dsagent.analysis.relationships import find_relationships
from dsagent.evidence.store import EvidenceStore
from dsagent.profiling.profile import profile_dataset
from dsagent.quality.checks import check_duplicates, check_missingness, check_outliers


class _SequencedStubLLM:
    """Returns each response in order, repeating the last one if invoked
    more times than there are responses.
    """

    def __init__(self, responses: list[FindingList]):
        self._responses = responses
        self.call_count = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.last_messages = messages
        response = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return response


class _RaisingStubLLM:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise RuntimeError("simulated Groq failure")


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


def _finding(id_, refs):
    return Finding(
        id=id_,
        headline="headline",
        evidence_refs=refs,
        severity="info",
        meaning="meaning",
        next_step="next step",
    )


def test_happy_path_verifies_on_first_attempt(store):
    good = _finding("f1", ["missing.total_missing_cells"])
    llm = _SequencedStubLLM([FindingList(findings=[good])])

    final = run_reasoning_layer(store, llm)

    assert final.error is None
    assert final.attempt == 1
    assert len(final.verification.verified) == 1
    assert final.verification.rejected_finding_ids == []


def test_bad_finding_triggers_retry_that_succeeds(store):
    bad = _finding("bad", ["missing.invented_field"])
    good = _finding("good", ["duplicates.exact_duplicate_rows"])
    llm = _SequencedStubLLM([FindingList(findings=[bad]), FindingList(findings=[good])])

    final = run_reasoning_layer(store, llm, max_attempts=2)

    assert final.attempt == 2
    assert llm.call_count == 2
    assert [v.finding.id for v in final.verification.verified] == ["good"]


def test_exhausting_retries_does_not_loop_forever(store):
    always_bad = _finding("bad", ["missing.invented_field"])
    llm = _SequencedStubLLM([FindingList(findings=[always_bad])])  # same bad answer every call

    final = run_reasoning_layer(store, llm, max_attempts=3)

    assert final.attempt == 3
    assert llm.call_count == 3
    assert final.verification.verified == []
    assert final.verification.rejected_finding_ids == ["bad"]


def test_llm_exception_is_caught_not_raised(store):
    llm = _RaisingStubLLM()

    final = run_reasoning_layer(store, llm)

    assert final.error is not None
    assert "simulated Groq failure" in final.error
    assert final.verification is None


def test_retry_feedback_reaches_second_attempt(store):
    bad = _finding("bad", ["missing.invented_field"])
    good = _finding("good", ["duplicates.exact_duplicate_rows"])
    llm = _SequencedStubLLM([FindingList(findings=[bad]), FindingList(findings=[good])])

    run_reasoning_layer(store, llm, max_attempts=2)

    second_call_text = llm.last_messages[-1].content
    assert "invented_field" in second_call_text or "don't exist" in second_call_text
