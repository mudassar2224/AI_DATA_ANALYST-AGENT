"""Tests for the QA graph in dsagent.agents.graph — mirrors
tests/test_agents_graph.py's coverage (happy path, retry, exhaustion,
exception handling) for the question-answering loop.
"""

from __future__ import annotations

import pytest

from dsagent.agents.graph import run_qa
from dsagent.agents.qa import QAAnswer
from dsagent.analysis.relationships import find_relationships
from dsagent.evidence.store import EvidenceStore
from dsagent.profiling.profile import profile_dataset
from dsagent.quality.checks import check_duplicates, check_missingness, check_outliers


class _SequencedStubLLM:
    def __init__(self, responses: list[QAAnswer]):
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


def test_happy_path(store):
    good = QAAnswer(
        answer="One duplicate row.",
        evidence_refs=["duplicates.exact_duplicate_rows"],
        confidence="grounded",
    )
    llm = _SequencedStubLLM([good])

    final = run_qa("Are there duplicates?", store, llm)

    assert final.error is None
    assert final.attempt == 1
    assert final.verified.is_grounded is True


def test_cannot_answer_does_not_trigger_retry(store):
    unanswerable = QAAnswer(answer="Not covered.", evidence_refs=[], confidence="cannot_answer")
    llm = _SequencedStubLLM([unanswerable])

    final = run_qa("What's the capital of France?", store, llm)

    assert final.attempt == 1
    assert llm.call_count == 1
    assert final.verified.is_grounded is True
    assert final.verified.answer.confidence == "cannot_answer"


def test_bad_ref_triggers_retry_that_succeeds(store):
    bad = QAAnswer(answer="x", evidence_refs=["fake.field"], confidence="grounded")
    good = QAAnswer(
        answer="One duplicate row.",
        evidence_refs=["duplicates.exact_duplicate_rows"],
        confidence="grounded",
    )
    llm = _SequencedStubLLM([bad, good])

    final = run_qa("Are there duplicates?", store, llm, max_attempts=2)

    assert final.attempt == 2
    assert final.verified.is_grounded is True


def test_exhausting_retries_does_not_loop_forever(store):
    always_bad = QAAnswer(answer="x", evidence_refs=["fake.field"], confidence="grounded")
    llm = _SequencedStubLLM([always_bad])

    final = run_qa("Are there duplicates?", store, llm, max_attempts=3)

    assert final.attempt == 3
    assert llm.call_count == 3
    assert final.verified.is_grounded is False


def test_llm_exception_is_caught_not_raised(store):
    llm = _RaisingStubLLM()

    final = run_qa("Are there duplicates?", store, llm)

    assert final.error is not None
    assert "simulated Groq failure" in final.error
    assert final.verified is None
