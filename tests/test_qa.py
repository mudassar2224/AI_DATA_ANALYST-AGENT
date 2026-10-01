"""Tests for dsagent.agents.qa. Stub LLM only, same pattern as
tests/test_interpreter.py — no network call anywhere here.
"""

from __future__ import annotations

import pytest

from dsagent.agents.qa import QAAnswer, answer_question, verify_answer
from dsagent.analysis.relationships import find_relationships
from dsagent.evidence.store import EvidenceStore
from dsagent.profiling.profile import profile_dataset
from dsagent.quality.checks import check_duplicates, check_missingness, check_outliers


class _StubLLM:
    def __init__(self, response):
        self._response = response
        self.last_messages = None

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.last_messages = messages
        return self._response


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


def test_answer_question_returns_pydantic_response(store):
    answer = QAAnswer(
        answer="Age has some missing values.",
        evidence_refs=["missing.by_column_pct.age"],
        confidence="grounded",
    )
    stub = _StubLLM(answer)
    result = answer_question("Does age have missing values?", store, stub)
    assert result.confidence == "grounded"
    assert result.evidence_refs == ["missing.by_column_pct.age"]


def test_answer_question_handles_dict_response(store):
    stub = _StubLLM(
        {
            "answer": "Not covered by the evidence.",
            "evidence_refs": [],
            "confidence": "cannot_answer",
        }
    )
    result = answer_question("What's the weather?", store, stub)
    assert result.confidence == "cannot_answer"


def test_question_appears_in_prompt(store):
    stub = _StubLLM(QAAnswer(answer="x", evidence_refs=[], confidence="cannot_answer"))
    answer_question("Is there a duplicate row?", store, stub)
    assert "Is there a duplicate row?" in stub.last_messages[-1].content


def test_verify_grounded_answer(store):
    answer = QAAnswer(
        answer="There is one duplicate row.",
        evidence_refs=["duplicates.exact_duplicate_rows"],
        confidence="grounded",
    )
    verified = verify_answer(answer, store)
    assert verified.is_grounded is True
    assert verified.resolved_values["duplicates.exact_duplicate_rows"] == "1"


def test_verify_rejects_invented_ref(store):
    answer = QAAnswer(
        answer="Something",
        evidence_refs=["duplicates.made_up_field"],
        confidence="grounded",
    )
    verified = verify_answer(answer, store)
    assert verified.is_grounded is False


def test_verify_cannot_answer_is_trivially_grounded(store):
    # no evidence_refs needed when the model correctly says it can't answer
    answer = QAAnswer(answer="Not covered.", evidence_refs=[], confidence="cannot_answer")
    verified = verify_answer(answer, store)
    assert verified.is_grounded is True


def test_verify_grounded_with_no_refs_is_rejected(store):
    # claiming "grounded" but citing nothing is not trustworthy
    answer = QAAnswer(answer="Something", evidence_refs=[], confidence="grounded")
    verified = verify_answer(answer, store)
    assert verified.is_grounded is False
