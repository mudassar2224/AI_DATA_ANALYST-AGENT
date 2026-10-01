"""Tests for dsagent.agents.interpreter.

Uses a small stub in place of a real ChatGroq instance — interpret_findings
takes `llm` as a parameter specifically so these tests never touch the
network. What's tested here is OUR code: prompt construction, response
parsing, and the retry-feedback path. Whether Groq itself responds well
to the prompt is the one thing only a live key can confirm — see
docs/architecture.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsagent.agents.interpreter import (
    _FallbackStructuredClient,
    _retry_backoff_seconds,
    interpret_findings,
    summarize_evidence,
)
from dsagent.agents.schemas import Finding, FindingList
from dsagent.analysis.relationships import find_relationships
from dsagent.evidence.store import EvidenceStore
from dsagent.profiling.profile import profile_dataset
from dsagent.quality.checks import check_duplicates, check_missingness, check_outliers


class _StubStructuredRunnable:
    def __init__(self, response):
        self._response = response
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return self._response


class _StubLLM:
    """Mimics just enough of BaseChatModel's interface for interpret_findings:
    llm.with_structured_output(schema) -> a runnable with .invoke(messages).
    """

    def __init__(self, response):
        self._response = response
        self.structured_runnable: _StubStructuredRunnable | None = None

    def with_structured_output(self, schema):
        self.structured_runnable = _StubStructuredRunnable(self._response)
        return self.structured_runnable


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


def _sample_finding() -> Finding:
    return Finding(
        id="income-missing",
        headline="Income has some missing data",
        evidence_refs=["missing.by_column_pct.income"],
        severity="watch",
        meaning="Worth checking before modeling.",
        next_step="Compare missingness across segments.",
    )


def test_returns_findings_from_pydantic_response(store):
    stub = _StubLLM(FindingList(findings=[_sample_finding()]))
    findings = interpret_findings(store, stub)
    assert len(findings) == 1
    assert findings[0].id == "income-missing"


def test_handles_dict_response_defensively(store):
    # Some structured-output configs can return a plain dict rather than a
    # FindingList instance — interpret_findings must handle both.
    stub = _StubLLM({"findings": [_sample_finding().model_dump()]})
    findings = interpret_findings(store, stub)
    assert len(findings) == 1


def test_prompt_includes_retry_feedback(store):
    stub = _StubLLM(FindingList(findings=[]))
    interpret_findings(store, stub, retry_feedback="Your previous refs were invented.")
    sent_text = stub.structured_runnable.last_messages[-1].content
    assert "invented" in sent_text


def test_summary_mentions_disguised_missing_column(store):
    summary = summarize_evidence(store)
    assert "city" in summary
    assert "Unknown" in summary or "disguised" in summary.lower()


def test_summary_lists_all_column_names(store):
    summary = summarize_evidence(store)
    for col in ["id", "age", "city", "income"]:
        assert col in summary


def test_retry_backoff_grows_then_caps():
    assert _retry_backoff_seconds(0) == 0.5
    assert _retry_backoff_seconds(1) == 1.0
    assert _retry_backoff_seconds(5) == 4.0


def test_structured_output_retries_json_mode_after_tool_call_validation_error():
    class _Runnable:
        def __init__(self, response=None, error=None):
            self.response = response
            self.error = error

        def invoke(self, messages, **kwargs):
            if self.error:
                raise self.error
            return self.response

    class _Client:
        def __init__(self):
            self.methods = []

        def with_structured_output(self, schema, **kwargs):
            self.methods.append(kwargs.get("method", "default"))
            if kwargs.get("method") == "json_mode":
                return _Runnable(response={"ok": True})
            return _Runnable(
                error=RuntimeError(
                    "Tool call validation failed: attempted to call tool "
                    "'analysis' which was not in request.tools"
                )
            )

    client = _Client()
    fallback_client = type("_Clients", (), {"_clients": [("groq", client)]})()
    structured = _FallbackStructuredClient(fallback_client, object)

    assert structured.invoke([]) == {"ok": True}
    assert client.methods == ["default", "json_mode"]


def test_summary_caps_size_for_wide_dataset():
    # Reproduces the real failure: a 184-column dataset produced a
    # 13.8k-token request against an 8k TPM limit before the cap existed.
    # 200 columns here is a comparable shape.
    rng = np.random.default_rng(0)
    n = 300
    wide_df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, n) for i in range(200)})
    wide_df.loc[rng.choice(n, 5, replace=False), "col_3"] = None  # give it one real signal

    profile = profile_dataset(wide_df)
    column_types = {c.name: c.inferred_type for c in profile.columns}
    store = EvidenceStore(
        profile=profile,
        missing=check_missingness(wide_df),
        duplicates=check_duplicates(wide_df),
        outliers=check_outliers(wide_df),
        relationships=find_relationships(wide_df, column_types),
    )

    summary = summarize_evidence(store)

    # rough chars-per-token heuristic (~4) keeps this comfortably under an
    # 8000-token TPM budget even before the system prompt and knowledge
    # text are added on top
    assert len(summary) < 11_000
    assert "more column(s) not detailed" in summary


def test_summary_prioritizes_columns_with_missing_data_when_capped():
    rng = np.random.default_rng(0)
    n = 100
    wide_df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, n) for i in range(40)})
    # only col_39 (last, so it'd never make an arbitrary "first N" cut) has
    # missing data -- it must still appear in a capped summary
    wide_df.loc[0:10, "col_39"] = None

    profile = profile_dataset(wide_df)
    column_types = {c.name: c.inferred_type for c in profile.columns}
    store = EvidenceStore(
        profile=profile,
        missing=check_missingness(wide_df),
        duplicates=check_duplicates(wide_df),
        outliers=check_outliers(wide_df),
        relationships=find_relationships(wide_df, column_types),
    )

    summary = summarize_evidence(store, max_columns_detailed=10)
    assert "col_39" in summary


def test_summary_lists_every_column_name_even_when_detail_is_capped():
    # Regression test for a real failure: a user asked a 184-column
    # dataset's chat "give me the name of all columns" and got "I can't,
    # the evidence only shows a subset" -- because the detailed section
    # (correctly) only shows the most relevant columns, but nothing else
    # in the prompt ever named the rest. The full name list is cheap
    # (a few hundred tokens even at 184 columns) and now always present,
    # independent of the detailed-stats cap.
    rng = np.random.default_rng(0)
    n = 50
    wide_df = pd.DataFrame({f"col_{i:03d}": rng.normal(0, 1, n) for i in range(60)})

    profile = profile_dataset(wide_df)
    column_types = {c.name: c.inferred_type for c in profile.columns}
    store = EvidenceStore(
        profile=profile,
        missing=check_missingness(wide_df),
        duplicates=check_duplicates(wide_df),
        outliers=check_outliers(wide_df),
        relationships=find_relationships(wide_df, column_types),
    )

    summary = summarize_evidence(store, max_columns_detailed=10)

    # fewer than 10 columns get a full detailed-stats line (distinguished
    # from an outlier-flag line, which also starts with "  - " but never
    # contains "unique values")
    detailed_count = sum(
        1
        for line in summary.splitlines()
        if line.startswith("  - col_") and "unique values" in line
    )
    assert detailed_count <= 10

    # ...but every single column name still appears in the name-list line,
    # checked as an exact comma-separated entry (not a substring search --
    # "col_005" is a substring of nothing here since names are zero-padded
    # to equal width, which is what makes exact-membership checking valid)
    name_list_line = next(line for line in summary.splitlines() if line.startswith("  col_"))
    listed_names = {n.strip() for n in name_list_line.split(",")}
    assert listed_names == {f"col_{i:03d}" for i in range(60)}
