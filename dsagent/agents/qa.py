"""The question-answering agent.

The original plan had separate Planner and Chat agents — Planner turns a
question into a call against a tool registry, Chat answers grounded
follow-ups. They're merged into one agent here because, in this
architecture, there's no separate tool-selection decision left to make:
every specialist (profiler, quality audit, relationships) already runs
eagerly on every upload, so the evidence a question needs is always
already in the EvidenceStore. What Planner would have "planned" is a
no-op. So this one agent does what's actually left: read a question,
answer it from the evidence that's already there, and say so plainly
when it can't.

Same trust design as the interpreter: the model can't write a number
into its answer that isn't backed by a resolved evidence_ref, and the
same fail-closed verifier machinery (dsagent.agents.grounding) checks it.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from dsagent.agents.grounding import resolve_refs
from dsagent.agents.interpreter import summarize_evidence
from dsagent.evidence.store import EvidenceStore
from dsagent.knowledge.base import retrieve

SYSTEM_PROMPT_QA = """You answer questions about a dataset using ONLY the \
evidence summary you're given below — computed by ordinary Python code, \
not by you. You are not shown the raw data, only this summary.

Rules:
- Your answer must be qualitative — describe the pattern, never restate a \
specific number. The real number is displayed separately, sourced from \
the evidence directly.
- Cite at least one evidence_ref for any claim, using the exact \
dotted-path names shown in the evidence summary. Never invent a ref.
- If the evidence summary doesn't actually contain what's needed to \
answer the question, set confidence to "cannot_answer" and say what \
would be needed instead — don't guess.
- Use "associated with", never "causes".
"""

Confidence = Literal["grounded", "cannot_answer"]


class QAAnswer(BaseModel):
    answer: str = Field(description="Qualitative answer, or what's missing if cannot_answer.")
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Dotted-path refs backing the answer. Empty only when cannot_answer.",
    )
    confidence: Confidence = Field(
        description="'grounded' if the evidence answers this, 'cannot_answer' otherwise."
    )


class VerifiedAnswer(BaseModel):
    answer: QAAnswer
    resolved_values: dict[str, str] = Field(default_factory=dict)
    is_grounded: bool


def _relevant_tags(question: str) -> list[str]:
    # cheap keyword routing into the same knowledge base the interpreter
    # uses — good enough for a small, fixed tag vocabulary
    q = question.lower()
    tags = []
    if any(w in q for w in ("missing", "null", "empty", "na")):
        tags += ["missing", "quality"]
    if any(w in q for w in ("duplicate", "dup")):
        tags += ["duplicates", "quality"]
    if any(w in q for w in ("outlier", "unusual", "extreme")):
        tags.append("outliers")
    if any(w in q for w in ("relat", "correlat", "associat", "depend")):
        tags += [
            "relationship",
            "numeric_numeric",
            "categorical_numeric",
            "categorical_categorical",
        ]
    if any(w in q for w in ("imbalance", "balanced", "target", "class")):
        tags.append("imbalance")
    if any(w in q for w in ("cardinality", "unique", "categor")):
        tags.append("cardinality")
    return list(dict.fromkeys(tags))


def answer_question(
    question: str,
    store: EvidenceStore,
    llm: BaseChatModel,
    retry_feedback: str = "",
) -> QAAnswer:
    knowledge = retrieve(_relevant_tags(question))
    knowledge_text = (
        "\n".join(f"- {k.text}" for k in knowledge)
        if knowledge
        else "(no specific reference thresholds retrieved for this question)"
    )

    parts = [
        f"Evidence summary:\n{summarize_evidence(store)}",
        f"\nReference knowledge:\n{knowledge_text}",
        f"\nQuestion: {question}",
    ]
    if retry_feedback:
        parts.insert(0, f"Your previous attempt had problems:\n{retry_feedback}\n")

    messages = [SystemMessage(content=SYSTEM_PROMPT_QA), HumanMessage(content="\n".join(parts))]
    structured_llm = llm.with_structured_output(QAAnswer)
    result = structured_llm.invoke(messages)

    if isinstance(result, QAAnswer):
        return result
    if isinstance(result, dict):
        return QAAnswer.model_validate(result)
    raise TypeError(f"Unexpected structured output type from LLM: {type(result)!r}")


def verify_answer(answer: QAAnswer, store: EvidenceStore) -> VerifiedAnswer:
    if answer.confidence == "cannot_answer":
        return VerifiedAnswer(answer=answer, resolved_values={}, is_grounded=True)

    if not answer.evidence_refs:
        return VerifiedAnswer(answer=answer, resolved_values={}, is_grounded=False)

    resolved, ok = resolve_refs(answer.evidence_refs, store)
    return VerifiedAnswer(answer=answer, resolved_values=resolved, is_grounded=ok)


def answer_rejection_feedback(answer: QAAnswer) -> str:
    return (
        f"Your answer '{answer.answer}' cited refs that don't exist in the evidence: "
        f"{answer.evidence_refs}. Use only refs shown in the evidence summary, or set "
        f"confidence to 'cannot_answer' if the evidence genuinely doesn't cover this."
    )
