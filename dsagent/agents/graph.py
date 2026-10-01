"""LangGraph orchestration for the reasoning layer.

Two graphs. build_graph()/run_reasoning_layer() is the auto-run
interpret -> verify loop that produces the "AI-interpreted findings"
section on every upload. build_qa_graph()/run_qa() is the same
interpret/verify pattern applied to one user question at a time — see
dsagent.agents.qa for why there's no separate planner step.

Both graphs take the chat model as a parameter rather than constructing
one internally, which is what let every node in this package get tested
without a real Groq connection — see tests/test_agents_graph.py and
tests/test_qa_graph.py. On any exception from the LLM call itself (auth,
rate limit, network), both graphs stop and report the error instead of
retrying blindly.

Next-best-analysis and MCP aren't built; see docs/architecture.md.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from dsagent.agents.interpreter import interpret_findings
from dsagent.agents.qa import (
    QAAnswer,
    VerifiedAnswer,
    answer_question,
    answer_rejection_feedback,
    verify_answer,
)
from dsagent.agents.schemas import Finding, VerificationResult
from dsagent.agents.verifier import rejection_feedback, verify_findings
from dsagent.evidence.store import EvidenceStore

# ---------------------------------------------------------------------------
# Auto-run findings graph
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: EvidenceStore
    attempt: int = 0
    raw_findings: list[Finding] = []
    verification: VerificationResult | None = None
    retry_feedback: str = ""
    error: str | None = None


def build_graph(
    llm: BaseChatModel,
    max_findings: int = 8,
    max_attempts: int = 2,
):
    def interpret_node(state: AgentState) -> dict:
        try:
            raw = interpret_findings(
                state.store,
                llm,
                max_findings=max_findings,
                retry_feedback=state.retry_feedback,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not raised
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"raw_findings": raw, "attempt": state.attempt + 1, "retry_feedback": ""}

    def verify_node(state: AgentState) -> dict:
        if state.error:
            return {}
        result = verify_findings(state.raw_findings, state.store)
        feedback = rejection_feedback(state.raw_findings, result)
        return {"verification": result, "retry_feedback": feedback}

    def route_after_verify(state: AgentState) -> str:
        if state.error:
            return END
        if state.retry_feedback and state.attempt < max_attempts:
            return "interpret"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("interpret", interpret_node)
    graph.add_node("verify", verify_node)
    graph.add_edge(START, "interpret")
    graph.add_edge("interpret", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"interpret": "interpret", END: END})

    return graph.compile()


def run_reasoning_layer(
    store: EvidenceStore,
    llm: BaseChatModel,
    max_findings: int = 8,
    max_attempts: int = 2,
) -> AgentState:
    """Convenience wrapper: build the graph, run it once, return typed state."""
    compiled = build_graph(llm, max_findings=max_findings, max_attempts=max_attempts)
    result = compiled.invoke(AgentState(store=store))
    return AgentState.model_validate(result)


# ---------------------------------------------------------------------------
# Question-answering graph
# ---------------------------------------------------------------------------


class QAState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    store: EvidenceStore
    attempt: int = 0
    verified: VerifiedAnswer | None = None
    retry_feedback: str = ""
    error: str | None = None


def build_qa_graph(
    llm: BaseChatModel,
    max_attempts: int = 2,
):
    def ask_node(state: QAState) -> dict:
        try:
            raw_answer: QAAnswer = answer_question(
                state.question, state.store, llm, retry_feedback=state.retry_feedback
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller, not raised
            return {"error": f"{type(exc).__name__}: {exc}"}
        verified = verify_answer(raw_answer, state.store)
        feedback = "" if verified.is_grounded else answer_rejection_feedback(raw_answer)
        return {"verified": verified, "attempt": state.attempt + 1, "retry_feedback": feedback}

    def route_after_ask(state: QAState) -> str:
        if state.error:
            return END
        if state.retry_feedback and state.attempt < max_attempts:
            return "ask"
        return END

    graph = StateGraph(QAState)
    graph.add_node("ask", ask_node)
    graph.add_edge(START, "ask")
    graph.add_conditional_edges("ask", route_after_ask, {"ask": "ask", END: END})

    return graph.compile()


def run_qa(
    question: str,
    store: EvidenceStore,
    llm: BaseChatModel,
    max_attempts: int = 2,
) -> QAState:
    """Convenience wrapper: build the QA graph, run it once, return typed state."""
    compiled = build_qa_graph(llm, max_attempts=max_attempts)
    result = compiled.invoke(QAState(question=question, store=store))
    return QAState.model_validate(result)
