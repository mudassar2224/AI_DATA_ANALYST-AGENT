"""The interpreter agent: reads the evidence store plus retrieved
knowledge-base context, and asks an LLM to draft qualitative findings.

The `llm` parameter is injected (not constructed inside this module) on
purpose — it's what makes this testable with a stub in place of a real
ChatGroq instance. See build_default_llm() for the production wiring.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dsagent.agents.schemas import Finding, FindingList
from dsagent.evidence.store import EvidenceStore
from dsagent.knowledge.base import KnowledgeEntry, retrieve


def _normalise_api_keys(api_key: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if api_key is None:
        return []
    if isinstance(api_key, str):
        candidates = [api_key]
    else:
        candidates = list(api_key)

    keys: list[str] = []
    for candidate in candidates:
        cleaned = (candidate or "").strip()
        if cleaned and cleaned not in keys:
            keys.append(cleaned)
    return keys


def _retry_backoff_seconds(attempt_index: int) -> float:
    """Back off gradually when moving to the next configured provider/key."""
    return min(0.5 * (2**attempt_index), 4.0)


def _is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {401, 403, 429, 500, 502, 503, 504}

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            value = int(response.status_code)
        except (TypeError, ValueError):
            value = None
        if value in {401, 403, 429, 500, 502, 503, 504}:
            return True

    text = str(exc).lower()
    retry_tokens = (
        "rate limit",
        "429",
        "too many requests",
        "unauthorized",
        "forbidden",
        "timed out",
        "timeout",
        "temporar",
        "server error",
        "503",
        "504",
        "502",
        "500",
    )
    return any(token in text for token in retry_tokens)


def _is_structured_output_compatibility_error(exc: Exception) -> bool:
    """Detect provider 400s caused by an invalid tool-call response.

    Some Groq/model combinations return a tool name that does not match the
    tool registered by LangChain's default function-calling structured-output
    mode. JSON mode avoids that tool-name negotiation entirely.
    """
    text = str(exc).lower()
    return (
        "tool call validation failed" in text
        or "not in request.tools" in text
        or ("invalid_request_error" in text and "tool" in text)
    )


class _FallbackStructuredClient:
    def __init__(self, fallback_client: _FallbackLLMClient, schema: Any):
        self._fallback_client = fallback_client
        self._schema = schema

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt_index, client in enumerate(self._fallback_client._clients):
            try:
                structured = client[1].with_structured_output(self._schema)
                return structured.invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - rotate to next configured key
                last_exc = exc
                if _is_structured_output_compatibility_error(exc):
                    try:
                        structured = client[1].with_structured_output(
                            self._schema, method="json_mode"
                        )
                        return structured.invoke(messages, **kwargs)
                    except Exception as json_exc:  # noqa: BLE001 - preserve provider fallback
                        last_exc = json_exc
                        if not _is_retryable_api_error(json_exc):
                            raise
                if not _is_retryable_api_error(exc):
                    raise
                if attempt_index < len(self._fallback_client._clients) - 1:
                    time.sleep(_retry_backoff_seconds(attempt_index))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No valid Groq API keys were available to answer the request.")


class _FallbackLLMClient:
    def __init__(
        self,
        model: str,
        api_keys: list[str],
        *,
        openrouter_model: str | None = None,
        openrouter_api_key: str | None = None,
    ):
        from langchain_groq import ChatGroq
        from langchain_openai import ChatOpenAI

        self.model = model
        self.api_keys = api_keys
        self._clients: list[tuple[str, Any]] = []

        for key in api_keys:
            self._clients.append(
                (
                    "groq",
                    ChatGroq(model=model, api_key=key, temperature=0, max_retries=0, timeout=60),
                )
            )

        if openrouter_api_key:
            self._clients.append(
                (
                    "openrouter",
                    ChatOpenAI(
                        model=openrouter_model or model,
                        api_key=openrouter_api_key,
                        base_url="https://openrouter.ai/api/v1",
                        temperature=0,
                        max_retries=0,
                        timeout=60,
                        default_headers={
                            "HTTP-Referer": "https://github.com",
                            "X-Title": "AI Data Scientist",
                        },
                    ),
                )
            )

    def with_structured_output(self, schema: Any) -> _FallbackStructuredClient:
        return _FallbackStructuredClient(self, schema)

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt_index, (_, client) in enumerate(self._clients):
            try:
                return client.invoke(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - rotate to next configured provider/key
                last_exc = exc
                if not _is_retryable_api_error(exc):
                    raise
                if attempt_index < len(self._clients) - 1:
                    time.sleep(_retry_backoff_seconds(attempt_index))
        if last_exc is not None:
            raise last_exc

        raise RuntimeError("No valid LLM provider keys were available to answer the request.")


SYSTEM_PROMPT = """You are the interpretation layer of a data-analysis tool. \
You are given a compact, deterministic summary of a dataset's profile, \
quality checks, and relationship scan, computed by ordinary Python code — \
you did not compute any of it yourself. Your job is to explain what it \
means, not to restate the numbers.

Rules:
- Every finding's headline must be qualitative — describe the pattern, \
never repeat a specific number from the evidence. The exact numbers are \
displayed separately, sourced directly from the evidence, not from you.
- Every finding must cite at least one evidence_ref, using the exact \
dotted-path names shown in the evidence summary below. Never invent a \
column name, field name, or ref that wasn't shown to you.
- Only write a finding when the evidence actually supports it. Fewer, \
well-grounded findings are better than many speculative ones.
- Use "associated with", never "causes" — nothing here establishes causation.
- Treat the reference thresholds you're given as rules of thumb, not \
absolute cutoffs; say so if a value sits near a boundary.
"""


def _relevant_tags(store: EvidenceStore) -> list[str]:
    tags: list[str] = []
    if store.missing.total_missing_cells > 0:
        tags += ["missing", "quality"]
    if store.duplicates.exact_duplicate_rows > 0:
        tags += ["duplicates", "quality"]
    if store.outliers:
        tags.append("outliers")
    for result in store.relationships.results:
        if result.is_significant:
            tags.append(result.relationship_type)
    if store.relationships.results:
        tags.append("relationship")
    for col in store.profile.columns:
        if col.inferred_type == "categorical" and col.unique_count > 50:
            tags.append("cardinality")
    return list(dict.fromkeys(tags))


def _select_columns_to_detail(store: EvidenceStore, max_columns_detailed: int):
    """Pick which columns are worth describing in full when there are more
    than max_columns_detailed of them. Prioritizes columns that actually
    carry a story — missing data, or a role in a significant relationship
    — over an arbitrary "first N" cut, so wide datasets (100+ columns)
    still get a useful summary instead of just a truncated list.
    """
    columns = store.profile.columns
    if len(columns) <= max_columns_detailed:
        return columns, 0

    noteworthy: set[str] = set()
    for col in columns:
        if col.missing_pct > 0:
            noteworthy.add(col.name)
    for r in store.relationships.results:
        if r.is_significant:
            noteworthy.add(r.column_a)
            noteworthy.add(r.column_b)

    prioritized = [c for c in columns if c.name in noteworthy][:max_columns_detailed]
    if len(prioritized) < max_columns_detailed:
        filler = [c for c in columns if c.name not in noteworthy]
        prioritized += filler[: max_columns_detailed - len(prioritized)]

    return prioritized, len(columns) - len(prioritized)


def summarize_evidence(
    store: EvidenceStore,
    max_relationships: int = 8,
    max_columns_detailed: int = 25,
    max_outliers_detailed: int = 10,
    max_chars: int = 10_000,
) -> str:
    """A compact, LLM-readable text summary of the evidence store, with the
    exact dotted-path refs the interpreter must use to cite it.

    Capped on three axes so a wide dataset can't blow the model's
    per-request token budget: at most max_columns_detailed columns get a
    full line (the rest are just counted), at most max_outliers_detailed
    outlier flags are listed (sorted by extremity), and the whole thing is
    hard-truncated at max_chars as a final safety net regardless of how
    the first two caps land. A 184-column dataset (178 numeric — meaning
    178 outlier-flag lines on top of 184 column lines) produced a
    13.8k-token request against an 8k TPM limit before these caps
    existed — see tests/test_interpreter.py for the regression tests.
    """
    shown_columns, omitted_count = _select_columns_to_detail(store, max_columns_detailed)
    all_names = [c.name for c in store.profile.columns]

    lines = [
        f"Dataset: {store.profile.row_count} rows, {store.profile.column_count} columns "
        f"({store.profile.numeric_columns} numeric, {store.profile.categorical_columns} "
        f"categorical, {store.profile.datetime_columns} datetime).",
        f"Ref 'profile.duplicate_row_count': {store.profile.duplicate_row_count} duplicate rows.",
        "",
        "All column names, in order (ref: 'profile.column_count' for the count, "
        "'profile.columns[<name>]' for one column's detail) — this is the complete "
        "list, even when detailed stats below only cover a subset:",
        "  " + ", ".join(all_names),
        "",
        "Columns — cite a specific stat as 'profile.columns[<name>].<field>', "
        "e.g. 'profile.columns[age].missing_pct' or 'profile.columns[age].mean'. "
        + (
            f"Only {len(shown_columns)} of the {len(all_names)} columns above get a "
            "full stats line here — the rest are in the name list only."
            if omitted_count
            else "Every column is detailed below."
        ),
    ]
    for col in shown_columns:
        stats = ""
        if col.inferred_type == "numeric":
            stats = f", mean={col.mean}, min={col.min}, max={col.max}"
        lines.append(
            f"  - {col.name} ({col.inferred_type}): {col.missing_pct}% missing, "
            f"{col.unique_count} unique values{stats}"
        )
    if omitted_count:
        lines.append(
            f"  ... and {omitted_count} more column(s) not detailed here (dataset has "
            f"{store.profile.column_count} total) — the ones shown were picked for having "
            "missing data or a significant relationship."
        )

    lines += [
        "",
        f"Missingness — ref 'missing.total_missing_cells': {store.missing.total_missing_cells} "
        f"cells ({store.missing.overall_missing_pct}% overall). "
        f"Ref 'missing.by_column_pct.<name>' for a specific column's %.",
    ]
    if store.missing.total_missing_cells == 0:
        lines.append(
            "  The total is zero, so every column has 0.0% missing values; "
            "cite 'profile.columns[<name>].missing_pct' for a specific column."
        )
    if store.missing.columns_with_disguised_missing:
        lines.append(
            "  Disguised-missing tokens (e.g. 'Unknown', 'N/A') found in: "
            + ", ".join(store.missing.columns_with_disguised_missing)
        )

    identifier_columns = [
        col
        for col in store.profile.columns
        if col.name.lower() == "id" or col.name.lower().endswith("_id")
    ]
    if identifier_columns:
        lines.append("")
        lines.append("Likely identifier uniqueness:")
        for col in identifier_columns[:10]:
            lines.append(
                f"  - {col.name}: {col.unique_count} unique values across "
                f"{store.profile.row_count} rows (ref: "
                f"'profile.columns[{col.name}].unique_count')."
            )

    binary_numeric_columns = [
        col
        for col in store.profile.columns
        if col.inferred_type == "numeric" and col.unique_count == 2
    ]
    if binary_numeric_columns:
        lines.append("")
        lines.append("Binary numeric columns, useful as possible targets:")
        for col in binary_numeric_columns[:10]:
            lines.append(
                f"  - {col.name}: mean={col.mean}, min={col.min}, max={col.max} "
                f"(ref: 'profile.columns[{col.name}].mean')."
            )

    if store.outliers:
        lines.append("")
        sorted_outliers = sorted(store.outliers, key=lambda o: -o.outlier_pct)
        shown_outliers = sorted_outliers[:max_outliers_detailed]
        omitted_outliers = len(sorted_outliers) - len(shown_outliers)
        lines.append(
            f"Outliers, top {len(shown_outliers)} by extremity (ref: 'outliers[<column>]'):"
        )
        for flag in shown_outliers:
            lines.append(
                f"  - {flag.column}: {flag.outlier_count} values ({flag.outlier_pct}%) "
                f"outside [{flag.lower_bound}, {flag.upper_bound}]"
            )
        if omitted_outliers:
            lines.append(f"  ... and {omitted_outliers} more column(s) with some outliers.")

    significant = [r for r in store.relationships.results if r.is_significant]
    if significant:
        lines.append("")
        lines.append(
            f"Significant relationships after FDR correction, top {max_relationships} "
            f"(ref: 'relationships.results[<col_a>,<col_b>]'):"
        )
        for r in significant[:max_relationships]:
            lines.append(
                f"  - {r.column_a} <-> {r.column_b}: {r.metric_name}={r.metric_value} "
                f"({r.effect_label})"
            )
    elif store.relationships.results:
        lines.append("")
        lines.append("No relationships survived FDR correction — none were significant.")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = (
            text[:max_chars] + "\n  ... (truncated to stay within the model's request size limit)"
        )
    return text


def _knowledge_text(entries: list[KnowledgeEntry]) -> str:
    if not entries:
        return "(no specific reference thresholds retrieved for this dataset)"
    return "\n".join(f"- {e.text}" for e in entries)


def interpret_findings(
    store: EvidenceStore,
    llm: BaseChatModel,
    max_findings: int = 8,
    retry_feedback: str = "",
) -> list[Finding]:
    """Ask the LLM for a structured list of Findings grounded in `store`.

    Raises whatever the underlying llm call raises (e.g. an auth or rate
    limit error from Groq) — the caller (the graph node) is responsible
    for catching that and falling back gracefully.
    """
    tags = _relevant_tags(store)
    knowledge = retrieve(tags)

    human_parts = [
        f"Evidence summary:\n{summarize_evidence(store)}",
        f"\nReference knowledge:\n{_knowledge_text(knowledge)}",
        f"\nWrite up to {max_findings} findings.",
    ]
    if retry_feedback:
        human_parts.insert(0, f"Your previous attempt had problems:\n{retry_feedback}\n")

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n".join(human_parts))]

    structured_llm = llm.with_structured_output(FindingList)
    result = structured_llm.invoke(messages)

    if isinstance(result, FindingList):
        return result.findings
    if isinstance(result, dict):
        return FindingList.model_validate(result).findings
    raise TypeError(f"Unexpected structured output type from LLM: {type(result)!r}")


def build_default_llm(
    model: str,
    api_key: str | list[str] | tuple[str, ...] | None,
    *,
    openrouter_api_key: str | None = None,
    openrouter_model: str | None = None,
) -> BaseChatModel:
    """Production wiring for the app's LLM chain: Groq first, OpenRouter second.

    The app uses Groq as the primary provider with rotation across multiple keys,
    then falls through to OpenRouter if Groq keys are exhausted or rate-limited.
    This keeps the architecture within the Groq + OpenRouter plan you asked for,
    without introducing a direct OpenAI-specific gateway path.
    """
    keys = _normalise_api_keys(api_key)

    if not keys and not openrouter_api_key:
        raise ValueError("At least one Groq key or OpenRouter key is required to build the LLM.")

    if not keys:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=openrouter_model or model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_retries=0,
            timeout=60,
            default_headers={
                "HTTP-Referer": "https://github.com",
                "X-Title": "AI Data Scientist",
            },
        )

    return _FallbackLLMClient(
        model=model,
        api_keys=keys,
        openrouter_model=openrouter_model or model,
        openrouter_api_key=openrouter_api_key,
    )
