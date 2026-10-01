"""Shared grounding helper: resolve a list of evidence_refs against an
EvidenceStore, fail-closed. Used by both dsagent.agents.verifier (for
Findings) and dsagent.agents.qa (for QAAnswers) so the two don't
duplicate the same resolution loop.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from dsagent.evidence.store import EvidenceResolutionError, EvidenceStore


def format_value(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def resolve_refs(refs: list[str], store: EvidenceStore) -> tuple[dict[str, str], bool]:
    """Resolve every ref against store. Returns (resolved_values, all_ok).

    Fail-closed: stops at the first ref that doesn't resolve and reports
    all_ok=False, rather than returning a partial set as if it were safe
    to use.
    """
    resolved: dict[str, str] = {}
    for ref in refs:
        try:
            value = store.resolve(ref)
        except EvidenceResolutionError:
            return resolved, False
        resolved[ref] = format_value(value)
    return resolved, True
