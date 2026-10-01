"""The evidence store: the single object every downstream reader (the
interpreter agent, the verifier, and later the chat/planner agents) reads
from instead of importing each specialist's output separately.

resolve() does dotted-path lookup (e.g. "missing.by_column_pct.Income")
against the store's own structure, which is what lets the verifier check
an LLM-cited reference against real computed data without knowing the
store's internal layout in advance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from dsagent.analysis.relationships import RelationshipReport
from dsagent.profiling.profile import DatasetProfile
from dsagent.quality.checks import DuplicateReport, MissingReport, OutlierFlag


class EvidenceResolutionError(KeyError):
    """Raised when a dotted-path reference doesn't resolve to real evidence."""


class EvidenceStore(BaseModel):
    profile: DatasetProfile
    missing: MissingReport
    duplicates: DuplicateReport
    outliers: list[OutlierFlag]
    relationships: RelationshipReport

    def resolve(self, ref: str) -> Any:
        """Resolve a dotted path against this store's own data.

        Supports attribute access, dict-key access, and list indexing by
        a matching `column`/`name`/`column_a`+`column_b` field so a ref
        like "outliers[monthly_charge]" or "relationships[age,churn]"
        finds the right list item without the caller needing an index.
        """
        parts = ref.split(".")
        current: Any = self
        for part in parts:
            current = self._step(current, part)
        return current

    def _step(self, current: Any, part: str) -> Any:
        key, bracket = self._split_bracket(part)

        if key:
            if isinstance(current, BaseModel):
                if key not in type(current).model_fields:
                    raise EvidenceResolutionError(f"No field '{key}' on {type(current).__name__}")
                current = getattr(current, key)
            elif isinstance(current, dict):
                if key not in current:
                    raise EvidenceResolutionError(f"No key '{key}' in dict")
                current = current[key]
            else:
                raise EvidenceResolutionError(f"Can't resolve '{key}' on {type(current).__name__}")

        if bracket is not None:
            current = self._resolve_bracket(current, bracket)

        return current

    @staticmethod
    def _split_bracket(part: str) -> tuple[str, str | None]:
        if "[" in part and part.endswith("]"):
            key, rest = part.split("[", 1)
            return key, rest[:-1]
        return part, None

    @staticmethod
    def _resolve_bracket(current: Any, bracket: str) -> Any:
        if not isinstance(current, list):
            raise EvidenceResolutionError("Bracket lookup used on a non-list value")

        wanted = {v.strip() for v in bracket.split(",")}
        for item in current:
            if isinstance(item, BaseModel):
                names = {
                    str(getattr(item, f))
                    for f in ("column", "name", "column_a", "column_b")
                    if f in type(item).model_fields
                }
                if wanted <= names:
                    return item
        raise EvidenceResolutionError(f"No list item matching [{bracket}]")
