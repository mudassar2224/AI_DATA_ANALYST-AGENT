"""Schemas for the reasoning layer.

The core design choice: a Finding's headline is qualitative only — the
LLM interprets, it doesn't restate numbers. Every number the user sees
comes from evidence_refs resolved against the real EvidenceStore, not
from the model's text. That's what makes the verifier's job mechanical
(does this ref exist and does it say what the finding claims?) instead
of a fuzzy "does this number look right" check.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "watch", "concern"]


class Finding(BaseModel):
    id: str = Field(description="Short slug, e.g. 'income-missingness'")
    headline: str = Field(
        description=(
            "One sentence, qualitative only — no specific numbers. "
            "E.g. 'Income has substantial, unevenly distributed missingness' "
            "not 'Income is missing in 8.4% of rows'."
        )
    )
    evidence_refs: list[str] = Field(
        description=(
            "Dotted paths into the evidence store that support this finding, "
            "e.g. 'missing.by_column_pct.Income' or 'outliers[monthly_charge]' "
            "or 'relationships.results[tenure_months,churn]'. Every ref must "
            "point at something that actually appears in the evidence you "
            "were given — never invent a column or field name."
        )
    )
    severity: Severity = Field(
        description="info: neutral. watch: worth noting. concern: likely needs action."
    )
    meaning: str = Field(description="One or two sentences on why this matters.")
    next_step: str = Field(description="One concrete follow-up analysis or check.")


class FindingList(BaseModel):
    findings: list[Finding]


class VerifiedFinding(BaseModel):
    finding: Finding
    resolved_values: dict[str, str] = Field(
        default_factory=dict, description="ref -> the real value it resolved to, for display"
    )
    dropped_refs: list[str] = Field(default_factory=list, description="refs that failed to resolve")

    @property
    def is_fully_verified(self) -> bool:
        return len(self.dropped_refs) == 0


class VerificationResult(BaseModel):
    verified: list[VerifiedFinding]
    rejected_finding_ids: list[str] = Field(
        default_factory=list, description="findings dropped entirely (no refs resolved at all)"
    )
