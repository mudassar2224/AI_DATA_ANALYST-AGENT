"""The verifier: checks every evidence_ref an interpreted Finding cites
against the real EvidenceStore. No LLM call here at all — this is
ordinary Python, which is the point. A finding is verified only if it
cites at least one ref and every ref it cites resolves; anything else is
rejected outright (fail closed) rather than partially trusted.
"""

from __future__ import annotations

from dsagent.agents.grounding import resolve_refs
from dsagent.agents.schemas import Finding, VerificationResult, VerifiedFinding
from dsagent.evidence.store import EvidenceStore


def verify_findings(findings: list[Finding], store: EvidenceStore) -> VerificationResult:
    verified: list[VerifiedFinding] = []
    rejected: list[str] = []

    for finding in findings:
        if not finding.evidence_refs:
            rejected.append(finding.id)
            continue

        resolved_values, ok = resolve_refs(finding.evidence_refs, store)
        if ok:
            verified.append(
                VerifiedFinding(finding=finding, resolved_values=resolved_values, dropped_refs=[])
            )
        else:
            rejected.append(finding.id)

    return VerificationResult(verified=verified, rejected_finding_ids=rejected)


def rejection_feedback(findings: list[Finding], result: VerificationResult) -> str:
    """A short message describing what went wrong, for the interpreter to
    read on a retry attempt. Empty string if nothing was rejected.
    """
    if not result.rejected_finding_ids:
        return ""

    by_id = {f.id: f for f in findings}
    lines = []
    for finding_id in result.rejected_finding_ids:
        finding = by_id.get(finding_id)
        if finding is None:
            continue
        if not finding.evidence_refs:
            lines.append(f"- '{finding.headline}' cited no evidence_refs at all.")
        else:
            lines.append(
                f"- '{finding.headline}' cited refs that don't exist in the evidence: "
                f"{finding.evidence_refs}"
            )

    return (
        "Some findings were rejected because they weren't grounded in real evidence. "
        "Fix them by citing only refs that appear in the evidence summary below, "
        "or drop the finding entirely:\n" + "\n".join(lines)
    )
