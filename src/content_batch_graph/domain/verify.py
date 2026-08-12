"""
Verify a raw finding against the real source text before it can be trusted.

This is the single most important discipline in the manual protocol this project
automates: a critic's claim is worthless until it's confirmed to exist, verbatim, in
the actual file. No LLM output is ever passed downstream unverified.
"""

from __future__ import annotations

from content_batch_graph.state import Finding, VerifiedFinding


def verify_finding(finding: Finding, source_text: str) -> VerifiedFinding | None:
    """
    Returns a VerifiedFinding if finding['quoted_text'] exists verbatim in source_text,
    otherwise None. A finding whose quoted text can't be found verbatim is not a real
    finding — it's a hallucinated or paraphrased claim, and gets dropped, not trusted.
    """
    if finding["quoted_text"] and finding["quoted_text"] in source_text:
        return VerifiedFinding(
            quoted_text=finding["quoted_text"],
            issue=finding["issue"],
            category=finding["category"],
            verified=True,
        )
    return None


def verify_findings(
    findings: list[Finding], source_text: str
) -> tuple[list[VerifiedFinding], list[Finding]]:
    """
    Splits findings into (verified, rejected) against source_text.
    verified: quoted_text confirmed present verbatim -> safe to act on.
    rejected: quoted_text not found -> dropped, never passed downstream.
    """
    verified: list[VerifiedFinding] = []
    rejected: list[Finding] = []
    for finding in findings:
        result = verify_finding(finding, source_text)
        if result is not None:
            verified.append(result)
        else:
            rejected.append(finding)
    return verified, rejected
