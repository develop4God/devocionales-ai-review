"""
Verify a raw finding against the real source text before it can be trusted.

This is the single most important discipline in the manual protocol this project
automates: a critic's claim is worthless until it's confirmed to exist, verbatim, in
the actual file. No LLM output is ever passed downstream unverified.
"""

from __future__ import annotations

import re

from content_batch_graph.state import Finding, VerifiedFinding


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _found_at_word_boundary(quoted_text: str, source_text: str) -> bool:
    """
    True if quoted_text occurs in source_text as a whole span, not merely as a
    substring inside a longer word. Only the two edges of quoted_text are checked
    against their neighboring characters in source_text -- internal characters
    (e.g. spaces inside a multi-word quoted phrase) are never required to be
    boundaries, so this works the same for a single word or a full sentence span.

    Concretely: "refleje" must never match inside "reflejen" (no boundary between
    the "e" quoted_text ends on and the "n" that follows it in source_text), the
    exact bug that let verify_findings pass a hallucinated finding through and
    corrupt "reflejen" into "reflejenn" downstream.
    """
    for match in re.finditer(re.escape(quoted_text), source_text):
        start, end = match.start(), match.end()
        before_ok = start == 0 or not (
            _is_word_char(source_text[start - 1]) and _is_word_char(quoted_text[0])
        )
        after_ok = end == len(source_text) or not (
            _is_word_char(source_text[end]) and _is_word_char(quoted_text[-1])
        )
        if before_ok and after_ok:
            return True
    return False


def verify_finding(finding: Finding, source_text: str) -> VerifiedFinding | None:
    """
    Returns a VerifiedFinding if finding['quoted_text'] exists verbatim in
    source_text at a real word boundary, otherwise None. A finding whose quoted
    text can't be found this way is not a real finding — it's a hallucinated or
    paraphrased claim (or a substring match inside a different, longer word — e.g.
    "refleje" inside the already-correct "reflejen"), and gets dropped, not trusted.
    """
    if finding["quoted_text"] and _found_at_word_boundary(
        finding["quoted_text"], source_text
    ):
        return VerifiedFinding(
            quoted_text=finding["quoted_text"],
            issue=finding["issue"],
            category=finding["category"],
            proposed_text=finding.get("proposed_text"),
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
