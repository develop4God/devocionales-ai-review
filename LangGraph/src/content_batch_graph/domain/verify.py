"""
Verify a raw finding against the real source text before it can be trusted.

This is the single most important discipline in the manual protocol this project
automates: a critic's claim is worthless until it's confirmed to exist, verbatim, in
the actual file. No LLM output is ever passed downstream unverified.
"""

from __future__ import annotations

import re
import unicodedata

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


def _fold_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _drop_non_ascii(text: str) -> str:
    return "".join(c for c in text if ord(c) < 128)


def _snap_accent_damaged_quote(quoted_text: str, source_text: str) -> str | None:
    """
    Some models quote accented text with accents stripped ("allegresse") or the
    accented letters dropped entirely ("allgresse", "sparer"), so a real finding
    fails the exact match above and is silently rejected (observed on FR 2027:
    "rien peut me séparer" and "aux allégresse des autres" were both lost this way).
    Returns the real span from source_text when exactly one distinct span matches
    quoted_text after accent folding or dropping non-ASCII letters, else None.
    Deliberately narrow: no general fuzzy matching, so a hallucinated misspelling
    can't be snapped onto an unrelated word.
    """
    if len(quoted_text) < 4:
        return None
    n_words = len(quoted_text.split())
    tokens = list(re.finditer(r"\S+", source_text))
    folded, dropped = _fold_accents(quoted_text), _drop_non_ascii(quoted_text)
    matches: set[str] = set()
    for i in range(len(tokens) - n_words + 1):
        span = source_text[tokens[i].start() : tokens[i + n_words - 1].end()]
        span = re.sub(r"^\W+|\W+$", "", span)
        if span == quoted_text:
            continue
        if _fold_accents(span) == folded or _drop_non_ascii(span) == dropped:
            matches.add(span)
    return matches.pop() if len(matches) == 1 else None


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
    snapped = _snap_accent_damaged_quote(finding["quoted_text"], source_text)
    if snapped and _found_at_word_boundary(snapped, source_text):
        # proposed_text came from the same damaged output, so it is not trusted;
        # critic_pass proposes the replacement from the real text instead.
        return VerifiedFinding(
            quoted_text=snapped,
            issue=finding["issue"],
            category=finding["category"],
            proposed_text=None,
            verified=True,
            snapped_from=finding["quoted_text"],
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
