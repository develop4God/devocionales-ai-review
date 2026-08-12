"""
Builds a single, self-contained review report from a completed graph run — the
artifact meant to be read to validate a fix, not the graph's raw state.

Exists because the graph's own outputs (fix_summary prose, individual critic
reasoning strings) are scattered across several state fields and don't answer the
two questions a reviewer actually needs on every run: exactly which words changed
(word-level diff, so nothing outside the flagged spans is silently touched), and
was each change backed by a real source of truth (KWF dictionary hit) or by
unaided model judgment (weaker, worth a closer look) — especially important when
reviewing a language the reader doesn't speak.
"""

from __future__ import annotations

import difflib
from typing import TypedDict

_CONTEXT_CHARS = 60


class WordChange(TypedDict):
    change_type: str  # "replace", "insert", "delete"
    before: str
    after: str


class ReviewFinding(TypedDict):
    category: str
    quoted_text: str
    replacement_text: str | None
    is_valid: bool
    grounded: (
        bool  # True if this was a dictionary-confirmed SOT hit, not model judgment
    )
    critic_reasoning: str
    context_before: str  # a window of surrounding text, from the ORIGINAL
    context_after: str  # the same window, from the FIXED text


class ReviewReport(TypedDict):
    entry_id: str | None
    language: str
    word_changes: list[WordChange]
    findings: list[ReviewFinding]
    unchanged_word_count: int
    changed_word_count: int
    drift_detected: bool | None
    drift_notes: str | None
    validation_passed: bool | None


def _word_level_diff(before: str, after: str) -> tuple[list[WordChange], int, int]:
    before_words = before.split()
    after_words = after.split()
    matcher = difflib.SequenceMatcher(None, before_words, after_words)

    changes: list[WordChange] = []
    changed_count = 0
    unchanged_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged_count += i2 - i1
            continue
        changed_count += max(i2 - i1, j2 - j1)
        changes.append(
            WordChange(
                change_type=tag,
                before=" ".join(before_words[i1:i2]),
                after=" ".join(after_words[j1:j2]),
            )
        )
    return changes, unchanged_count, changed_count


def _context_window(text: str, quoted_text: str) -> str:
    idx = text.find(quoted_text)
    if idx == -1:
        return ""
    start = max(0, idx - _CONTEXT_CHARS)
    end = min(len(text), idx + len(quoted_text) + _CONTEXT_CHARS)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def build_review_report(
    original_text: str,
    fixed_text: str,
    critic_findings: list[dict],
    entry_id: str | None,
    language: str,
    drift_detected: bool | None = None,
    drift_notes: str | None = None,
    validation_passed: bool | None = None,
) -> ReviewReport:
    """
    critic_findings: the full list of CriticFinding dicts (both applied and
    rejected), each with quoted_text/replacement_text/is_valid/critic_reasoning.
    A finding is marked "grounded" if its critic_reasoning mentions the KWF
    dictionary check (the SOT short-circuit's reasoning always does) — a real,
    if simple, signal that this verdict came from a checkable source rather than
    unaided model judgment.
    """
    word_changes, unchanged, changed = _word_level_diff(original_text, fixed_text)

    findings: list[ReviewFinding] = []
    for f in critic_findings:
        reasoning = f.get("critic_reasoning", "")
        findings.append(
            ReviewFinding(
                category=f["category"],
                quoted_text=f["quoted_text"],
                replacement_text=f.get("replacement_text"),
                is_valid=f["is_valid"],
                grounded="KWF" in reasoning,
                critic_reasoning=reasoning,
                context_before=_context_window(original_text, f["quoted_text"]),
                context_after=(
                    _context_window(fixed_text, f["replacement_text"])
                    if f.get("replacement_text")
                    else _context_window(fixed_text, f["quoted_text"])
                ),
            )
        )

    return ReviewReport(
        entry_id=entry_id,
        language=language,
        word_changes=word_changes,
        findings=findings,
        unchanged_word_count=unchanged,
        changed_word_count=changed,
        drift_detected=drift_detected,
        drift_notes=drift_notes,
        validation_passed=validation_passed,
    )


def format_report_text(report: ReviewReport) -> str:
    """Renders a ReviewReport as plain text for fast human/AI scanning."""
    lines = [
        f"=== Review report: {report['entry_id']} ({report['language']}) ===",
        (
            f"Words changed: {report['changed_word_count']} / "
            f"unchanged: {report['unchanged_word_count']}"
        ),
        "",
        "-- Word-level diff --",
    ]
    for c in report["word_changes"]:
        lines.append(f"  [{c['change_type']}] {c['before']!r} -> {c['after']!r}")

    lines.append("")
    lines.append("-- Findings --")
    for f in report["findings"]:
        verdict = "APPLIED" if f["is_valid"] else "REJECTED"
        source = (
            "SOT-GROUNDED (dictionary-confirmed)"
            if f["grounded"]
            else "model judgment only"
        )
        lines.append(
            f"\n  [{f['category']}] {f['quoted_text']!r} -> {verdict} ({source})"
        )
        if f["replacement_text"]:
            lines.append(f"    replacement: {f['replacement_text']!r}")
        lines.append(f"    reasoning: {f['critic_reasoning']}")
        lines.append(f"    before: {f['context_before']!r}")
        lines.append(f"    after:  {f['context_after']!r}")

    lines.append("")
    lines.append("-- Drift check --")
    lines.append(f"  drift_detected: {report['drift_detected']}")
    if report["drift_notes"]:
        lines.append(f"  notes: {report['drift_notes']}")

    lines.append("")
    lines.append(f"-- Validation: {report['validation_passed']} --")

    return "\n".join(lines)
