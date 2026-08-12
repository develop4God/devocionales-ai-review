"""
The shared graph state schema. One TypedDict, defined once, read/written by every node.

This is the first slice only: flag -> verify -> human-confirm. Fields for later phases
(translate fan-out, fix, validate-with-real-tools, round tracking, rule proposals) are
added when the node that needs them is actually being built, not speculatively now.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class Finding(TypedDict):
    """One claim from a flag pass, before it has been checked against the real file."""

    quoted_text: str  # the exact phrase the flag pass claims is problematic
    issue: str  # human-readable description of what's wrong
    category: str  # e.g. "typo", "grammar", "awkward_phrasing"


class VerifiedFinding(Finding):
    """A Finding that has been checked against the source file and confirmed present."""

    verified: Literal[True]


class BatchState(TypedDict):
    # ── Identity of this run ──────────────────────────────────────────────
    file_path: str  # the file being reviewed, on disk
    file_text: str  # its content, read once by the flag node
    language: str  # the language file_text is written in, e.g. "Spanish"

    # ── Flag pass output ────────────────────────────────────────────────────
    raw_findings: list[Finding]  # everything the flag pass claimed, unverified

    # ── Verify (triage) pass output ─────────────────────────────────────────
    verified_findings: list[VerifiedFinding]  # findings confirmed present in file_text
    rejected_findings: list[
        Finding
    ]  # findings NOT found verbatim — dropped, not trusted

    # ── Human gate ───────────────────────────────────────────────────────────
    human_decision: Literal["approved", "rejected"] | None

    # ── Fix pass output ─────────────────────────────────────────────────────
    fixed_text: str | None  # file_text with approved findings corrected
    fix_summary: str | None  # concise human-readable summary of what changed
    fix_attempts: int  # how many times fix_pass has run this batch, for loop-cap

    # ── Validate pass output ────────────────────────────────────────────────
    validation_passed: bool | None
    validation_error: str | None
