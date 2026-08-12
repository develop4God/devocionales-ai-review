"""
The flag pass: read source text, produce raw (unverified) findings.

This module's public function, run_flag_pass, is the seam a real LLM call plugs into.
For this first slice it is intentionally a stub — proving the graph's mechanics
(state flow, verification, human interrupt/resume) does not require a live model call,
and building the real flag-pass prompt/role is its own deliberate piece of work, not
something to rush in alongside graph wiring. Replacing this function's body is the
only change needed to go from stub to real model call; nothing about its signature
or the nodes that call it should need to change.
"""

from __future__ import annotations

from content_batch_graph.state import Finding


def run_flag_pass(source_text: str) -> list[Finding]:
    """
    STUB — deterministic placeholder, not a model call.

    Returns a Finding for every literal occurrence of "teh" (a common typo) in
    source_text, so the verify step downstream has something real and checkable
    to operate on. A real implementation calls an LLM with a role/persona prompt
    and parses its structured output into the same list[Finding] shape.
    """
    findings: list[Finding] = []
    needle = "teh"
    start = 0
    while True:
        idx = source_text.find(needle, start)
        if idx == -1:
            break
        findings.append(
            Finding(
                quoted_text=needle,
                issue="Likely typo: 'teh' should probably be 'the'.",
                category="typo",
            )
        )
        start = idx + len(needle)
    return findings
