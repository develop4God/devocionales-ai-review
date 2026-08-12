"""
The fix pass: apply each critic-approved finding's exact replacement, surgically.

No LLM call here — critic_pass already produced the exact replacement_text for each
finding. This does a plain string replace of quoted_text -> replacement_text, so the
only text that changes is the flagged span itself, nothing else.
"""

from __future__ import annotations

from content_batch_graph.state import CriticFinding


def run_fix_pass(file_text: str, findings: list[CriticFinding]) -> tuple[str, str]:
    """
    Returns (fixed_text, summary). Only findings with is_valid=True and a
    replacement_text are applied. If none qualify, returns file_text unchanged.
    """
    applicable = [f for f in findings if f["is_valid"] and f["replacement_text"]]
    if not applicable:
        return file_text, "No approved findings to fix."

    fixed_text = file_text
    changes = []
    for finding in applicable:
        fixed_text = fixed_text.replace(
            finding["quoted_text"], finding["replacement_text"]
        )
        changes.append(
            f"({finding['category']}) {finding['quoted_text']!r} -> {finding['replacement_text']!r}"
        )

    summary = "Applied " + "; ".join(changes)
    return fixed_text, summary
