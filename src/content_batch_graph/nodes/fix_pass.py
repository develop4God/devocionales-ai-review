"""The fix_pass node: applies corrections for approved findings, tracks attempt count."""

from __future__ import annotations

from content_batch_graph.domain.fix import run_fix_pass
from content_batch_graph.state import BatchState


def fix_pass(state: BatchState) -> dict:
    findings = (
        state["verified_findings"] if state["human_decision"] == "approved" else []
    )
    fixed_text, summary = run_fix_pass(state["file_text"], findings, state["language"])
    return {
        "fixed_text": fixed_text,
        "fix_summary": summary,
        "fix_attempts": state.get("fix_attempts", 0) + 1,
    }
