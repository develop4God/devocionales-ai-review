"""The fix_pass node: applies corrections for approved findings, tracks attempt count."""

from __future__ import annotations

from content_batch_graph.domain.fix import run_fix_pass
from content_batch_graph.state import BatchState


def fix_pass(state: BatchState) -> dict:
    decision = state["human_decision"] or {}
    apply_indices = decision.get("apply", [])
    critic_findings = state["critic_findings"]
    findings = [
        critic_findings[i] for i in apply_indices if 0 <= i < len(critic_findings)
    ]
    fixed_text, summary = run_fix_pass(state["file_text"], findings)
    return {
        "fixed_text": fixed_text,
        "fix_summary": summary,
        "fix_attempts": state.get("fix_attempts", 0) + 1,
    }
