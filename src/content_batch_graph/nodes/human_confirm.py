"""
The human_confirm node: pauses the graph and hands verified findings to a human for
approval before anything downstream (fix, in a later slice) acts on them.

Uses LangGraph's interrupt() — the graph genuinely stops here. The caller must resume
it with Command(resume=...); nothing about this node's exit is optional or bypassable.
"""

from __future__ import annotations

from langgraph.types import interrupt

from content_batch_graph.state import BatchState


def human_confirm(state: BatchState) -> dict:
    if state.get("drift_detected"):
        payload = {
            "drift_detected": state["drift_detected"],
            "drift_notes": state["drift_notes"],
            "fixed_text": state["fixed_text"],
            "question": "Drift detected after fix. Approve to retry, anything else to stop.",
        }
    else:
        payload = {
            "verified_findings": state["verified_findings"],
            "rejected_findings": state["rejected_findings"],
            "critic_findings": state.get("critic_findings", []),
            "question": "Approve these critic-reviewed findings?",
        }
    decision = interrupt(payload)
    return {"human_decision": decision}
