"""
The human_confirm node: pauses the graph and hands verified findings to a human for
approval before anything downstream (fix) acts on them.

Uses LangGraph's interrupt() — the graph genuinely stops here. The caller must resume
it with Command(resume={"apply": [indices]}); nothing about this node's exit is
optional or bypassable. "apply" indexes into critic_findings — only those specific
findings get applied by fix_pass, everything else is dismissed. An empty/missing
"apply" list stops the graph without applying anything.
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
            "critic_findings": state.get("critic_findings", []),
            "question": (
                "Drift detected after fix. Resume with {'apply': [indices]} into "
                "critic_findings to retry the fix on those findings, or {'apply': []} "
                "to stop."
            ),
        }
    else:
        payload = {
            "verified_findings": state["verified_findings"],
            "rejected_findings": state["rejected_findings"],
            "discarded_findings": state.get("discarded_findings", []),
            "critic_findings": state.get("critic_findings", []),
            "question": (
                "Which critic-reviewed findings should be applied? Resume with "
                "{'apply': [indices]} into critic_findings, or {'apply': []} to "
                "dismiss all."
            ),
        }
    decision = interrupt(payload)
    return {"human_decision": decision}
