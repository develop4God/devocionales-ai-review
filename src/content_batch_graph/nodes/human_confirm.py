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
    decision = interrupt(
        {
            "verified_findings": state["verified_findings"],
            "rejected_findings": state["rejected_findings"],
            "question": "Approve these verified findings?",
        }
    )
    return {"human_decision": decision}
