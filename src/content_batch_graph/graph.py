"""
The graph definition. One StateGraph, wired here and nowhere else.

flag -> verify -> critic -> human_confirm -> (approved) -> fix -> drift_check
                                           -> (rejected) -> END
    drift_check -> (no drift) -> validate -> (passed) -> END
                                           -> (failed, under cap) -> fix
                -> (drift)    -> human_confirm -> (approved) -> fix
                                               -> (anything else) -> END

Later phases (translate fan-out, round-2 independence, final cross-check, index
update, rule proposal) are added as their own nodes and edges when they're built —
this file grows, it is never duplicated.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from content_batch_graph.nodes.critic_pass import critic_pass
from content_batch_graph.nodes.drift_check_pass import drift_check_pass
from content_batch_graph.nodes.fix_pass import fix_pass
from content_batch_graph.nodes.flag_pass import flag_pass
from content_batch_graph.nodes.human_confirm import human_confirm
from content_batch_graph.nodes.validate_pass import validate_pass
from content_batch_graph.nodes.verify_pass import verify_pass
from content_batch_graph.state import BatchState

MAX_FIX_ATTEMPTS = 3


def _after_human_confirm(state: BatchState) -> str:
    return "fix_pass" if state["human_decision"] == "approved" else END


def _after_drift_check(state: BatchState) -> str:
    return "human_confirm" if state["drift_detected"] else "validate_pass"


def _after_validate(state: BatchState) -> str:
    if state["validation_passed"]:
        return END
    if state["fix_attempts"] >= MAX_FIX_ATTEMPTS:
        return END
    return "fix_pass"


def build_graph():
    builder = StateGraph(BatchState)

    builder.add_node("flag_pass", flag_pass)
    builder.add_node("verify_pass", verify_pass)
    builder.add_node("critic_pass", critic_pass)
    builder.add_node("human_confirm", human_confirm)
    builder.add_node("fix_pass", fix_pass)
    builder.add_node("drift_check_pass", drift_check_pass)
    builder.add_node("validate_pass", validate_pass)

    builder.add_edge(START, "flag_pass")
    builder.add_edge("flag_pass", "verify_pass")
    builder.add_edge("verify_pass", "critic_pass")
    builder.add_edge("critic_pass", "human_confirm")
    builder.add_conditional_edges("human_confirm", _after_human_confirm)
    builder.add_edge("fix_pass", "drift_check_pass")
    builder.add_conditional_edges("drift_check_pass", _after_drift_check)
    builder.add_conditional_edges("validate_pass", _after_validate)

    return builder


def compile_graph(checkpoint_path: str | Path):
    """
    Compiles the graph with a real SqliteSaver checkpointer at checkpoint_path.
    A checkpointer is required for interrupt()/Command(resume=...) to survive across
    separate process invocations, not just within one Python process's memory.
    """
    conn_cm = SqliteSaver.from_conn_string(str(checkpoint_path))
    checkpointer = conn_cm.__enter__()
    graph = build_graph().compile(checkpointer=checkpointer)
    return graph, conn_cm
