"""
The graph definition. One StateGraph, wired here and nowhere else.

First slice: flag -> verify -> human_confirm. Later phases (translate fan-out, fix,
real-tool validation, round-2 independence, final cross-check, index update, rule
proposal) are added as their own nodes and edges when they're built — this file grows,
it is never duplicated.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from content_batch_graph.nodes.flag_pass import flag_pass
from content_batch_graph.nodes.human_confirm import human_confirm
from content_batch_graph.nodes.verify_pass import verify_pass
from content_batch_graph.state import BatchState


def build_graph():
    builder = StateGraph(BatchState)

    builder.add_node("flag_pass", flag_pass)
    builder.add_node("verify_pass", verify_pass)
    builder.add_node("human_confirm", human_confirm)

    builder.add_edge(START, "flag_pass")
    builder.add_edge("flag_pass", "verify_pass")
    builder.add_edge("verify_pass", "human_confirm")
    builder.add_edge("human_confirm", END)

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
