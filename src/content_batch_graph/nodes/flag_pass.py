"""The flag_pass node: reads file_text from state, calls the flag domain logic."""

from __future__ import annotations

from content_batch_graph.domain.flag import run_flag_pass
from content_batch_graph.state import BatchState


def flag_pass(state: BatchState) -> dict:
    findings = run_flag_pass(state["file_text"], state["language"])
    return {"raw_findings": findings}
