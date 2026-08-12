"""The drift_check_pass node: re-checks the surgically fixed text for new/unresolved issues."""

from __future__ import annotations

from content_batch_graph.domain.drift import run_drift_check
from content_batch_graph.state import BatchState


def drift_check_pass(state: BatchState) -> dict:
    drift_detected, notes = run_drift_check(
        state["fixed_text"] or "", state["critic_findings"], state["language"]
    )
    return {"drift_detected": drift_detected, "drift_notes": notes}
