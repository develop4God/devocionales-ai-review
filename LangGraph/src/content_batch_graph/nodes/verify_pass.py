"""The verify_pass node: checks raw_findings against file_text, splits verified/rejected."""

from __future__ import annotations

from content_batch_graph.domain.verify import verify_findings
from content_batch_graph.state import BatchState


def verify_pass(state: BatchState) -> dict:
    verified, rejected = verify_findings(state["raw_findings"], state["file_text"])
    return {"verified_findings": verified, "rejected_findings": rejected}
