"""The prune_pass node: cuts structurally useless findings before critic_pass."""

from __future__ import annotations

from content_batch_graph.domain.prune import prune_findings
from content_batch_graph.state import BatchState


def prune_pass(state: BatchState) -> dict:
    kept, discarded = prune_findings(state["verified_findings"])
    return {"verified_findings": kept, "discarded_findings": discarded}
