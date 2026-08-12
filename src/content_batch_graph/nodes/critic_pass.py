"""The critic_pass node: independently judges each verified finding, one call per finding."""

from __future__ import annotations

from content_batch_graph.domain.critic import run_critic_pass_batch
from content_batch_graph.state import BatchState


def critic_pass(state: BatchState) -> dict:
    critic_findings = run_critic_pass_batch(
        state["file_text"], state["verified_findings"], state["language"]
    )
    return {"critic_findings": critic_findings}
