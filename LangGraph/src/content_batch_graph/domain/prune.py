"""
Prune structurally useless findings before they reach critic_pass.

A cheap, deterministic, no-model-call filter that runs after verify_pass: cuts a
finding only when it came with a proposed_text that's a no-op — identical to the
text it's supposed to replace. Roles that don't ask for proposed_text at all (e.g.
native_reader) are untouched here; their findings have no proposed_text to judge,
by design, and still go to critic_pass exactly as before. This only affects roles
that opted into supplying proposed_text (e.g. native_reader_batch), where an
unchanged proposal is a real signal the finding isn't worth a critic call.

Nothing here is a silent drop: every cut finding is recorded with why, so the rules
can be calibrated against real discard reports instead of being trusted blindly.
"""

from __future__ import annotations

from content_batch_graph.state import VerifiedFinding


def _discard_reason(finding: VerifiedFinding) -> str | None:
    """Returns why this finding should be discarded, or None if it should be kept."""
    proposed = finding.get("proposed_text")
    if not proposed or not proposed.strip():
        # No proposed_text at all: either the role never asks for one (normal,
        # nothing to prune) or the role asked and got nothing back (also nothing
        # actionable here) — either way, not this layer's call. critic_pass judges it.
        return None

    if proposed.strip().casefold() == finding["quoted_text"].strip().casefold():
        return "no-op — proposed_text is identical to quoted_text"

    return None


def prune_findings(
    findings: list[VerifiedFinding],
) -> tuple[list[VerifiedFinding], list[dict]]:
    """
    Splits verified findings into (kept, discarded).

    kept: has a real, non-identical proposed_text — worth a critic_pass call.
    discarded: [{quoted_text, issue, reason}, ...] — cut for free, before critic.
    """
    kept: list[VerifiedFinding] = []
    discarded: list[dict] = []

    for finding in findings:
        reason = _discard_reason(finding)
        if reason is None:
            kept.append(finding)
        else:
            discarded.append(
                {
                    "quoted_text": finding["quoted_text"],
                    "issue": finding["issue"],
                    "reason": reason,
                }
            )

    return kept, discarded
