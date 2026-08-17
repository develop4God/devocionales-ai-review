"""
Pre-filter review-collect Finding output against verify_pass/prune_pass's own
domain logic, outside the graph, so a batch driver can skip a full graph
invocation (and its checkpoint round-trip) for items where nothing survives.

Both this module and content_batch_graph.nodes.verify_pass/prune_pass call the
same domain.verify.verify_findings / domain.prune.prune_findings -- this is a
second call site, not a second implementation.
"""

from __future__ import annotations

from content_batch_graph.domain.prune import prune_findings
from content_batch_graph.domain.verify import verify_findings
from content_batch_graph.state import Finding


def resolve_field_path(
    document: dict, language_key: str, entry_id: str, field: str
) -> str | None:
    """Mirrors domain/scan.py's data.{lang}.{date}.{i}.{field} convention."""
    lang_data = document.get("data", {}).get(language_key, {})
    for date, entries in lang_data.items():
        for i, entry in enumerate(entries):
            if entry.get("id") == entry_id:
                if entry.get(field, "") != "":
                    return f"data.{language_key}.{date}.{i}.{field}"
    return None


def pre_filter(
    pending: list[tuple[str, str]],
    by_key: dict[tuple[str, str], list[Finding]],
    document: dict,
    language_key: str,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """
    Runs verify_pass/prune_pass's own logic (no model call, no graph) against every
    pending item upfront, so the real "how many will actually reach critic" number
    is known immediately instead of discovered one slow graph invocation at a time.

    Returns (still_pending, pre_pruned_rows). still_pending is pending items that
    have at least one finding left after verify+prune -- these still go through the
    full graph. pre_pruned_rows are ready-to-write ledger rows for items where
    nothing survived verify+prune -- these never need a critic call, so the full
    graph is skipped for them entirely.
    """
    still_pending: list[tuple[str, str]] = []
    pre_pruned_rows: list[dict] = []

    for entry_id, field in pending:
        field_path = resolve_field_path(document, language_key, entry_id, field)
        if field_path is None:
            still_pending.append((entry_id, field))  # let the caller raise its own error
            continue
        parts = field_path.split(".")
        date, i = parts[2], int(parts[3])
        text = document["data"][language_key][date][i][field]

        verified, rejected = verify_findings(by_key[(entry_id, field)], text)
        kept, discarded = prune_findings(verified)

        if kept:
            still_pending.append((entry_id, field))
        else:
            status = "rejected_by_verify" if not verified else "pruned_after_verify"
            pre_pruned_rows.append(
                {
                    "entry_id": entry_id,
                    "field": field,
                    "thread_id": f"{entry_id}:{field}",
                    "raw_findings_count": len(by_key[(entry_id, field)]),
                    "rejected_count": len(rejected),
                    "verified_count": len(verified),
                    "discarded_count": len(discarded),
                    "critic_findings": [],
                    "applied_indices": [],
                    "fix_summary": None,
                    "drift_detected": None,
                    "drift_notes": None,
                    "validation_passed": None,
                    "validation_error": None,
                    "status": status,
                }
            )

    return still_pending, pre_pruned_rows
