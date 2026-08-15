"""
Parse a downloaded native_reader_batch review results JSONL back into Finding
objects, keyed by (entry_id, field).

Review-side counterpart to batch_collect.py (which parses generation results):
same tolerant-of-partial-failure shape — one bad line records an error and the
rest of the batch still collects, rather than losing hundreds of good results to
one malformed response.

Every record here is still raw/unverified, same as a live flag_pass result
(domain/flag.py) — nothing in domain/verify.py has looked at these findings yet.
A real test batch surfaced at least one wrong finding (a correct Spanish word
flagged as a typo), so a caller applying these to real content must run them
through verify_findings (or an equivalent review step) first, not apply them
directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from content_batch_graph.domain.batch_collect import extract_content
from content_batch_graph.domain.batch_io import review_collection_path, utc_timestamp
from content_batch_graph.domain.roles import Role, build_finding_schema
from content_batch_graph.state import Finding

# review_{lang}_{version}_{entry_id}_{field}, e.g.
# "review_es_RVR1960_filipenses2_9-11_20250616_RVR1960_oracion" — entry_id
# itself can contain underscores, so this can't be a plain split("_"). lang has
# no underscores (see domain/devotional_gen._LANGUAGE_NAMES's keys) and field is
# always exactly "reflexion" or "oracion" (domain/review_gen.REVIEWED_FIELDS),
# so anchoring on those from both ends leaves entry_id as whatever's left in the
# middle, however many underscores it itself contains.
_CUSTOM_ID_RE = re.compile(
    r"^review_(?P<lang>[^_]+)_(?P<version>[^_]+)_(?P<entry_id>.+)_(?P<field>reflexion|oracion)$"
)


class ReviewResult:
    """One parsed review-batch line: which entry/field it's about, and its findings."""

    __slots__ = ("entry_id", "field", "lang", "version", "findings")

    def __init__(
        self,
        entry_id: str,
        field: str,
        lang: str,
        version: str,
        findings: list[Finding],
    ) -> None:
        self.entry_id = entry_id
        self.field = field
        self.lang = lang
        self.version = version
        self.findings = findings


def parse_custom_id(custom_id: str) -> dict | None:
    """(lang, version, entry_id, field) recovered from a review custom_id, or None."""
    m = _CUSTOM_ID_RE.match(custom_id or "")
    return m.groupdict() if m else None


def parse_review_results(
    results_path: Path, role: Role
) -> tuple[list[ReviewResult], list[dict]]:
    """
    Read a review-batch results JSONL into (results, errors).

    results: one ReviewResult per successfully-parsed line.
    errors:  [{custom_id, reason}, ...] — one per line that couldn't be used.

    role is needed to rebuild the same Finding schema the batch was submitted
    with (build_review_response_format(role) at build time) — schema.
    model_validate_json is what actually enforces category is one of the role's
    declared categories, not just "the JSON parsed."
    """
    schema = build_finding_schema(role)
    results: list[ReviewResult] = []
    errors: list[dict] = []

    with open(Path(results_path), encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                errors.append(
                    {"custom_id": f"<line {lineno}>", "reason": "invalid JSONL line"}
                )
                continue

            custom_id = result.get("custom_id", "")
            parts = parse_custom_id(custom_id)
            if parts is None:
                errors.append(
                    {
                        "custom_id": custom_id or f"<line {lineno}>",
                        "reason": "unparseable custom_id",
                    }
                )
                continue

            content = extract_content(result)
            if not content:
                errors.append(
                    {"custom_id": custom_id, "reason": "missing response content"}
                )
                continue

            try:
                parsed = schema.model_validate_json(content)
            except ValidationError as exc:
                errors.append(
                    {
                        "custom_id": custom_id,
                        "reason": f"content doesn't match the Finding schema: {exc}",
                    }
                )
                continue

            findings: list[Finding] = [
                Finding(
                    quoted_text=finding.quoted_text,
                    issue=finding.issue,
                    category=finding.category,
                    proposed_text=finding.proposed_text or None,
                )
                for finding in parsed.findings
            ]

            results.append(
                ReviewResult(
                    entry_id=parts["entry_id"],
                    field=parts["field"],
                    lang=parts["lang"],
                    version=parts["version"],
                    findings=findings,
                )
            )

    return results, errors


def write_review_collection(
    lang: str,
    results: list[ReviewResult],
    errors: list[dict] | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Write collected review results to data/reviews/review_{lang}_{ts}.json.

    Every finding here is still raw/unverified (see this module's own
    docstring) — the "verified" field is always False, a placeholder for a
    future verify_findings pass over batch results, not a claim that anything
    has actually been checked. errors travels with the data (same reasoning as
    batch_collect.write_year_collection) so a partial collection is
    self-describing when opened later, without cross-referencing stdout.
    """
    out_path = (
        Path(out_path) if out_path else review_collection_path(lang, utc_timestamp())
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "lang": lang,
        "count": len(results),
        "errors": errors or [],
        "data": [
            {
                "entry_id": r.entry_id,
                "field": r.field,
                "version": r.version,
                "findings": [{**f, "verified": False} for f in r.findings],
            }
            for r in results
        ],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path
