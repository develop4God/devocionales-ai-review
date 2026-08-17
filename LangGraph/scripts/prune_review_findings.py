"""
Run verify_pass/prune_pass's own domain logic against a review-collect JSON
(no model call, no graph) and write a postprune file: fully_pruned_rows
(nothing survived verify+prune) vs still_pending_rows (findings that would
reach critic), same shape as review_pt_20260817_010829_postprune.json.

Usage: uv run python scripts/prune_review_findings.py \
    --review-json data/reviews/review_fr_20260817_165523.json \
    --corpus-file ../../devocionales-json/Devocional_year_2025_fr_LSG1910.json \
    --language-key fr \
    --out data/reviews/review_fr_20260817_165523_postprune.json
"""

from __future__ import annotations

import argparse
import json

from content_batch_graph.domain.review_prefilter import (
    classify_item,
    resolve_field_path,
)
from content_batch_graph.state import Finding


def load_review_findings(review_json_path: str) -> dict[tuple[str, str], list[Finding]]:
    with open(review_json_path, encoding="utf-8") as f:
        review = json.load(f)

    by_key: dict[tuple[str, str], list[Finding]] = {}
    for item in review["data"]:
        findings = item.get("findings") or []
        if not findings:
            continue
        key = (item["entry_id"], item["field"])
        by_key[key] = [
            Finding(
                quoted_text=f["quoted_text"],
                issue=f["issue"],
                category=f["category"],
                proposed_text=f.get("proposed_text"),
            )
            for f in findings
        ]
    return by_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--review-json", required=True)
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument(
        "--language-key", required=True, help='corpus data.{key} — e.g. "fr"'
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    by_key = load_review_findings(args.review_json)
    with open(args.corpus_file, encoding="utf-8") as f:
        document = json.load(f)

    fully_pruned_rows: list[dict] = []
    still_pending_rows: list[dict] = []
    rejected_by_verify_count = 0
    pruned_after_verify_count = 0
    still_pending_findings_count = 0

    for (entry_id, field), findings in by_key.items():
        field_path = resolve_field_path(document, args.language_key, entry_id, field)
        if field_path is None:
            raise SystemExit(f"could not resolve field path for {entry_id}:{field}")
        parts = field_path.split(".")
        date, i = parts[2], int(parts[3])
        text = document["data"][args.language_key][date][i][field]

        result = classify_item(entry_id, field, findings, text)

        if result.kept:
            still_pending_findings_count += len(result.kept_findings)
            still_pending_rows.append(
                {
                    "entry_id": entry_id,
                    "field": field,
                    "text": text,
                    "raw_findings_count": len(findings),
                    "rejected_count": result.rejected_count,
                    "verified_count": result.verified_count,
                    "discarded_count": result.discarded_count,
                    "kept_findings": [dict(f) for f in result.kept_findings],
                }
            )
        else:
            if result.pre_pruned_row["status"] == "rejected_by_verify":
                rejected_by_verify_count += 1
            else:
                pruned_after_verify_count += 1
            fully_pruned_rows.append(result.pre_pruned_row)

    out = {
        "source_review_json": args.review_json,
        "corpus_file": args.corpus_file,
        "language_key": args.language_key,
        "total_flagged_rows": len(by_key),
        "total_raw_findings": sum(len(v) for v in by_key.values()),
        "fully_pruned_rows": fully_pruned_rows,
        "still_pending_rows": still_pending_rows,
        "summary": {
            "fully_pruned_count": len(fully_pruned_rows),
            "rejected_by_verify_count": rejected_by_verify_count,
            "pruned_after_verify_count": pruned_after_verify_count,
            "still_pending_count": len(still_pending_rows),
            "still_pending_findings_count": still_pending_findings_count,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"wrote {args.out}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
