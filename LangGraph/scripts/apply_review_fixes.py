"""
Apply ledger-approved review fixes to a real devocionales-json corpus file.

This is the first script in this pipeline that actually writes to the corpus --
everything upstream (review-collect, run_review_batch_pipeline.py) is validation
-only by design. Scope is entirely ledger-driven: only rows whose status is
exactly "approved_for_apply" are touched. Every other status (validated_not_applied,
drift_needs_review, rejected_*, debatable_hold, needs_human_judgment) is skipped,
so marking a row's status is the actual approval gate -- this script has no
independent judgment of its own.

For each approved row: read the corpus file, locate field_path (recomputed fresh
from entry_id/field via domain/review_prefilter.resolve_field_path -- never taken
on faith from a prior run), apply the approved critic_findings via
domain/fix.run_fix_pass (the same surgical quoted_text -> replacement_text
substitution fix_pass uses in the graph), then re-validate with
domain/validate.validate_field_fix before writing anything.

Every corpus file this touches gets a .bak copy written next to it before the
first write (once per run, not once per field) -- belt-and-suspenders on top of
git, since the corpus repo is expected to be on its own branch with a clean
working tree, not main, precisely so this is trivially revertable either way.

Dry-run by default. Pass --write to actually modify the corpus file on disk.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from content_batch_graph.domain.fix import run_fix_pass
from content_batch_graph.domain.review_prefilter import resolve_field_path
from content_batch_graph.domain.validate import validate_field_fix
from content_batch_graph.state import CriticFinding


def load_approved_rows(ledger_path: str) -> list[dict]:
    rows = []
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["status"] == "approved_for_apply":
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--language-key", default="es")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write changes to --corpus-file. Without this flag, "
        "everything runs (splice + validate) but the file on disk is untouched.",
    )
    args = parser.parse_args()

    approved = load_approved_rows(args.ledger)
    print(f"{len(approved)} ledger row(s) with status=approved_for_apply")
    if not approved:
        print("nothing to apply.")
        return 0

    with open(args.corpus_file, encoding="utf-8") as f:
        document = json.load(f)

    results = []
    for row in approved:
        entry_id, field = row["entry_id"], row["field"]
        field_path = resolve_field_path(document, args.language_key, entry_id, field)
        if field_path is None:
            print(f"  SKIP {entry_id}:{field} — could not resolve field_path "
                  f"in {args.corpus_file} (entry may not exist in this corpus file)")
            continue

        parts = field_path.split(".")
        date, i = parts[2], int(parts[3])
        current_text = document["data"][args.language_key][date][i][field]

        findings = [
            CriticFinding(
                quoted_text=cf["quoted_text"],
                issue="",
                category=cf["category"],
                verified=True,
                is_valid=cf["is_valid"],
                replacement_text=cf.get("replacement_text"),
                critic_reasoning=cf["critic_reasoning"],
            )
            for i2, cf in enumerate(row["critic_findings"])
            if i2 in row["applied_indices"]
        ]

        fixed_text, summary = run_fix_pass(current_text, findings)
        if fixed_text == current_text:
            print(f"  SKIP {entry_id}:{field} — fix produced no change (unexpected)")
            continue

        passed, error = validate_field_fix(args.corpus_file, field_path, fixed_text)
        if not passed:
            print(f"  SKIP {entry_id}:{field} — re-validation failed: {error}")
            continue

        results.append((entry_id, field, field_path, date, i, fixed_text, summary))
        print(f"  OK   {entry_id}:{field} — {summary}")

    print(f"\n{len(results)}/{len(approved)} ready to write")
    if not results:
        return 0

    if not args.write:
        print("\nDRY RUN — no file was modified. Pass --write to apply for real.")
        return 0

    backup_path = Path(args.corpus_file).with_suffix(
        Path(args.corpus_file).suffix + ".bak"
    )
    shutil.copy2(args.corpus_file, backup_path)
    print(f"\nbackup written to {backup_path}")

    for entry_id, field, field_path, date, i, fixed_text, summary in results:
        document["data"][args.language_key][date][i][field] = fixed_text

    with open(args.corpus_file, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {len(results)} fix(es) to {args.corpus_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
