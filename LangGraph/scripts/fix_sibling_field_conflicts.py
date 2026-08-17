"""
One-time correction: 3 ledger rows were flipped to "approved_for_apply" by
mark_pt_apply_ready.py before the sibling-field blind spot was found (the
critic, drift_check, and native review all judge a fix using only the field
being edited, never checking the same entry's "versiculo" field -- so a
reflexion/oracion phrase that intentionally echoes its own entry's scripture
quote can get "corrected" into text that no longer matches the verse).

This script sets those 3 rows' status to "rejected_scripture_echo_conflict"
(new, distinct from every other status so the reason is traceable in the
ledger itself) so apply_review_fixes.py's approved_for_apply filter skips
them, without touching the other 35 already-correct approved rows.

Usage:
    uv run python scripts/fix_sibling_field_conflicts.py --ledger <path> [--write]
"""

from __future__ import annotations

import argparse
import json

CONFLICTS = {
    ("mateus1926ARC20260731", "reflexion"),
    ("filipenses234ARC20260523", "reflexion"),
    ("filipenses234ARC20260523", "oracion"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rows = []
    with open(args.ledger, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    changed = 0
    for row in rows:
        key = (row["entry_id"], row["field"])
        if key in CONFLICTS and row["status"] == "approved_for_apply":
            print(f"  {row['entry_id']}:{row['field']} "
                  f"approved_for_apply -> rejected_scripture_echo_conflict")
            row["status"] = "rejected_scripture_echo_conflict"
            changed += 1

    print(f"\n{changed} row(s) corrected")

    if not args.write:
        print("DRY RUN — ledger not modified. Pass --write to apply for real.")
        return 0

    with open(args.ledger, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote corrected ledger to {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
