"""
Standalone script (not part of the graph): given a critic-confirmed typo finding,
bank it in durable pattern memory, scan the rest of a language's corpus for the same
literal string, and write a proposal report. Never applies anything automatically —
a human reviews the report and decides what to fix, matching this project's rule
that recurring patterns are proposed, never written without explicit confirmation.

Usage:
    uv run python scripts/propose_corpus_fixes.py \
        --corpus-dir /path/to/devocionales-json \
        --language fil \
        --quoted-text "espirituwal" \
        --replacement-text "espiritwal" \
        --category typo \
        --source-entry-id Juan155ASND20250801 \
        --source-file-path /path/to/Devocional_year_2025_fil_ASND.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from content_batch_graph.domain.pattern_memory import PatternEntry, save_pattern
from content_batch_graph.domain.scan import scan_corpus_for_pattern


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--quoted-text", required=True)
    parser.add_argument("--replacement-text", required=True)
    parser.add_argument("--category", default="typo")
    parser.add_argument("--source-entry-id", default=None)
    parser.add_argument("--source-file-path", default=None)
    parser.add_argument("--report-path", default="data/corpus_fix_proposals.json")
    args = parser.parse_args()

    if args.category != "typo":
        print(
            f"Refusing: category '{args.category}' is not safe for a literal "
            "corpus-wide scan. Only 'typo' findings (exact strings) are scanned.",
            file=sys.stderr,
        )
        sys.exit(1)

    pattern = PatternEntry(
        quoted_text=args.quoted_text,
        replacement_text=args.replacement_text,
        language=args.language,
        category=args.category,
        source_entry_id=args.source_entry_id,
        source_file_path=args.source_file_path,
    )
    save_pattern(pattern)

    matches = scan_corpus_for_pattern(args.corpus_dir, pattern)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"Banked pattern: {args.quoted_text!r} -> {args.replacement_text!r}")
    print(f"Scanned corpus for language '{args.language}': {len(matches)} match(es)")
    for m in matches:
        print(
            f"  - {m['file_path']} [{m['field_path']}] "
            f"(entry_id={m['entry_id']}, occurrences={m['occurrences']})"
        )
    print(f"\nProposal report written to {report_path} — nothing was applied.")


if __name__ == "__main__":
    main()
