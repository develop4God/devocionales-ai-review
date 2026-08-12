"""
Scan a corpus of devotional JSON files for other occurrences of a confirmed typo
pattern, and propose (never apply) the same fix at each match.

Only ever called with typo-category patterns (see domain/pattern_memory.py) — an
exact literal string match is safe to propose as-is; other categories are
context-dependent and are never scanned for.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import TypedDict

from content_batch_graph.domain.pattern_memory import PatternEntry


class ScanMatch(TypedDict):
    file_path: str
    field_path: str  # dot-path to the matching reflexion field
    entry_id: str | None
    quoted_text: str
    replacement_text: str
    occurrences: int  # how many times quoted_text appears in that field


def find_devotional_files(corpus_dir: str, language: str) -> list[str]:
    """
    Returns paths to devotional JSON files for the given language in corpus_dir,
    matching this project's naming convention: Devocional_year_*_<language>_*.json.
    """
    pattern = str(Path(corpus_dir) / f"Devocional_year_*_{language}_*.json")
    return sorted(glob.glob(pattern))


_SCANNED_FIELDS = ("reflexion", "oracion")


def scan_file_for_pattern(
    file_path: str, language: str, pattern: PatternEntry
) -> list[ScanMatch]:
    """
    Reads one devotional JSON file and returns a ScanMatch for every reflexion/oracion
    field containing pattern['quoted_text'] verbatim. Read-only — never modifies the file.
    """
    with open(file_path, encoding="utf-8") as f:
        document = json.load(f)

    matches: list[ScanMatch] = []
    lang_data = document.get("data", {}).get(language, {})
    for date, entries in lang_data.items():
        for i, entry in enumerate(entries):
            for field in _SCANNED_FIELDS:
                text = entry.get(field, "")
                count = text.count(pattern["quoted_text"])
                if count > 0:
                    matches.append(
                        ScanMatch(
                            file_path=file_path,
                            field_path=f"data.{language}.{date}.{i}.{field}",
                            entry_id=entry.get("id"),
                            quoted_text=pattern["quoted_text"],
                            replacement_text=pattern["replacement_text"],
                            occurrences=count,
                        )
                    )
    return matches


def scan_corpus_for_pattern(corpus_dir: str, pattern: PatternEntry) -> list[ScanMatch]:
    """
    Scans every devotional file for pattern['language'] in corpus_dir, returning
    every match across all files/entries. Excludes the pattern's own source file
    at the exact field_path it was confirmed on isn't tracked here — the source
    entry will legitimately re-match too, since the fix hasn't been applied yet;
    callers decide what to do with that.
    """
    language = pattern["language"]
    files = find_devotional_files(corpus_dir, language)
    matches: list[ScanMatch] = []
    for file_path in files:
        matches.extend(scan_file_for_pattern(file_path, language, pattern))
    return matches
