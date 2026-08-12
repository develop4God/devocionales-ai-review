"""
Durable pattern memory: a JSON store of critic-confirmed typo fixes, so a fix seen
once can be proposed (never auto-applied) against the rest of the corpus later,
instead of waiting for flag_pass to rediscover the same typo entry-by-entry.

Only the "typo" category is banked here — an exact, literal string is safe to search
for elsewhere. "awkward_phrasing"/"grammar" fixes are context-dependent rewrites, not
safe to blindly match as a literal substring elsewhere, so they're intentionally
excluded (see domain/scan.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE_PATH = _REPO_ROOT / "data" / "pattern_memory.json"


class PatternEntry(TypedDict):
    quoted_text: str  # the exact confirmed-typo string
    replacement_text: str  # the critic-confirmed correction
    language: str
    category: str  # always "typo" today; other categories are never banked
    source_entry_id: str | None  # which entry this was first confirmed on
    source_file_path: str | None


def load_patterns(store_path: Path = DEFAULT_STORE_PATH) -> list[PatternEntry]:
    """Returns all banked pattern entries, or an empty list if the store doesn't exist."""
    if not store_path.exists():
        return []
    with open(store_path, encoding="utf-8") as f:
        return json.load(f)


def save_pattern(entry: PatternEntry, store_path: Path = DEFAULT_STORE_PATH) -> None:
    """
    Appends entry to the store, skipping if an identical (quoted_text, replacement_text,
    language) entry is already banked — no duplicate accumulation across repeated runs.
    """
    store_path.parent.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns(store_path)

    for existing in patterns:
        if (
            existing["quoted_text"] == entry["quoted_text"]
            and existing["replacement_text"] == entry["replacement_text"]
            and existing["language"] == entry["language"]
        ):
            return

    patterns.append(entry)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)
