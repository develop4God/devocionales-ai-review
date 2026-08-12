import json
from pathlib import Path

from content_batch_graph.domain.pattern_memory import (
    PatternEntry,
    load_patterns,
    save_pattern,
)


def _entry(
    quoted_text: str = "espirituwal", replacement_text: str = "espiritwal"
) -> PatternEntry:
    return PatternEntry(
        quoted_text=quoted_text,
        replacement_text=replacement_text,
        language="fil",
        category="typo",
        source_entry_id="Juan155ASND20250801",
        source_file_path="/some/path.json",
    )


def test_load_patterns_returns_empty_list_when_store_missing(tmp_path: Path):
    store_path = tmp_path / "does_not_exist.json"
    assert load_patterns(store_path) == []


def test_save_pattern_persists_and_loads_back(tmp_path: Path):
    store_path = tmp_path / "patterns.json"
    save_pattern(_entry(), store_path)

    loaded = load_patterns(store_path)
    assert len(loaded) == 1
    assert loaded[0]["quoted_text"] == "espirituwal"
    assert loaded[0]["replacement_text"] == "espiritwal"


def test_save_pattern_skips_exact_duplicate(tmp_path: Path):
    store_path = tmp_path / "patterns.json"
    save_pattern(_entry(), store_path)
    save_pattern(_entry(), store_path)

    assert len(load_patterns(store_path)) == 1


def test_save_pattern_keeps_distinct_entries(tmp_path: Path):
    store_path = tmp_path / "patterns.json"
    save_pattern(_entry("espirituwal", "espiritwal"), store_path)
    save_pattern(_entry("malinap", "malinaw"), store_path)

    loaded = load_patterns(store_path)
    assert len(loaded) == 2


def test_save_pattern_writes_valid_utf8_json(tmp_path: Path):
    store_path = tmp_path / "patterns.json"
    save_pattern(_entry("café", "cafe"), store_path)

    with open(store_path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw[0]["quoted_text"] == "café"
