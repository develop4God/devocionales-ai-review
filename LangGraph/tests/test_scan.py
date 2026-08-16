import json
from pathlib import Path

from content_batch_graph.domain.pattern_memory import PatternEntry
from content_batch_graph.domain.scan import (
    find_devotional_files,
    scan_corpus_for_pattern,
    scan_file_for_pattern,
)


def _write_devotional(
    path: Path,
    language: str,
    date: str,
    entry_id: str,
    reflexion: str = "",
    oracion: str = "",
) -> None:
    data = {
        "data": {
            language: {
                date: [
                    {
                        "id": entry_id,
                        "date": date,
                        "reflexion": reflexion,
                        "oracion": oracion,
                    }
                ]
            }
        }
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _pattern(
    quoted_text: str = "espirituwal", replacement_text: str = "espiritwal"
) -> PatternEntry:
    return PatternEntry(
        quoted_text=quoted_text,
        replacement_text=replacement_text,
        language="fil",
        category="typo",
        source_entry_id=None,
        source_file_path=None,
    )


def test_find_devotional_files_matches_naming_convention(tmp_path: Path):
    (tmp_path / "Devocional_year_2025_fil_ASND.json").write_text("{}")
    (tmp_path / "Devocional_year_2026_fil_MBB05.json").write_text("{}")
    (tmp_path / "Devocional_year_2025_fr_LSG1910.json").write_text("{}")
    (tmp_path / "unrelated.json").write_text("{}")

    files = find_devotional_files(str(tmp_path), "fil")

    assert len(files) == 2
    assert all("_fil_" in f for f in files)


def test_find_devotional_files_includes_legacy_untagged_rvr1960_file_for_es(
    tmp_path: Path,
):
    # es/RVR1960 predates the _{lang}_{version}_ naming convention and carries
    # no tag at all — confirmed against devocionales-json/index.json, which maps
    # this exact filename to es/RVR1960.
    (tmp_path / "Devocional_year_2025.json").write_text("{}")
    (tmp_path / "Devocional_year_2025_es_NVI.json").write_text("{}")

    files = find_devotional_files(str(tmp_path), "es")

    assert len(files) == 2
    names = {Path(f).name for f in files}
    assert names == {"Devocional_year_2025.json", "Devocional_year_2025_es_NVI.json"}


def test_find_devotional_files_legacy_exception_is_es_only(tmp_path: Path):
    # The untagged filename only means RVR1960 for "es" — a same-shaped
    # untagged file under a different language string must not be swept in.
    (tmp_path / "Devocional_year_2025.json").write_text("{}")

    assert find_devotional_files(str(tmp_path), "fr") == []


def test_scan_file_for_pattern_finds_match(tmp_path: Path):
    file_path = tmp_path / "Devocional_year_2025_fil_ASND.json"
    _write_devotional(
        file_path, "fil", "2025-08-01", "entry1", "May espirituwal na kahulugan ito."
    )

    matches = scan_file_for_pattern(str(file_path), "fil", _pattern())

    assert len(matches) == 1
    assert matches[0]["entry_id"] == "entry1"
    assert matches[0]["field_path"] == "data.fil.2025-08-01.0.reflexion"
    assert matches[0]["occurrences"] == 1
    assert matches[0]["replacement_text"] == "espiritwal"


def test_scan_file_for_pattern_counts_multiple_occurrences(tmp_path: Path):
    file_path = tmp_path / "Devocional_year_2025_fil_ASND.json"
    _write_devotional(
        file_path,
        "fil",
        "2025-08-01",
        "entry1",
        "espirituwal at espirituwal na buhay.",
    )

    matches = scan_file_for_pattern(str(file_path), "fil", _pattern())

    assert len(matches) == 1
    assert matches[0]["occurrences"] == 2


def test_scan_file_for_pattern_matches_oracion_too(tmp_path: Path):
    file_path = tmp_path / "Devocional_year_2025_fil_ASND.json"
    _write_devotional(
        file_path,
        "fil",
        "2025-08-01",
        "entry1",
        reflexion="Walang typo dito.",
        oracion="Panalangin na may espirituwal na kahulugan.",
    )

    matches = scan_file_for_pattern(str(file_path), "fil", _pattern())

    assert len(matches) == 1
    assert matches[0]["field_path"] == "data.fil.2025-08-01.0.oracion"


def test_scan_file_for_pattern_matches_both_fields_independently(tmp_path: Path):
    file_path = tmp_path / "Devocional_year_2025_fil_ASND.json"
    _write_devotional(
        file_path,
        "fil",
        "2025-08-01",
        "entry1",
        reflexion="May espirituwal na kahulugan ito.",
        oracion="Panalangin na may espirituwal na kahulugan.",
    )

    matches = scan_file_for_pattern(str(file_path), "fil", _pattern())

    assert len(matches) == 2
    assert {m["field_path"] for m in matches} == {
        "data.fil.2025-08-01.0.reflexion",
        "data.fil.2025-08-01.0.oracion",
    }


def test_scan_file_for_pattern_returns_empty_when_no_match(tmp_path: Path):
    file_path = tmp_path / "Devocional_year_2025_fil_ASND.json"
    _write_devotional(file_path, "fil", "2025-08-01", "entry1", "Walang typo dito.")

    matches = scan_file_for_pattern(str(file_path), "fil", _pattern())

    assert matches == []


def test_scan_file_for_pattern_does_not_modify_the_file(tmp_path: Path):
    file_path = tmp_path / "Devocional_year_2025_fil_ASND.json"
    original_text = "May espirituwal na kahulugan ito."
    _write_devotional(file_path, "fil", "2025-08-01", "entry1", original_text)

    scan_file_for_pattern(str(file_path), "fil", _pattern())

    with open(file_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["data"]["fil"]["2025-08-01"][0]["reflexion"] == original_text


def test_scan_corpus_for_pattern_covers_multiple_files(tmp_path: Path):
    _write_devotional(
        tmp_path / "Devocional_year_2025_fil_ASND.json",
        "fil",
        "2025-08-01",
        "entry1",
        "espirituwal na buhay.",
    )
    _write_devotional(
        tmp_path / "Devocional_year_2025_fil_MBB05.json",
        "fil",
        "2025-08-02",
        "entry2",
        "isa pang espirituwal na sanggunian.",
    )
    _write_devotional(
        tmp_path / "Devocional_year_2025_fr_LSG1910.json",
        "fr",
        "2025-08-01",
        "entry3",
        "espirituwal should never match a French file's language key.",
    )

    matches = scan_corpus_for_pattern(str(tmp_path), _pattern())

    assert len(matches) == 2
    assert {m["entry_id"] for m in matches} == {"entry1", "entry2"}
