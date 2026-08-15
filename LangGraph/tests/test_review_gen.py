import json

import pytest
from batch_common import BatchProviderConfig

from content_batch_graph.domain.review_gen import (
    ReviewUnit,
    build_review_batch,
    build_review_response_format,
    build_review_system,
    build_review_user,
    custom_id_for,
    review_units_from_corpus,
    review_units_from_file,
    version_from_filename,
)
from content_batch_graph.domain.roles import get_role

PROVIDER = BatchProviderConfig(
    provider_id="p_batch",
    base_url="https://x.test/v1",
    model="accounts/x/models/m1",
    env_var="X_API_KEY",
)


def _role():
    return get_role("native_reader_batch")


def _write_corpus_file(tmp_path, lang="es", version="RVR1960"):
    doc = {
        "data": {
            lang: {
                "2026-01-01": [
                    {"id": "e1", "reflexion": "Un dia especial.", "oracion": "Amen."},
                    {"id": "e2", "reflexion": "", "oracion": "Otra oracion."},
                ],
                "2026-01-02": [
                    {"id": "e3", "reflexion": "Otro dia."},
                    {"reflexion": "Entrada sin id."},
                ],
            }
        }
    }
    path = tmp_path / f"Devocional_year_2026_{lang}_{version}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


# ── prompts ───────────────────────────────────────────────────────────────────


def test_review_system_substitutes_language_and_describes_the_findings_contract():
    system = build_review_system("es", _role())
    assert "{language}" not in system
    assert "Spanish" in system
    assert "findings" in system
    for forbidden in ("reflexion", "oracion"):
        assert forbidden not in system


def test_review_response_format_matches_the_roles_finding_schema():
    response_format = build_review_response_format(_role())
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    finding_props = next(iter(schema["$defs"].values()))["properties"]
    assert set(finding_props) == {"quoted_text", "issue", "proposed_text", "category"}
    assert finding_props["category"]["enum"] == ["typo", "grammar"]


def test_review_system_is_identical_across_units_for_caching():
    a = build_review_system("es", _role())
    b = build_review_system("es", _role())
    assert a == b


def test_review_user_carries_only_the_text_to_review():
    user = build_review_user("Un texto con un error de tipeo.")
    assert "Un texto con un error de tipeo." in user
    assert "Text to review" in user


def test_custom_id_uses_review_prefix_distinct_from_generation():
    cid = custom_id_for("es", "RVR1960", "e1", "reflexion")
    assert cid == "review_es_RVR1960_e1_reflexion"
    assert cid.startswith("review_")


# ── review_units_from_file / corpus ─────────────────────────────────────────────


def test_review_units_from_file_collects_non_empty_fields_only(tmp_path):
    path = _write_corpus_file(tmp_path)
    units = review_units_from_file(str(path), "es")

    # e2's empty reflexion is skipped; e2's oracion is kept.
    keys = {(u.entry_id, u.field) for u in units}
    assert ("e1", "reflexion") in keys
    assert ("e1", "oracion") in keys
    assert ("e2", "reflexion") not in keys
    assert ("e2", "oracion") in keys


def test_review_units_from_file_skips_entries_without_an_id(tmp_path):
    path = _write_corpus_file(tmp_path)
    units = review_units_from_file(str(path), "es")
    assert all(u.entry_id for u in units)
    assert not any(u.text == "Entrada sin id." for u in units)


def test_review_units_from_file_returns_empty_for_missing_language(tmp_path):
    path = _write_corpus_file(tmp_path, lang="es")
    assert review_units_from_file(str(path), "fr") == []


def test_review_units_from_corpus_aggregates_across_files(tmp_path):
    _write_corpus_file(tmp_path, lang="es", version="RVR1960")
    _write_corpus_file(tmp_path, lang="es", version="NVI")
    units = review_units_from_corpus(str(tmp_path), "es")
    # 4 non-empty fields per file (e1 reflexion+oracion, e2 oracion, e3 reflexion)
    # x 2 files
    assert len(units) == 8


def test_review_units_from_corpus_labels_each_unit_with_its_own_version(tmp_path):
    _write_corpus_file(tmp_path, lang="es", version="RVR1960")
    _write_corpus_file(tmp_path, lang="es", version="NVI")
    units = review_units_from_corpus(str(tmp_path), "es")
    versions = {u.version for u in units}
    assert versions == {"RVR1960", "NVI"}


# ── version_from_filename ───────────────────────────────────────────────────────


def test_version_from_filename_reads_the_tagged_version():
    assert version_from_filename("Devocional_year_2025_es_NVI.json") == "NVI"
    assert version_from_filename("Devocional_year_2025_pt_ARC.json") == "ARC"


def test_version_from_filename_falls_back_to_rvr1960_for_the_legacy_untagged_file():
    assert version_from_filename("Devocional_year_2025.json") == "RVR1960"
    assert version_from_filename("/some/dir/Devocional_year_2026.json") == "RVR1960"


def test_version_from_filename_raises_for_unrecognized_names():
    with pytest.raises(ValueError):
        version_from_filename("not_a_devotional_file.json")


# ── build_review_batch ──────────────────────────────────────────────────────────


def _units():
    return [
        ReviewUnit("e1", "reflexion", "Un dia especial.", "RVR1960"),
        ReviewUnit("e1", "oracion", "Amen.", "RVR1960"),
    ]


def test_build_review_batch_produces_one_record_per_unit():
    records = build_review_batch(_units(), "es", PROVIDER, _role())
    assert len(records) == 2
    assert [r["custom_id"] for r in records] == [
        "review_es_RVR1960_e1_reflexion",
        "review_es_RVR1960_e1_oracion",
    ]


def test_build_review_batch_labels_each_record_with_its_own_units_version():
    units = [
        ReviewUnit("e1", "reflexion", "Un dia especial.", "RVR1960"),
        ReviewUnit("e2", "reflexion", "Otro dia.", "NVI"),
    ]
    records = build_review_batch(units, "es", PROVIDER, _role())
    assert [r["custom_id"] for r in records] == [
        "review_es_RVR1960_e1_reflexion",
        "review_es_NVI_e2_reflexion",
    ]


def test_build_review_batch_record_body_shape():
    record = build_review_batch(_units(), "es", PROVIDER, _role())[0]
    body = record["body"]
    # No method/url (old, wrong OpenAI-shaped envelope) and no model in body
    # (Fireworks sets model at job-submission time, not per dataset line).
    assert "method" not in record
    assert "url" not in record
    assert "model" not in body
    assert body["max_tokens"] == 4098
    assert body["temperature"] == 0.2
    # response_format enforces the Finding JSON schema on every record — a real
    # batch test confirmed this eliminates whole-paragraph quoted_text echoes
    # and Markdown drift that the old free-text format produced (see
    # roles.build_finding_schema / review_gen.build_review_response_format).
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]["schema"]
    finding_props = next(iter(schema["$defs"].values()))["properties"]
    assert set(finding_props) == {"quoted_text", "issue", "proposed_text", "category"}
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user"]
    assert body["messages"][0]["content"] == "Text to review:\nUn dia especial."


def test_build_review_batch_never_embeds_a_system_message_per_unit():
    # The shared system prompt (build_review_system) travels as the batch job's
    # own system_prompt at submit time -- never repeated per dataset line here.
    records = build_review_batch(_units(), "es", PROVIDER, _role())
    for record in records:
        assert all(m["role"] != "system" for m in record["body"]["messages"])


def test_build_review_batch_honors_overrides():
    record = build_review_batch(
        _units(), "es", PROVIDER, _role(), max_tokens=256, temperature=0.0
    )[0]
    assert record["body"]["max_tokens"] == 256
    assert record["body"]["temperature"] == 0.0


def test_build_review_batch_merges_provider_extra_record_fields():
    provider = BatchProviderConfig(
        provider_id="p",
        base_url="https://x.test/v1",
        model="m",
        env_var="K",
        extra_record_fields={"top_p": 0.9},
    )
    record = build_review_batch(_units(), "es", provider, _role())[0]
    assert record["body"]["top_p"] == 0.9


def test_build_review_batch_records_are_json_serializable():
    for record in build_review_batch(_units(), "es", PROVIDER, _role()):
        assert json.loads(json.dumps(record)) == record


def test_build_review_batch_empty_units_produces_no_records():
    assert build_review_batch([], "es", PROVIDER, _role()) == []
