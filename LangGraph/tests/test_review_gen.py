import json

from batch_common import BatchProviderConfig

from content_batch_graph.domain.review_gen import (
    ReviewUnit,
    build_review_batch,
    build_review_system,
    build_review_user,
    custom_id_for,
    review_units_from_corpus,
    review_units_from_file,
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


def test_review_system_substitutes_language_and_has_no_json_contract():
    system = build_review_system("es", _role())
    assert "{language}" not in system
    assert "Spanish" in system
    for forbidden in ("reflexion", "oracion", "JSON"):
        assert forbidden not in system


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


# ── build_review_batch ──────────────────────────────────────────────────────────


def _units():
    return [
        ReviewUnit("e1", "reflexion", "Un dia especial."),
        ReviewUnit("e1", "oracion", "Amen."),
    ]


def test_build_review_batch_produces_one_record_per_unit():
    records = build_review_batch(_units(), "es", "RVR1960", PROVIDER, _role())
    assert len(records) == 2
    assert [r["custom_id"] for r in records] == [
        "review_es_RVR1960_e1_reflexion",
        "review_es_RVR1960_e1_oracion",
    ]


def test_build_review_batch_record_body_shape():
    record = build_review_batch(_units(), "es", "RVR1960", PROVIDER, _role())[0]
    assert record["method"] == "POST"
    assert record["url"] == "/v1/chat/completions"
    body = record["body"]
    assert body["model"] == PROVIDER.model
    assert body["max_tokens"] == 1024
    assert body["temperature"] == 0.2
    assert "response_format" not in body
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][1]["content"] == "Text to review:\nUn dia especial."


def test_build_review_batch_shares_one_system_prompt_across_all_units():
    records = build_review_batch(_units(), "es", "RVR1960", PROVIDER, _role())
    systems = {r["body"]["messages"][0]["content"] for r in records}
    assert len(systems) == 1


def test_build_review_batch_honors_overrides():
    record = build_review_batch(
        _units(), "es", "RVR1960", PROVIDER, _role(), max_tokens=256, temperature=0.0
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
    record = build_review_batch(_units(), "es", "RVR1960", provider, _role())[0]
    assert record["body"]["top_p"] == 0.9


def test_build_review_batch_records_are_json_serializable():
    for record in build_review_batch(_units(), "es", "RVR1960", PROVIDER, _role()):
        assert json.loads(json.dumps(record)) == record


def test_build_review_batch_empty_units_produces_no_records():
    assert build_review_batch([], "es", "RVR1960", PROVIDER, _role()) == []
