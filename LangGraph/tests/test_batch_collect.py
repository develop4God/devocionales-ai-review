import json
from datetime import date

from content_batch_graph.domain.batch_collect import (
    date_from_custom_id,
    extract_content,
    parse_results,
    write_year_collection,
)


def _result_line(custom_id: str, content: str, wrapped: bool = True) -> dict:
    inner = {"choices": [{"message": {"content": content}}]}
    return {"custom_id": custom_id, "response": {"body": inner} if wrapped else inner}


def _good(custom_id: str, reflexion="Una reflexión.", oracion="Una oración.") -> dict:
    return _result_line(
        custom_id, json.dumps({"reflexion": reflexion, "oracion": oracion})
    )


def _write(tmp_path, lines):
    p = tmp_path / "results.jsonl"
    p.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
        encoding="utf-8",
    )
    return p


# ── custom_id ─────────────────────────────────────────────────────────────────


def test_date_from_custom_id_round_trips():
    assert date_from_custom_id("gen_es_RVR1960_2026_03-05") == date(2026, 3, 5)


def test_date_from_custom_id_handles_leap_day():
    assert date_from_custom_id("gen_es_RVR1960_2028_02-29") == date(2028, 2, 29)


def test_date_from_custom_id_rejects_an_impossible_date():
    assert date_from_custom_id("gen_es_RVR1960_2026_02-30") is None


def test_date_from_custom_id_rejects_foreign_ids():
    assert date_from_custom_id("something_else") is None
    assert date_from_custom_id("") is None


# ── extract_content ───────────────────────────────────────────────────────────


def test_extract_content_handles_wrapped_and_unwrapped_shapes():
    assert extract_content(_result_line("x", "hi", wrapped=True)) == "hi"
    assert extract_content(_result_line("x", "hi", wrapped=False)) == "hi"


def test_extract_content_returns_none_on_a_malformed_response():
    assert extract_content({"custom_id": "x"}) is None
    assert extract_content({"custom_id": "x", "response": {"body": {}}}) is None


# ── parse_results ─────────────────────────────────────────────────────────────


def test_parse_results_returns_sorted_records(tmp_path):
    path = _write(
        tmp_path,
        [_good("gen_es_RVR1960_2026_03-05"), _good("gen_es_RVR1960_2026_01-02")],
    )
    records, errors = parse_results(path)
    assert errors == []
    assert [r["date"] for r in records] == ["2026-01-02", "2026-03-05"]
    assert records[0] == {
        "date": "2026-01-02",
        "reflexion": "Una reflexión.",
        "oracion": "Una oración.",
    }


def test_parse_results_strips_markdown_fences_around_json(tmp_path):
    content = '```json\n{"reflexion": "R", "oracion": "O"}\n```'
    path = _write(tmp_path, [_result_line("gen_es_RVR1960_2026_01-01", content)])
    records, errors = parse_results(path)
    assert errors == []
    assert records[0]["reflexion"] == "R"


def test_parse_results_skips_bad_lines_without_losing_good_ones(tmp_path):
    path = _write(
        tmp_path,
        [
            _good("gen_es_RVR1960_2026_01-01"),
            _result_line("gen_es_RVR1960_2026_01-02", "not json at all"),
            _result_line("gen_es_RVR1960_2026_01-03", json.dumps({"reflexion": "R"})),
            _good("bad_custom_id"),
            {"custom_id": "gen_es_RVR1960_2026_01-05"},  # no response content
            _good("gen_es_RVR1960_2026_01-06"),
        ],
    )
    records, errors = parse_results(path)
    assert [r["date"] for r in records] == ["2026-01-01", "2026-01-06"]
    reasons = {e["custom_id"]: e["reason"] for e in errors}
    assert "not a JSON object" in reasons["gen_es_RVR1960_2026_01-02"]
    assert "oracion" in reasons["gen_es_RVR1960_2026_01-03"]
    assert "unparseable custom_id" in reasons["bad_custom_id"]
    assert "missing response content" in reasons["gen_es_RVR1960_2026_01-05"]


def test_parse_results_flags_empty_string_fields(tmp_path):
    path = _write(tmp_path, [_good("gen_es_RVR1960_2026_01-01", reflexion="   ")])
    records, errors = parse_results(path)
    assert records == []
    assert "reflexion" in errors[0]["reason"]


def test_parse_results_records_an_invalid_jsonl_line(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("this is not json\n", encoding="utf-8")
    records, errors = parse_results(path)
    assert records == []
    assert errors[0]["reason"] == "invalid JSONL line"


def test_parse_results_ignores_blank_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(_good("gen_es_RVR1960_2026_01-01")) + "\n\n   \n", encoding="utf-8"
    )
    records, errors = parse_results(path)
    assert len(records) == 1 and errors == []


# ── write_year_collection ─────────────────────────────────────────────────────


def test_write_year_collection_writes_sorted_payload(tmp_path):
    records = [
        {"date": "2026-02-01", "reflexion": "B", "oracion": "b"},
        {"date": "2026-01-01", "reflexion": "A", "oracion": "a"},
    ]
    out = write_year_collection(
        "es", "RVR1960", 2026, records, errors=[], out_path=tmp_path / "c.json"
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["lang"] == "es"
    assert payload["version"] == "RVR1960"
    assert payload["year"] == 2026
    assert payload["count"] == 2
    assert [d["date"] for d in payload["data"]] == ["2026-01-01", "2026-02-01"]


def test_write_year_collection_embeds_the_errors_summary(tmp_path):
    errors = [{"custom_id": "gen_es_RVR1960_2026_01-02", "reason": "boom"}]
    out = write_year_collection(
        "es", "RVR1960", 2026, [], errors=errors, out_path=tmp_path / "c.json"
    )
    assert json.loads(out.read_text(encoding="utf-8"))["errors"] == errors


def test_write_year_collection_defaults_to_the_genomes_path(tmp_path):
    from content_batch_graph.domain import batch_io

    out = write_year_collection("xx", "TEST", 1999, [], errors=[])
    try:
        assert out == batch_io.collection_path("xx", "TEST", 1999)
        assert out.exists()
    finally:
        out.unlink(missing_ok=True)


def test_write_year_collection_preserves_non_ascii(tmp_path):
    out = write_year_collection(
        "es",
        "RVR1960",
        2026,
        [{"date": "2026-01-01", "reflexion": "Señor", "oracion": "Amén"}],
        errors=[],
        out_path=tmp_path / "c.json",
    )
    assert "Señor" in out.read_text(encoding="utf-8")
