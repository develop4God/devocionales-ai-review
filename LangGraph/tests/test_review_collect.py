import json

from content_batch_graph.domain.review_collect import (
    parse_custom_id,
    parse_review_results,
    write_review_collection,
)
from content_batch_graph.domain.roles import get_role


def _role():
    return get_role("native_reader_batch")


def _result_line(custom_id: str, content: str) -> dict:
    return {
        "custom_id": custom_id,
        "response": {
            "choices": [{"message": {"content": content}}],
        },
    }


def _findings_json(findings: list[dict]) -> str:
    return json.dumps({"findings": findings})


def _write(tmp_path, lines):
    p = tmp_path / "results.jsonl"
    p.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines),
        encoding="utf-8",
    )
    return p


# ── parse_custom_id ──────────────────────────────────────────────────────────────


def test_parse_custom_id_recovers_all_four_parts():
    parts = parse_custom_id("review_es_RVR1960_juan1516_reflexion")
    assert parts == {
        "lang": "es",
        "version": "RVR1960",
        "entry_id": "juan1516",
        "field": "reflexion",
    }


def test_parse_custom_id_handles_an_entry_id_with_underscores():
    # The real hazard this regex exists for: entry_id itself contains
    # underscores (filipenses2_9-11_20250616_RVR1960), so a naive split("_")
    # can't tell where entry_id ends and version begins.
    parts = parse_custom_id(
        "review_es_RVR1960_filipenses2_9-11_20250616_RVR1960_oracion"
    )
    assert parts["lang"] == "es"
    assert parts["version"] == "RVR1960"
    assert parts["entry_id"] == "filipenses2_9-11_20250616_RVR1960"
    assert parts["field"] == "oracion"


def test_parse_custom_id_rejects_foreign_ids():
    assert parse_custom_id("gen_es_RVR1960_2026_03-05") is None
    assert parse_custom_id("") is None


# ── parse_review_results ─────────────────────────────────────────────────────────


def test_parse_review_results_collects_findings(tmp_path):
    path = _write(
        tmp_path,
        [
            _result_line(
                "review_es_RVR1960_e1_reflexion",
                _findings_json(
                    [
                        {
                            "quoted_text": "un dia",
                            "issue": "missing accent",
                            "proposed_text": "un día",
                            "category": "typo",
                        }
                    ]
                ),
            )
        ],
    )
    results, errors = parse_review_results(path, _role())

    assert errors == []
    assert len(results) == 1
    r = results[0]
    assert (r.lang, r.version, r.entry_id, r.field) == ("es", "RVR1960", "e1", "reflexion")
    assert len(r.findings) == 1
    assert r.findings[0]["quoted_text"] == "un dia"
    assert r.findings[0]["proposed_text"] == "un día"
    assert r.findings[0]["category"] == "typo"


def test_parse_review_results_handles_empty_findings(tmp_path):
    path = _write(
        tmp_path, [_result_line("review_es_RVR1960_e1_oracion", _findings_json([]))]
    )
    results, errors = parse_review_results(path, _role())
    assert errors == []
    assert results[0].findings == []


def test_parse_review_results_records_invalid_jsonl_line(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text("not valid json\n", encoding="utf-8")
    results, errors = parse_review_results(path, _role())
    assert results == []
    assert errors[0]["reason"] == "invalid JSONL line"


def test_parse_review_results_records_unparseable_custom_id(tmp_path):
    path = _write(tmp_path, [_result_line("not_a_review_id", _findings_json([]))])
    results, errors = parse_review_results(path, _role())
    assert results == []
    assert errors[0]["reason"] == "unparseable custom_id"


def test_parse_review_results_records_missing_content(tmp_path):
    line = {"custom_id": "review_es_RVR1960_e1_reflexion", "response": {}}
    path = _write(tmp_path, [line])
    results, errors = parse_review_results(path, _role())
    assert results == []
    assert errors[0]["reason"] == "missing response content"


def test_parse_review_results_records_schema_mismatch(tmp_path):
    # category "not_a_real_category" isn't in native_reader_batch's declared
    # categories (typo, grammar) -- the schema must reject it, not silently
    # accept an out-of-band category.
    path = _write(
        tmp_path,
        [
            _result_line(
                "review_es_RVR1960_e1_reflexion",
                _findings_json(
                    [
                        {
                            "quoted_text": "x",
                            "issue": "y",
                            "category": "not_a_real_category",
                        }
                    ]
                ),
            )
        ],
    )
    results, errors = parse_review_results(path, _role())
    assert results == []
    assert "Finding schema" in errors[0]["reason"]


def test_parse_review_results_records_malformed_content_json(tmp_path):
    path = _write(
        tmp_path, [_result_line("review_es_RVR1960_e1_reflexion", "not json at all")]
    )
    results, errors = parse_review_results(path, _role())
    assert results == []
    assert "Finding schema" in errors[0]["reason"]


def test_parse_review_results_one_bad_line_does_not_lose_the_rest(tmp_path):
    path = _write(
        tmp_path,
        [
            _result_line("garbage", _findings_json([])),
            _result_line("review_es_RVR1960_e1_reflexion", _findings_json([])),
        ],
    )
    results, errors = parse_review_results(path, _role())
    assert len(results) == 1
    assert len(errors) == 1


def test_parse_review_results_skips_blank_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    good = json.dumps(_result_line("review_es_RVR1960_e1_reflexion", _findings_json([])))
    path.write_text(f"\n{good}\n\n", encoding="utf-8")
    results, errors = parse_review_results(path, _role())
    assert len(results) == 1
    assert errors == []


# ── write_review_collection ───────────────────────────────────────────────────


def test_write_review_collection_writes_findings_marked_unverified(tmp_path):
    path = _write(
        tmp_path,
        [
            _result_line(
                "review_es_RVR1960_e1_reflexion",
                _findings_json(
                    [
                        {
                            "quoted_text": "x",
                            "issue": "y",
                            "proposed_text": "z",
                            "category": "typo",
                        }
                    ]
                ),
            )
        ],
    )
    results, errors = parse_review_results(path, _role())
    out = tmp_path / "collected.json"
    written = write_review_collection("es", results, errors, out_path=out)

    assert written == out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["lang"] == "es"
    assert payload["count"] == 1
    assert payload["errors"] == []
    entry = payload["data"][0]
    assert entry["entry_id"] == "e1"
    assert entry["field"] == "reflexion"
    assert entry["version"] == "RVR1960"
    assert entry["findings"][0]["verified"] is False
    assert entry["findings"][0]["quoted_text"] == "x"


def test_write_review_collection_embeds_the_errors_summary(tmp_path):
    out = tmp_path / "collected.json"
    write_review_collection(
        "es", [], errors=[{"custom_id": "x", "reason": "bad"}], out_path=out
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["errors"] == [{"custom_id": "x", "reason": "bad"}]
    assert payload["count"] == 0
