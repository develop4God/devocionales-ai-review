"""JSONL round-trips and batch request-record shaping."""

from __future__ import annotations

import json

import pytest
from batch_common.jsonl import chat_request_record, read_jsonl, write_jsonl


def test_write_then_read_round_trips_records(tmp_path):
    records = [{"custom_id": "a", "n": 1}, {"custom_id": "b", "n": 2}]
    path = write_jsonl(tmp_path / "out.jsonl", records)
    assert list(read_jsonl(path)) == records


def test_write_jsonl_creates_missing_parent_dirs(tmp_path):
    path = write_jsonl(tmp_path / "deep" / "nested" / "out.jsonl", [{"a": 1}])
    assert path.exists()


def test_write_jsonl_preserves_non_ascii_unescaped(tmp_path):
    path = write_jsonl(tmp_path / "out.jsonl", [{"t": "oración Ñ"}])
    assert "oración Ñ" in path.read_text(encoding="utf-8")


def test_write_jsonl_emits_one_object_per_line(tmp_path):
    path = write_jsonl(tmp_path / "out.jsonl", [{"a": 1}, {"a": 2}, {"a": 3}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line) for line in lines)


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_raises_on_malformed_line(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(path))


def test_chat_request_record_shape():
    messages = [{"role": "user", "content": "hi"}]
    rec = chat_request_record("gen_es_RVR1960_2026_01-01", "m/1", messages)
    assert rec == {
        "custom_id": "gen_es_RVR1960_2026_01-01",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "m/1", "messages": messages},
    }


def test_chat_request_record_merges_extra_body_fields():
    rec = chat_request_record(
        "id-1",
        "m/1",
        [],
        max_tokens=2048,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    assert rec["body"]["max_tokens"] == 2048
    assert rec["body"]["temperature"] == 0.7
    assert rec["body"]["response_format"] == {"type": "json_object"}


def test_chat_request_record_honors_custom_endpoint():
    rec = chat_request_record("id-1", "m/1", [], endpoint="/v1/responses")
    assert rec["url"] == "/v1/responses"
