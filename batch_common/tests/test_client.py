"""BatchClient transport behavior, against a fake urllib layer — never the network."""

from __future__ import annotations

import json

import pytest
from batch_common.client import BatchClient
from batch_common.config import BatchAPIError
from conftest import http_error


def test_missing_env_var_raises_with_variable_name(cfg, monkeypatch):
    monkeypatch.delenv("FAKE_BATCH_API_KEY", raising=False)
    with pytest.raises(BatchAPIError, match="FAKE_BATCH_API_KEY"):
        BatchClient(cfg)


def test_base_url_trailing_slash_is_normalized(cfg, api_key, fake_http):
    # cfg.base_url ends in "/" — the built URL must not contain "//batches".
    fake_http.responses.append({"id": "batch-1"})
    BatchClient(cfg).submit("file-1")
    assert (
        fake_http.requests[0].full_url
        == "https://api.example.test/inference/v1/batches"
    )


def test_upload_posts_multipart_and_returns_file_id(cfg, api_key, fake_http, tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text('{"custom_id": "a"}\n', encoding="utf-8")
    fake_http.responses.append({"id": "file-abc"})

    file_id = BatchClient(cfg).upload(src)

    assert file_id == "file-abc"
    req = fake_http.requests[0]
    assert req.full_url.endswith("/files")
    assert req.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert req.get_header("Authorization") == "Bearer test-key"
    body = req.data.decode()
    assert 'name="purpose"' in body and "batch" in body
    assert '{"custom_id": "a"}' in body


def test_upload_accepts_file_id_alias(cfg, api_key, fake_http, tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    fake_http.responses.append({"file_id": "file-alias"})
    assert BatchClient(cfg).upload(src) == "file-alias"


def test_upload_without_id_in_response_raises(cfg, api_key, fake_http, tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    fake_http.responses.append({"object": "file"})
    with pytest.raises(BatchAPIError, match="no file_id"):
        BatchClient(cfg).upload(src)


def test_submit_sends_endpoint_and_completion_window_from_config(
    cfg, api_key, fake_http
):
    fake_http.responses.append({"id": "batch-xyz"})

    batch_id = BatchClient(cfg).submit("file-abc")

    assert batch_id == "batch-xyz"
    payload = json.loads(fake_http.requests[0].data.decode())
    assert payload == {
        "input_file_id": "file-abc",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }


def test_submit_without_id_in_response_raises(cfg, api_key, fake_http):
    fake_http.responses.append({"object": "batch"})
    with pytest.raises(BatchAPIError, match="no batch_id"):
        BatchClient(cfg).submit("file-abc")


def test_poll_returns_output_file_id_on_completed(cfg, api_key, fake_http):
    fake_http.responses += [
        {"status": "validating"},
        {"status": "in_progress", "request_counts": {"completed": 3, "total": 10}},
        {"status": "completed", "output_file_id": "file-out"},
    ]
    assert BatchClient(cfg).poll("batch-1", interval=0) == "file-out"
    assert len(fake_http.requests) == 3


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_poll_raises_on_non_completed_terminal_status(cfg, api_key, fake_http, status):
    fake_http.responses.append({"status": status})
    with pytest.raises(BatchAPIError, match=status):
        BatchClient(cfg).poll("batch-1", interval=0)


def test_poll_completed_without_output_file_id_raises(cfg, api_key, fake_http):
    fake_http.responses.append({"status": "completed"})
    with pytest.raises(BatchAPIError, match="output_file_id is missing"):
        BatchClient(cfg).poll("batch-1", interval=0)


def test_poll_times_out(cfg, api_key, fake_http):
    fake_http.responses += [{"status": "in_progress"}] * 5
    with pytest.raises(TimeoutError, match="timed out"):
        BatchClient(cfg).poll("batch-1", interval=0, timeout=0)


def test_download_writes_bytes_and_creates_parent_dir(
    cfg, api_key, fake_http, tmp_path
):
    fake_http.responses.append(b'{"custom_id": "a"}\n')
    dest = tmp_path / "nested" / "results.jsonl"

    out = BatchClient(cfg).download("file-out", dest)

    assert out == dest
    assert dest.read_text(encoding="utf-8") == '{"custom_id": "a"}\n'
    assert fake_http.requests[0].full_url.endswith("/files/file-out/content")


def test_http_error_is_wrapped_as_batch_api_error(cfg, api_key, fake_http):
    fake_http.responses.append(http_error(429, "rate limited"))
    with pytest.raises(BatchAPIError, match="HTTP 429: rate limited"):
        BatchClient(cfg).submit("file-abc")


def test_model_and_provider_id_properties(cfg, api_key, fake_http):
    client = BatchClient(cfg)
    assert client.model == "fake/model"
    assert client.provider_id == "fake_batch"
