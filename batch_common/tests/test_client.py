"""
BatchClient transport behavior against Fireworks' account-scoped batch API, using
a fake urllib layer — never the network. See fireworks_template.py for the
provider shape this client wraps.
"""

from __future__ import annotations

import json

import pytest
from conftest import http_error

from batch_common.client import BatchClient
from batch_common.config import BatchAPIError


def test_missing_api_key_raises_with_variable_name(cfg, account_id, monkeypatch):
    monkeypatch.delenv("FAKE_BATCH_API_KEY", raising=False)
    with pytest.raises(BatchAPIError, match="FAKE_BATCH_API_KEY"):
        BatchClient(cfg)


def test_missing_account_id_raises_with_variable_name(cfg, api_key, monkeypatch):
    monkeypatch.delenv("FAKE_BATCH_ACCOUNT_ID", raising=False)
    with pytest.raises(BatchAPIError, match="FAKE_BATCH_ACCOUNT_ID"):
        BatchClient(cfg)


def test_base_url_trailing_slash_is_normalized(cfg, api_key, account_id, fake_http):
    # cfg.base_url ends in "/" — the built URL must not contain a doubled slash
    # before "accounts".
    fake_http.responses.append({"datasetId": "ds-1"})
    BatchClient(cfg).create_dataset("ds-1")
    assert (
        fake_http.requests[0].full_url
        == "https://api.example.test/inference/v1/accounts/test-account/datasets"
    )


def test_create_dataset_posts_the_documented_body(cfg, api_key, account_id, fake_http):
    fake_http.responses.append({"datasetId": "batch-input-dataset"})
    dataset_id = BatchClient(cfg).create_dataset("batch-input-dataset")

    assert dataset_id == "batch-input-dataset"
    req = fake_http.requests[0]
    assert req.full_url.endswith("/accounts/test-account/datasets")
    assert req.get_header("Authorization") == "Bearer test-key"
    assert json.loads(req.data.decode()) == {
        "datasetId": "batch-input-dataset",
        "dataset": {"userUploaded": {}},
    }


def test_upload_posts_multipart_to_the_dataset_upload_endpoint(
    cfg, api_key, account_id, fake_http, tmp_path
):
    src = tmp_path / "in.jsonl"
    src.write_text('{"custom_id": "a"}\n', encoding="utf-8")
    fake_http.responses.append({})

    dataset_id = BatchClient(cfg).upload("batch-input-dataset", src)

    assert dataset_id == "batch-input-dataset"
    req = fake_http.requests[0]
    assert req.full_url.endswith(
        "/accounts/test-account/datasets/batch-input-dataset:upload"
    )
    assert req.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert req.get_header("Authorization") == "Bearer test-key"
    body = req.data.decode()
    assert 'name="file"' in body
    assert '{"custom_id": "a"}' in body


def test_submit_posts_model_and_dataset_ids(cfg, api_key, account_id, fake_http):
    fake_http.responses.append({})
    job_id = BatchClient(cfg).submit(
        "batch-input-dataset", "batch-output-dataset", "my-batch-job"
    )

    assert job_id == "my-batch-job"
    req = fake_http.requests[0]
    assert req.full_url.endswith(
        "/accounts/test-account/batchInferenceJobs?batchInferenceJobId=my-batch-job"
    )
    payload = json.loads(req.data.decode())
    assert payload == {
        "model": "fake/model",
        "inputDatasetId": "accounts/test-account/datasets/batch-input-dataset",
        "outputDatasetId": "accounts/test-account/datasets/batch-output-dataset",
    }


def test_submit_includes_system_prompt_and_inference_parameters(
    cfg, api_key, account_id, fake_http
):
    fake_http.responses.append({})
    BatchClient(cfg).submit(
        "in-ds",
        "out-ds",
        "job-1",
        system_prompt="You are a helpful assistant.",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )

    payload = json.loads(fake_http.requests[0].data.decode())
    assert payload["systemPrompt"] == "You are a helpful assistant."
    assert payload["inferenceParameters"] == {
        "maxTokens": 1024,
        "temperature": 0.7,
        "topP": 0.9,
    }


def test_submit_omits_system_prompt_and_inference_parameters_when_not_given(
    cfg, api_key, account_id, fake_http
):
    fake_http.responses.append({})
    BatchClient(cfg).submit("in-ds", "out-ds", "job-1")

    payload = json.loads(fake_http.requests[0].data.decode())
    assert "systemPrompt" not in payload
    assert "inferenceParameters" not in payload


def test_poll_returns_completed_state(cfg, api_key, account_id, fake_http):
    fake_http.responses += [
        {"state": "VALIDATING"},
        {"state": "RUNNING"},
        {"state": "COMPLETED"},
    ]
    assert BatchClient(cfg).poll("job-1", interval=0) == "COMPLETED"
    assert len(fake_http.requests) == 3
    assert fake_http.requests[0].full_url.endswith(
        "/accounts/test-account/batchInferenceJobs/job-1"
    )


@pytest.mark.parametrize("state", ["FAILED", "EXPIRED"])
def test_poll_raises_on_a_failure_state(cfg, api_key, account_id, fake_http, state):
    fake_http.responses.append({"state": state})
    with pytest.raises(BatchAPIError, match=state):
        BatchClient(cfg).poll("job-1", interval=0)


def test_poll_times_out(cfg, api_key, account_id, fake_http):
    fake_http.responses += [{"state": "RUNNING"}] * 5
    with pytest.raises(TimeoutError, match="timed out"):
        BatchClient(cfg).poll("job-1", interval=0, timeout=0)


def test_download_fetches_the_manifest_then_every_signed_url(
    cfg, api_key, account_id, fake_http, tmp_path
):
    fake_http.responses += [
        {
            "filenameToSignedUrls": {
                "results.jsonl": "https://signed.example.test/results",
                "errors.jsonl": "https://signed.example.test/errors",
            }
        },
        b'{"custom_id": "a"}\n',
        b"",
    ]
    dest_dir = tmp_path / "out"

    written = BatchClient(cfg).download("batch-output-dataset", dest_dir)

    assert len(written) == 2
    assert {p.name for p in written} == {"results.jsonl", "errors.jsonl"}
    assert (dest_dir / "results.jsonl").read_text(encoding="utf-8") == (
        '{"custom_id": "a"}\n'
    )
    manifest_req = fake_http.requests[0]
    assert manifest_req.full_url.endswith(
        "/accounts/test-account/datasets/batch-output-dataset:getDownloadEndpoint"
    )


def test_download_raises_when_manifest_has_no_signed_urls(
    cfg, api_key, account_id, fake_http, tmp_path
):
    fake_http.responses.append({"filenameToSignedUrls": {}})
    with pytest.raises(BatchAPIError, match="No filenameToSignedUrls"):
        BatchClient(cfg).download("batch-output-dataset", tmp_path / "out")


def test_http_error_is_wrapped_as_batch_api_error(cfg, api_key, account_id, fake_http):
    fake_http.responses.append(http_error(429, "rate limited"))
    with pytest.raises(BatchAPIError, match="HTTP 429: rate limited"):
        BatchClient(cfg).submit("in-ds", "out-ds", "job-1")


def test_model_and_provider_id_properties(cfg, api_key, account_id, fake_http):
    client = BatchClient(cfg)
    assert client.model == "fake/model"
    assert client.provider_id == "fake_batch"
