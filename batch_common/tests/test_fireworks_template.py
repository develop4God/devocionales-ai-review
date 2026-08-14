"""
fireworks_template.py: pure URL/payload construction and response parsing,
no network I/O — verified against docs.fireworks.ai/guides/batch-inference.
"""

from __future__ import annotations

from batch_common import fireworks_template as fw

BASE = "https://api.fireworks.ai/inference/v1"
ACCOUNT = "my-account"


def test_accounts_url_strips_trailing_slash_on_base():
    assert fw.accounts_url(f"{BASE}/", ACCOUNT) == f"{BASE}/accounts/{ACCOUNT}"


def test_create_dataset_request_shape():
    url, body = fw.create_dataset_request(BASE, ACCOUNT, "batch-input-dataset")
    assert url == f"{BASE}/accounts/{ACCOUNT}/datasets"
    assert body == {
        "datasetId": "batch-input-dataset",
        "dataset": {"userUploaded": {}},
    }


def test_upload_dataset_url():
    url = fw.upload_dataset_url(BASE, ACCOUNT, "batch-input-dataset")
    assert url == f"{BASE}/accounts/{ACCOUNT}/datasets/batch-input-dataset:upload"


def test_submit_job_request_minimal():
    url, body = fw.submit_job_request(
        BASE,
        ACCOUNT,
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "my-batch-job",
        "batch-input-dataset",
        "batch-output-dataset",
    )
    assert url == (
        f"{BASE}/accounts/{ACCOUNT}/batchInferenceJobs?batchInferenceJobId=my-batch-job"
    )
    assert body == {
        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "inputDatasetId": f"accounts/{ACCOUNT}/datasets/batch-input-dataset",
        "outputDatasetId": f"accounts/{ACCOUNT}/datasets/batch-output-dataset",
    }
    assert "systemPrompt" not in body
    assert "inferenceParameters" not in body


def test_submit_job_request_with_system_prompt_and_inference_parameters():
    _url, body = fw.submit_job_request(
        BASE,
        ACCOUNT,
        "m",
        "job-1",
        "in-ds",
        "out-ds",
        system_prompt="You are a helpful assistant.",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )
    assert body["systemPrompt"] == "You are a helpful assistant."
    assert body["inferenceParameters"] == {
        "maxTokens": 1024,
        "temperature": 0.7,
        "topP": 0.9,
    }


def test_submit_job_request_omits_unset_inference_parameters():
    _url, body = fw.submit_job_request(
        BASE, ACCOUNT, "m", "job-1", "in-ds", "out-ds", temperature=0.2
    )
    assert body["inferenceParameters"] == {"temperature": 0.2}


def test_job_status_url():
    url = fw.job_status_url(BASE, ACCOUNT, "my-batch-job")
    assert url == f"{BASE}/accounts/{ACCOUNT}/batchInferenceJobs/my-batch-job"


def test_parse_job_status_returns_state():
    assert fw.parse_job_status({"state": "RUNNING"}) == "RUNNING"


def test_parse_job_status_defaults_to_unknown_when_missing():
    assert fw.parse_job_status({}) == "unknown"


def test_is_failure_state_matches_failed_and_expired():
    assert fw.is_failure_state("FAILED") is True
    assert fw.is_failure_state("EXPIRED") is True


def test_is_failure_state_false_for_completed_and_in_progress_states():
    for state in ("VALIDATING", "PENDING", "RUNNING", "COMPLETED"):
        assert fw.is_failure_state(state) is False


def test_download_manifest_request_url():
    url = fw.download_manifest_request(BASE, ACCOUNT, "batch-output-dataset")
    assert url == (
        f"{BASE}/accounts/{ACCOUNT}/datasets/batch-output-dataset:getDownloadEndpoint"
    )


def test_parse_download_manifest_returns_filename_to_url_map():
    manifest = {
        "filenameToSignedUrls": {
            "results.jsonl": "https://signed.example.test/results",
        }
    }
    assert fw.parse_download_manifest(manifest) == {
        "results.jsonl": "https://signed.example.test/results",
    }


def test_parse_download_manifest_returns_empty_dict_when_missing():
    assert fw.parse_download_manifest({}) == {}
