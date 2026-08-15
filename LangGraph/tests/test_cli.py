"""
CLI tests. Every network-touching subcommand is exercised against a stubbed
BatchClient — no test here reaches the real Fireworks API.
"""

import json
from pathlib import Path
from typing import ClassVar

import pytest
from batch_common import BatchAPIError

from content_batch_graph import cli


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    # build-year must work with no key set at all; the network commands must fail
    # loudly rather than silently picking up an ambient key.
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


@pytest.fixture
def sample_jsonl(tmp_path):
    p = tmp_path / "batch_input_sample.jsonl"
    p.write_text('{"custom_id": "gen_es_RVR1960_2026_01-01"}\n', encoding="utf-8")
    return p


class _StubClient:
    """Records calls; stands in for batch_common.BatchClient."""

    instances: ClassVar[list] = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[tuple] = []
        _StubClient.instances.append(self)

    def upload(self, path):
        self.calls.append(("upload", path))
        return "file-123"

    def submit(self, file_id):
        self.calls.append(("submit", file_id))
        return "batch-456"

    def poll(self, batch_id, interval=30, timeout=86_400):
        self.calls.append(("poll", batch_id, interval, timeout))
        return "file-out-789"

    def download(self, file_id, dest):
        self.calls.append(("download", file_id, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}\n", encoding="utf-8")
        return dest


@pytest.fixture
def stub_client(monkeypatch):
    _StubClient.instances = []
    monkeypatch.setattr(cli, "BatchClient", _StubClient)
    return _StubClient


# ── parser ────────────────────────────────────────────────────────────────────


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_build_year_defaults_to_the_fireworks_batch_provider():
    args = cli.build_parser().parse_args(
        ["build-year", "--lang", "es", "--version", "RVR1960", "--year", "2026"]
    )
    assert args.provider == "fireworks_batch_devotional_gen"
    assert args.dry_run is False


# ── build-year ────────────────────────────────────────────────────────────────


def test_build_year_dry_run_writes_365_lines_with_no_api_key(capsys):
    rc = cli.main(
        [
            "build-year",
            "--lang",
            "es",
            "--version",
            "RVR1960",
            "--year",
            "2026",
            "--limit",
            "365",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Wrote 365 records" in out
    assert "Dry run" in out
    assert "est. prompt tokens" in out

    path = out.splitlines()[0].split("-> ", 1)[1]
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 365
    first = json.loads(lines[0])
    assert first["custom_id"] == "gen_es_RVR1960_2026_01-01"
    assert first["body"]["response_format"] == {"type": "json_object"}


def test_build_year_dry_run_single_line_with_native_reader_role(capsys):
    # A dry-run "does the prompt look right" check before anything reaches
    # Fireworks: one record, native_reader role, no API key, no network call.
    rc = cli.main(
        [
            "build-year",
            "--lang",
            "es",
            "--version",
            "RVR1960",
            "--year",
            "2026",
            "--role",
            "native_reader",
            "--limit",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Wrote 1 records" in out
    assert "Dry run" in out

    path = out.splitlines()[0].split("-> ", 1)[1]
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    # The system prompt (native_reader's persona) is never embedded per record —
    # it's passed as the batch job's own system_prompt at submit time (Fireworks'
    # documented shape for a shared system message), so the dry-run JSONL alone
    # only shows the per-day user message. Reviewing the resolved system prompt
    # itself is domain.devotional_gen.build_generation_system()'s job, not this
    # file's — see test_devotional_gen.py's own coverage of that function.
    assert all(m["role"] != "system" for m in record["body"]["messages"])
    assert record["body"]["messages"][0]["role"] == "user"


def test_build_year_limit_truncates(capsys):
    assert (
        cli.main(
            [
                "build-year",
                "--lang",
                "es",
                "--version",
                "RVR1960",
                "--year",
                "2026",
                "--limit",
                "3",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "Wrote 3 records" in capsys.readouterr().out


def test_build_year_leap_year_produces_366(capsys):
    assert (
        cli.main(
            [
                "build-year",
                "--lang",
                "es",
                "--version",
                "RVR1960",
                "--year",
                "2028",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "Wrote 366 records" in capsys.readouterr().out


def test_build_year_rejects_a_non_batch_provider(capsys):
    rc = cli.main(
        [
            "build-year",
            "--lang",
            "es",
            "--version",
            "RVR1960",
            "--year",
            "2026",
            "--provider",
            "anthropic_default",
            "--dry-run",
        ]
    )
    assert rc == 1
    assert "batch.supported" in capsys.readouterr().err


# ── submit / poll / download ──────────────────────────────────────────────────


def test_submit_uploads_then_submits(stub_client, sample_jsonl, capsys):
    assert cli.main(["submit", "--input", str(sample_jsonl)]) == 0
    calls = stub_client.instances[0].calls
    assert calls[0][0] == "upload" and calls[1] == ("submit", "file-123")
    assert "batch_id=batch-456" in capsys.readouterr().out


def test_submit_reports_a_missing_input_file(stub_client, capsys):
    assert cli.main(["submit", "--input", "definitely_missing.jsonl"]) == 1
    assert "not found" in capsys.readouterr().err


def test_submit_without_api_key_fails_loudly(capsys):
    # Real BatchClient this time — construction must refuse without the key.
    rc = cli.main(["submit", "--input", "whatever.jsonl"])
    assert rc == 1
    assert "FIREWORKS_API_KEY" in capsys.readouterr().err


def test_poll_returns_the_output_file_id(stub_client, capsys):
    assert cli.main(["poll", "--batch-id", "batch-456", "--interval", "1"]) == 0
    assert ("poll", "batch-456", 1, 86_400) == stub_client.instances[0].calls[0]
    assert "output_file_id=file-out-789" in capsys.readouterr().out


def test_poll_surfaces_a_batch_api_error(monkeypatch, stub_client, capsys):
    def boom(self, batch_id, interval=30, timeout=86_400):
        raise BatchAPIError("Batch ended with status='failed'")

    monkeypatch.setattr(_StubClient, "poll", boom)
    assert cli.main(["poll", "--batch-id", "batch-456"]) == 1
    assert "status='failed'" in capsys.readouterr().err


def test_download_writes_to_dest(stub_client, tmp_path, capsys):
    dest = tmp_path / "out" / "results.jsonl"
    assert (
        cli.main(["download", "--output-file-id", "file-out-789", "--dest", str(dest)])
        == 0
    )
    assert dest.exists()
    assert "Downloaded file-out-789" in capsys.readouterr().out


# ── collect ───────────────────────────────────────────────────────────────────


def test_collect_writes_a_year_collection(tmp_path, capsys):
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "custom_id": "gen_es_RVR1960_2026_01-01",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"reflexion": "R", "oracion": "O"}
                                    )
                                }
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "collected.json"
    rc = cli.main(
        [
            "collect",
            "--results",
            str(results),
            "--lang",
            "es",
            "--version",
            "RVR1960",
            "--year",
            "2026",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert "Collected 1 days (0 errors)" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["data"][0]["reflexion"] == "R"


def test_collect_reports_errors_without_failing(tmp_path, capsys):
    results = tmp_path / "results.jsonl"
    results.write_text('{"custom_id": "bogus"}\n', encoding="utf-8")
    out = tmp_path / "collected.json"
    assert (
        cli.main(
            [
                "collect",
                "--results",
                str(results),
                "--lang",
                "es",
                "--version",
                "RVR1960",
                "--year",
                "2026",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "Collected 0 days (1 errors)" in printed
    assert "unparseable custom_id" in printed


def test_collect_reports_a_missing_results_file(capsys):
    assert (
        cli.main(
            [
                "collect",
                "--results",
                "missing_results.jsonl",
                "--lang",
                "es",
                "--version",
                "RVR1960",
                "--year",
                "2026",
            ]
        )
        == 1
    )
    assert "not found" in capsys.readouterr().err


# ── pipeline ──────────────────────────────────────────────────────────────────


def _pipeline_argv(tmp_path, *extra):
    return [
        "pipeline",
        "--lang",
        "tl",
        "--version",
        "ASND",
        "--year",
        "2026",
        "--limit",
        "2",
        "--poll-interval",
        "1",
        "--out",
        str(tmp_path / "collected.json"),
        *extra,
    ]


_GOOD_RESULT_LINE = (
    json.dumps(
        {
            "custom_id": "gen_tl_ASND_2026_01-01",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"reflexion": "R", "oracion": "O"}
                                )
                            }
                        }
                    ]
                }
            },
        }
    )
    + "\n"
)


@pytest.fixture
def results_writing_client(stub_client, monkeypatch):
    """Stub whose download() writes a well-formed one-record results file."""

    def download(self, file_id, dest):
        self.calls.append(("download", file_id, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_GOOD_RESULT_LINE, encoding="utf-8")
        return dest

    monkeypatch.setattr(_StubClient, "download", download)
    return stub_client


def test_pipeline_dry_run_builds_only_and_makes_no_network_call(tmp_path, capsys):
    rc = cli.main(_pipeline_argv(tmp_path, "--dry-run"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "[1/5] Building batch for tl/ASND/2026" in out
    assert "2 records written to" in out
    assert "Dry run — the remaining steps WOULD run" in out
    for step in ("[2/5]", "[3/5]", "[4/5]", "[5/5]"):
        assert step in out
    assert "Nothing submitted" in out


def test_pipeline_happy_path_runs_all_five_steps(
    results_writing_client, tmp_path, capsys
):
    rc = cli.main(_pipeline_argv(tmp_path))
    assert rc == 0
    kinds = [c[0] for c in results_writing_client.instances[0].calls]
    assert kinds == ["upload", "submit", "poll", "download"]
    out = capsys.readouterr().out
    for step in ("[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]"):
        assert step in out
    assert "Pipeline complete." in out
    assert "records collected: 1" in out
    assert "errors:            0" in out
    collected = json.loads((tmp_path / "collected.json").read_text(encoding="utf-8"))
    assert collected["data"][0]["reflexion"] == "R"


def test_pipeline_uses_the_requested_poll_interval(results_writing_client, tmp_path):
    assert cli.main(_pipeline_argv(tmp_path)) == 0
    poll_call = next(
        c for c in results_writing_client.instances[0].calls if c[0] == "poll"
    )
    assert poll_call[2] == 1


def test_pipeline_stops_when_submit_fails(stub_client, monkeypatch, tmp_path, capsys):
    def boom(self, file_id):
        raise BatchAPIError("HTTP 401: bad key")

    monkeypatch.setattr(_StubClient, "submit", boom)
    assert cli.main(_pipeline_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "step 2/5 (submit) failed" in captured.err
    assert "HTTP 401" in captured.err
    assert "[3/5]" not in captured.out
    assert [c[0] for c in stub_client.instances[0].calls] == ["upload"]


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_pipeline_stops_on_a_terminal_poll_status(
    stub_client, monkeypatch, tmp_path, capsys, status
):
    def boom(self, batch_id, interval=30, timeout=86_400):
        raise BatchAPIError(f"Batch ended with status='{status}'")

    monkeypatch.setattr(_StubClient, "poll", boom)
    assert cli.main(_pipeline_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "step 3/5 (poll) failed" in captured.err
    assert f"status='{status}'" in captured.err
    assert "[4/5]" not in captured.out
    assert "download" not in [c[0] for c in stub_client.instances[0].calls]


def test_pipeline_stops_when_poll_times_out(stub_client, monkeypatch, tmp_path, capsys):
    def boom(self, batch_id, interval=30, timeout=86_400):
        raise TimeoutError("Batch polling timed out after 5s")

    monkeypatch.setattr(_StubClient, "poll", boom)
    assert cli.main(_pipeline_argv(tmp_path)) == 1
    assert "step 3/5 (poll) failed" in capsys.readouterr().err


def test_pipeline_stops_when_download_fails(stub_client, monkeypatch, tmp_path, capsys):
    def boom(self, file_id, dest):
        raise BatchAPIError("HTTP 500: file gone")

    monkeypatch.setattr(_StubClient, "download", boom)
    assert cli.main(_pipeline_argv(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "step 4/5 (download) failed" in captured.err
    assert "[5/5]" not in captured.out
    assert not (tmp_path / "collected.json").exists()


def test_pipeline_stops_on_malformed_results(
    stub_client, monkeypatch, tmp_path, capsys
):
    def download(self, file_id, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("this is not jsonl\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(_StubClient, "download", download)
    rc = cli.main(_pipeline_argv(tmp_path))
    captured = capsys.readouterr()
    assert rc == 1
    assert "step 5/5 (collect) failed" in captured.err


def test_pipeline_rejects_a_non_batch_provider(tmp_path, capsys):
    rc = cli.main(
        _pipeline_argv(tmp_path, "--provider", "anthropic_default", "--dry-run")
    )
    assert rc == 1


# ── review-build / review-submit ────────────────────────────────────────────────


def _write_review_corpus(tmp_path, lang="es", version="RVR1960"):
    doc = {
        "data": {
            lang: {
                "2026-01-01": [
                    {"id": "e1", "reflexion": "Un dia especial.", "oracion": "Amen."},
                    {"id": "e2", "reflexion": "", "oracion": "Otra oracion."},
                ],
            }
        }
    }
    path = tmp_path / f"Devocional_year_2026_{lang}_{version}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


class _ReviewStubClient:
    """
    Records calls; stands in for batch_common.BatchClient's real
    create_dataset/upload/submit sequence — distinct from _StubClient above,
    which stands in for the OLD upload(path)->submit(file_id) shape that
    review-submit deliberately does not use (see cmd_review_submit's
    docstring).
    """

    instances: ClassVar[list] = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[tuple] = []
        _ReviewStubClient.instances.append(self)

    def create_dataset(self, dataset_id, example_count):
        self.calls.append(("create_dataset", dataset_id, example_count))
        return dataset_id

    def upload(self, dataset_id, path):
        self.calls.append(("upload", dataset_id, path))
        return dataset_id

    def submit(self, input_dataset_id, output_dataset_id, job_id, **kwargs):
        self.calls.append(
            ("submit", input_dataset_id, output_dataset_id, job_id, kwargs)
        )
        return job_id


@pytest.fixture
def review_stub_client(monkeypatch):
    _ReviewStubClient.instances = []
    monkeypatch.setattr(cli, "BatchClient", _ReviewStubClient)
    return _ReviewStubClient


def test_review_build_dry_run_writes_all_reviewable_fields(tmp_path, capsys):
    _write_review_corpus(tmp_path)
    rc = cli.main(
        [
            "review-build",
            "--corpus-dir",
            str(tmp_path),
            "--lang",
            "es",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # e1 reflexion + e1 oracion + e2 oracion (e2 reflexion is empty, skipped)
    assert "Wrote 3 records" in out
    assert "versions covered: RVR1960" in out
    assert "Dry run" in out

    path = out.splitlines()[0].split("-> ", 1)[1]
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["custom_id"] == "review_es_RVR1960_e1_reflexion"
    assert first["body"]["response_format"]["type"] == "json_schema"
    assert all(m["role"] != "system" for m in first["body"]["messages"])


def test_review_build_labels_mixed_versions_correctly(tmp_path, capsys):
    _write_review_corpus(tmp_path, lang="es", version="RVR1960")
    _write_review_corpus(tmp_path, lang="es", version="NVI")
    rc = cli.main(
        ["review-build", "--corpus-dir", str(tmp_path), "--lang", "es", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Wrote 6 records" in out
    assert "versions covered: NVI, RVR1960" in out

    path = out.splitlines()[0].split("-> ", 1)[1]
    custom_ids = {
        json.loads(line)["custom_id"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    }
    assert "review_es_RVR1960_e1_reflexion" in custom_ids
    assert "review_es_NVI_e1_reflexion" in custom_ids


def test_review_build_limit_truncates(tmp_path, capsys):
    _write_review_corpus(tmp_path)
    rc = cli.main(
        [
            "review-build",
            "--corpus-dir",
            str(tmp_path),
            "--lang",
            "es",
            "--limit",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert "Wrote 1 records" in capsys.readouterr().out


def test_review_build_reports_no_reviewable_fields(tmp_path, capsys):
    rc = cli.main(
        ["review-build", "--corpus-dir", str(tmp_path), "--lang", "es", "--dry-run"]
    )
    assert rc == 1
    assert "No reviewable" in capsys.readouterr().err


def test_review_submit_creates_uploads_then_submits(
    review_stub_client, sample_jsonl, capsys
):
    rc = cli.main(
        ["review-submit", "--input", str(sample_jsonl), "--lang", "es"]
    )
    assert rc == 0
    calls = review_stub_client.instances[0].calls
    assert calls[0][0] == "create_dataset"
    assert calls[1][0] == "upload"
    assert calls[2][0] == "submit"
    # create_dataset and upload/submit all agree on the same generated dataset ids
    dataset_id = calls[0][1]
    assert calls[1][1] == dataset_id
    assert calls[2][1] == dataset_id
    out = capsys.readouterr().out
    assert "Submitted batch" in out


def test_review_submit_honors_an_explicit_job_id(
    review_stub_client, sample_jsonl, capsys
):
    rc = cli.main(
        [
            "review-submit",
            "--input",
            str(sample_jsonl),
            "--lang",
            "es",
            "--job-id",
            "my-review-job",
        ]
    )
    assert rc == 0
    calls = review_stub_client.instances[0].calls
    assert calls[0][1] == "my-review-job-in"
    assert calls[2][1:4] == ("my-review-job-in", "my-review-job-out", "my-review-job")


def test_review_submit_passes_system_prompt_and_inference_params(
    review_stub_client, sample_jsonl
):
    cli.main(
        [
            "review-submit",
            "--input",
            str(sample_jsonl),
            "--lang",
            "es",
            "--max-tokens",
            "512",
            "--temperature",
            "0.1",
        ]
    )
    calls = review_stub_client.instances[0].calls
    submit_kwargs = calls[2][4]
    assert "native Spanish speaker" in submit_kwargs["system_prompt"]
    assert submit_kwargs["max_tokens"] == 512
    assert submit_kwargs["temperature"] == 0.1


def test_review_submit_reports_a_missing_input_file(review_stub_client, capsys):
    rc = cli.main(
        ["review-submit", "--input", "definitely_missing.jsonl", "--lang", "es"]
    )
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_review_submit_rejects_an_empty_input_file(
    review_stub_client, tmp_path, capsys
):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    rc = cli.main(["review-submit", "--input", str(empty), "--lang", "es"])
    assert rc == 1
    assert "no records" in capsys.readouterr().err


def test_review_submit_without_api_key_fails_loudly(capsys):
    # Real BatchClient this time — construction must refuse without the key.
    rc = cli.main(["review-submit", "--input", "whatever.jsonl", "--lang", "es"])
    assert rc == 1
    assert "FIREWORKS_API_KEY" in capsys.readouterr().err
