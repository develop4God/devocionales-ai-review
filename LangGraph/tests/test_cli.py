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
