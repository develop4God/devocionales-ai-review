"""BatchPaths directory resolution."""

from __future__ import annotations

from batch_common.paths import BatchPaths


def test_dirs_derive_from_root(tmp_path):
    p = BatchPaths(tmp_path, "TESTPROJ")
    assert p.batch_input_dir == tmp_path / "data" / "batch_input"
    assert p.batch_output_dir == tmp_path / "data" / "batch_output"


def test_env_var_overrides_data_dir_name(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTPROJ_DATA_DIR", "alt_data")
    p = BatchPaths(tmp_path, "TESTPROJ")
    assert p.batch_input_dir == tmp_path / "alt_data" / "batch_input"


def test_ensure_dirs_is_idempotent(tmp_path):
    p = BatchPaths(tmp_path, "TESTPROJ")
    p.ensure_dirs()
    p.ensure_dirs()
    assert p.batch_input_dir.is_dir()
    assert p.batch_output_dir.is_dir()


def test_resolve_batch_input_returns_existing_path_untouched(tmp_path):
    p = BatchPaths(tmp_path, "TESTPROJ")
    here = tmp_path / "here.jsonl"
    here.write_text("{}\n", encoding="utf-8")
    assert p.resolve_batch_input(here) == here


def test_resolve_batch_input_falls_back_to_batch_input_dir(tmp_path):
    p = BatchPaths(tmp_path, "TESTPROJ")
    p.ensure_dirs()
    target = p.batch_input_dir / "bare.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    assert p.resolve_batch_input("bare.jsonl") == target


def test_resolve_batch_output_falls_back_to_batch_output_dir(tmp_path):
    p = BatchPaths(tmp_path, "TESTPROJ")
    p.ensure_dirs()
    target = p.batch_output_dir / "res.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    assert p.resolve_batch_output("res.jsonl") == target


def test_resolve_returns_original_path_when_nothing_found(tmp_path):
    p = BatchPaths(tmp_path, "TESTPROJ")
    # Caller gets back what they asked for, so the FileNotFoundError names it.
    assert p.resolve_batch_input("missing.jsonl").name == "missing.jsonl"
