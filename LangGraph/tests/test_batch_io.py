import re
from pathlib import Path

from content_batch_graph.domain import batch_io


def test_batch_input_path_naming_and_location():
    p = batch_io.batch_input_path(
        "es", "RVR1960", 2026, "deepseek-v3p2", "20260814_120000"
    )
    assert p.parent == batch_io.PATHS.batch_input_dir
    assert (
        p.name == "batch_input_es_RVR1960_2026_gen_deepseek-v3p2_20260814_120000.jsonl"
    )


def test_batch_output_path_mirrors_input_stem_under_batch_output():
    inp = batch_io.batch_input_path("es", "RVR1960", 2026, "m1", "20260814_120000")
    out = batch_io.batch_output_path(inp)
    assert out.parent == batch_io.PATHS.batch_output_dir
    assert out.name == f"{inp.stem}_results.jsonl"


def test_batch_output_path_accepts_a_foreign_path():
    out = batch_io.batch_output_path(Path("/somewhere/else/thing.jsonl"))
    assert out.name == "thing_results.jsonl"
    assert out.parent == batch_io.PATHS.batch_output_dir


def test_model_slug_takes_last_segment_of_a_fireworks_model_id():
    assert (
        batch_io.model_slug("accounts/fireworks/models/deepseek-v3p2")
        == "deepseek-v3p2"
    )
    assert batch_io.model_slug("gpt-4o-mini") == "gpt-4o-mini"


def test_utc_timestamp_format():
    assert re.fullmatch(r"\d{8}_\d{6}", batch_io.utc_timestamp())


def test_collection_path_naming():
    p = batch_io.collection_path("tl", "ASND", 2026)
    assert p.name == "Devocional_2026_tl_ASND_gen.json"
    assert p.parent == batch_io.GENOMES_DIR


def test_ensure_dirs_creates_all_three_and_is_idempotent():
    batch_io.ensure_dirs()
    batch_io.ensure_dirs()
    assert batch_io.PATHS.batch_input_dir.is_dir()
    assert batch_io.PATHS.batch_output_dir.is_dir()
    assert batch_io.GENOMES_DIR.is_dir()


def test_resolve_batch_input_falls_back_to_the_batch_input_dir():
    batch_io.ensure_dirs()
    target = batch_io.PATHS.batch_input_dir / "_resolve_probe.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    try:
        assert batch_io.resolve_batch_input("_resolve_probe.jsonl") == target
    finally:
        target.unlink()


def test_resolve_batch_output_returns_original_when_missing():
    assert batch_io.resolve_batch_output("nope.jsonl").name == "nope.jsonl"
