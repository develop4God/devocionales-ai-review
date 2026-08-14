"""
LangGraph's batch file path conventions.

Thin wrappers over batch_common.paths — this module owns *where* this project
puts batch files and what it names them; batch_common owns the generic
root/env-prefix mechanics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from batch_common import BatchPaths

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

PATHS = BatchPaths(_PROJECT_ROOT, env_prefix="LANGGRAPH")

# Where a collected year lands. data/genomes/ is already the gitignored home for
# generated corpora in this project, so a collection goes there rather than
# introducing a fourth data subdirectory.
GENOMES_DIR = PATHS.data_dir / "genomes"


def ensure_dirs() -> None:
    """Create batch_input/, batch_output/ and genomes/ if missing (idempotent)."""
    PATHS.ensure_dirs()
    GENOMES_DIR.mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    """Filename timestamp, UTC, YYYYmmdd_HHMMSS."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def model_slug(model: str) -> str:
    """
    Filename-safe slug for a model name.

    Fireworks model ids look like 'accounts/fireworks/models/deepseek-v3p2' — only
    the last segment is meaningful in a filename, and slashes can't appear in one.
    """
    return model.rstrip("/").split("/")[-1].replace(" ", "-")


def batch_input_path(
    lang: str, version: str, year: int, model_slug: str, ts: str
) -> Path:
    """data/batch_input/batch_input_{lang}_{version}_{year}_gen_{model_slug}_{ts}.jsonl"""
    PATHS.batch_input_dir.mkdir(parents=True, exist_ok=True)
    return (
        PATHS.batch_input_dir
        / f"batch_input_{lang}_{version}_{year}_gen_{model_slug}_{ts}.jsonl"
    )


def batch_output_path(input_path: Path) -> Path:
    """The results file matching an input file: same stem, under data/batch_output/."""
    PATHS.batch_output_dir.mkdir(parents=True, exist_ok=True)
    return PATHS.batch_output_dir / f"{Path(input_path).stem}_results.jsonl"


def collection_path(lang: str, version: str, year: int) -> Path:
    """data/genomes/Devocional_{year}_{lang}_{version}_gen.json"""
    GENOMES_DIR.mkdir(parents=True, exist_ok=True)
    return GENOMES_DIR / f"Devocional_{year}_{lang}_{version}_gen.json"


def resolve_batch_input(file_arg: str | Path) -> Path:
    """Resolve a --input CLI argument, falling back to data/batch_input/."""
    return PATHS.resolve_batch_input(file_arg)


def resolve_batch_output(file_arg: str | Path) -> Path:
    """Resolve a --results CLI argument, falling back to data/batch_output/."""
    return PATHS.resolve_batch_output(file_arg)
