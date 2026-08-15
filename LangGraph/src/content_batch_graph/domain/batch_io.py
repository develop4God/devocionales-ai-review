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

# Where a collected review batch lands — a report of issues found in existing
# content, not generated content itself, so it gets its own directory rather
# than living alongside genomes/.
REVIEWS_DIR = PATHS.data_dir / "reviews"


def ensure_dirs() -> None:
    """Create batch_input/, batch_output/, genomes/ and reviews/ if missing (idempotent)."""
    PATHS.ensure_dirs()
    GENOMES_DIR.mkdir(parents=True, exist_ok=True)
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


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
    lang: str, version: str, year: int, provider_id: str, model_slug: str, ts: str
) -> Path:
    """
    data/batch_input/input_{lang}_{version}_{year}_{provider_id}_{model_slug}_{ts}.jsonl

    provider_id (the providers.yml entry that produced this file, e.g.
    "fireworks_batch_devotional_gen") is the real provenance marker — model_slug
    alone doesn't distinguish two providers that happen to serve the same model
    name, and a bare "gen" literal said nothing about where the file came from.
    """
    PATHS.batch_input_dir.mkdir(parents=True, exist_ok=True)
    return (
        PATHS.batch_input_dir
        / f"input_{lang}_{version}_{year}_{provider_id}_{model_slug}_{ts}.jsonl"
    )


def review_batch_input_path(
    lang: str, provider_id: str, model_slug: str, ts: str
) -> Path:
    """
    data/batch_input/review_{lang}_{provider_id}_{model_slug}_{ts}.jsonl

    No year/version in the filename, unlike batch_input_path — a review batch
    reads review_gen.review_units_from_corpus(corpus_dir, lang), which can span
    every version file for that language in one run (e.g. es/RVR1960 + es/NVI
    together), so a single year or version wouldn't describe the file's actual
    contents. Each unit's own version still lands in its custom_id via
    review_gen.custom_id_for — this filename just isn't scoped to one.
    """
    PATHS.batch_input_dir.mkdir(parents=True, exist_ok=True)
    return (
        PATHS.batch_input_dir / f"review_{lang}_{provider_id}_{model_slug}_{ts}.jsonl"
    )


def batch_output_path(input_path: Path) -> Path:
    """
    The results file matching an input file, under data/batch_output/: same stem
    with "input_" swapped for "output_" (falls back to prefixing "output_" if the
    input file didn't have an "input_" prefix, e.g. a foreign/custom path).
    """
    PATHS.batch_output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    if stem.startswith("input_"):
        out_name = "output_" + stem.removeprefix("input_")
    else:
        out_name = "output_" + stem
    return PATHS.batch_output_dir / f"{out_name}.jsonl"


# devocionales-json has two legacy files with no _{lang}_{version} suffix at all
# (Devocional_year_2025.json, Devocional_year_2026.json — both es/RVR1960; see
# devocionales_scripts/validate_devocional_index.py's ("es", "RVR1960", year)
# mapping). That bare, suffix-less shape is a legacy exception, not the naming
# rule to follow going forward — collection_path always writes the full
# _{lang}_{version} suffix, so every file this pipeline produces has a consistent,
# unambiguous name.
_DEFAULT_VERSION = "RVR1960"


def collection_path(lang: str, version: str | None, year: int, ts: str) -> Path:
    """
    data/genomes/Devocional_year_{year}_{lang}_{version}_gen_o_{ts}.json

    version defaults to "RVR1960" if not given — always included in the filename,
    never omitted, so every file this pipeline writes has the same consistent
    shape (see _DEFAULT_VERSION above for why a bare, suffix-less name is not
    used here even though two legacy devocionales-json files have one).

    Timestamped like every other file this pipeline writes: a retry or a re-run
    after a partial failure must never silently overwrite a prior collection —
    there'd be no way to tell "attempt 3" from "the only attempt," and no way to
    compare a failed run's partial output against a successful retry afterward.
    """
    GENOMES_DIR.mkdir(parents=True, exist_ok=True)
    version = version or _DEFAULT_VERSION
    return GENOMES_DIR / f"Devocional_year_{year}_{lang}_{version}_gen_o_{ts}.json"


def review_collection_path(lang: str, ts: str) -> Path:
    """
    data/reviews/review_{lang}_{ts}.json

    No version/year in the filename, unlike collection_path — a review batch
    (review_gen.review_units_from_corpus) can span every version file for one
    language in one run, so the collected result isn't scoped to a single
    version or year either. Each finding's own (entry_id, field, version)
    still lands inside the file's data, via review_collect.ReviewResult.
    """
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    return REVIEWS_DIR / f"review_{lang}_{ts}.json"


def resolve_batch_input(file_arg: str | Path) -> Path:
    """Resolve a --input CLI argument, falling back to data/batch_input/."""
    return PATHS.resolve_batch_input(file_arg)


def resolve_batch_output(file_arg: str | Path) -> Path:
    """Resolve a --results CLI argument, falling back to data/batch_output/."""
    return PATHS.resolve_batch_output(file_arg)
