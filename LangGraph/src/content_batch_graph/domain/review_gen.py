"""
Build native_reader_batch review requests, as OpenAI-compatible batch records, for
every reflexion/oracion field in a devocionales-json corpus file.

Review-side counterpart to devotional_gen.py (which builds generation requests).
Reuses this project's own corpus-reading convention from domain/scan.py
(find_devotional_files, the data.{language}.{date}[i] shape) rather than
reimplementing corpus discovery here.

native_reader_batch (config/roles.yml) uses the same Finding JSON schema as the
live native_reader role (domain/roles.build_finding_schema), enforced via
response_format on every record — not the free-text "current text: / proposed
text: / explanation:" format this module used before 2026-08-15. That plain-text
format let a real test model echo back an entire paragraph as "current text"
instead of the specific problem span, and drift into ad-hoc Markdown formatting
(**current text:** instead of the instructed wording) — a schema-constrained
response makes the field boundaries explicit instead of relying on the model to
follow prose instructions consistently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from batch_common import BatchProviderConfig, chat_request_record

from content_batch_graph.domain.devotional_gen import language_name
from content_batch_graph.domain.roles import Role, build_finding_schema
from content_batch_graph.domain.scan import find_devotional_files

# Same fields domain/scan.py scans for corpus-wide pattern matches — kept as this
# module's own constant (not imported from scan.py, which keeps it private) so each
# module stays free to diverge later without coordinating a shared constant.
REVIEWED_FIELDS = ("reflexion", "oracion")

# Devocional_year_{year}_{lang}_{version}.json, e.g. Devocional_year_2025_es_NVI.json.
_VERSIONED_FILENAME = re.compile(r"^Devocional_year_\d{4}_[^_]+_(?P<version>.+)\.json$")
# es/RVR1960's own legacy exception: predates the _{lang}_{version} naming
# convention, so it carries no tag at all (Devocional_year_2025.json) — confirmed
# against devocionales-json/index.json, which maps this exact filename to
# es/RVR1960. domain/scan.py::find_devotional_files has the matching glob
# exception; keep both in sync if this convention ever changes.
_LEGACY_RVR1960_VERSION = "RVR1960"


def version_from_filename(file_path: str) -> str:
    """
    The Bible-version tag encoded in a devocionales-json filename, e.g.
    "NVI" from "Devocional_year_2025_es_NVI.json".

    Falls back to _LEGACY_RVR1960_VERSION for the untagged legacy filename
    (Devocional_year_{year}.json) — the one file this naming convention has no
    tag for. Raises ValueError for anything else unrecognized, rather than
    silently mislabeling a review batch with the wrong version.
    """
    name = Path(file_path).name
    if re.fullmatch(r"Devocional_year_\d{4}\.json", name):
        return _LEGACY_RVR1960_VERSION
    match = _VERSIONED_FILENAME.match(name)
    if not match:
        raise ValueError(f"Can't determine Bible version from filename: {name!r}")
    return match.group("version")


def build_review_system(lang: str, role: Role) -> str:
    """
    The one system prompt shared by every review record for this language.

    Contains only the role's persona with {language} substituted. The output
    shape itself is enforced separately, via response_format (see
    build_review_response_format) — the persona describes the JSON field
    contract in prose for the model's benefit, but response_format is what
    actually constrains the wire format.
    """
    name = language_name(lang)
    return role["persona"].replace("{language}", name).strip()


def build_review_user(source_text: str) -> str:
    """The per-entry user prompt: the exact text to review, nothing else."""
    return f"Text to review:\n{source_text}"


def build_review_response_format(role: Role) -> dict:
    """
    The response_format value for every record in a review batch: JSON-schema
    enforcement of role's Finding shape (domain/roles.build_finding_schema) —
    the same schema the live native_reader role gets via
    .with_structured_output() in domain/flag.py, just expressed as a raw JSON
    schema dict instead of a Pydantic model, since Fireworks' batch API takes
    response_format.json_schema as a plain schema object, not a Python type.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "flag_response",
            "schema": build_finding_schema(role).model_json_schema(),
        },
    }


def custom_id_for(lang: str, version: str, entry_id: str, field: str) -> str:
    """
    Deterministic per-record id: review_{lang}_{version}_{entry_id}_{field}.

    Distinct prefix from devotional_gen.py's "gen_..." ids, so a results collector
    can tell a review-batch result apart from a generation-batch result at a glance.
    """
    return f"review_{lang}_{version}_{entry_id}_{field}"


class ReviewUnit:
    """One (entry_id, field, text) unit pulled from a corpus file, pending review."""

    __slots__ = ("entry_id", "field", "text", "version")

    def __init__(self, entry_id: str, field: str, text: str, version: str) -> None:
        self.entry_id = entry_id
        self.field = field
        self.text = text
        # Which Bible version this text came from (e.g. "RVR1960", "NVI") —
        # per-unit, not per-corpus-call: find_devotional_files(corpus_dir, "es")
        # returns files for multiple versions at once (RVR1960 + NVI both match
        # lang "es"), so a single caller-supplied version string can't correctly
        # label every unit it returns.
        self.version = version


def review_units_from_file(file_path: str, lang: str) -> list[ReviewUnit]:
    """
    Reads one devocionales-json file and returns a ReviewUnit for every non-empty
    reflexion/oracion field under data.{lang}, in file order.

    Mirrors domain/scan.py::scan_file_for_pattern's corpus-walking shape (same
    data.{language}.{date}[i] structure, same entry.get("id")) but collects text to
    review instead of scanning for a known pattern.
    """
    version = version_from_filename(file_path)
    with open(file_path, encoding="utf-8") as f:
        document = json.load(f)

    units: list[ReviewUnit] = []
    lang_data = document.get("data", {}).get(lang, {})
    for entries in lang_data.values():
        for entry in entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            for field in REVIEWED_FIELDS:
                text = entry.get(field, "")
                if text:
                    units.append(ReviewUnit(entry_id, field, text, version))
    return units


def review_units_from_corpus(corpus_dir: str, lang: str) -> list[ReviewUnit]:
    """Every ReviewUnit across every devocionales-json file for lang in corpus_dir."""
    units: list[ReviewUnit] = []
    for file_path in find_devotional_files(corpus_dir, lang):
        units.extend(review_units_from_file(file_path, lang))
    return units


def build_review_batch(
    units: list[ReviewUnit],
    lang: str,
    provider: BatchProviderConfig,
    role: Role,
    max_tokens: int = 4098,
    temperature: float = 0.2,
) -> list[dict]:
    """
    One dataset input line per ReviewUnit, system message omitted.

    Every unit shares one system prompt (build_review_system, identical across
    the whole batch), so it's never repeated per line here — pass it as
    submit()'s system_prompt instead, Fireworks' documented shape for this exact
    case (docs.fireworks.ai/guides/batch-inference's "Job-level system prompt"),
    which keeps the upload small and prompt-caching intact.

    Lower default temperature than devotional_gen.py's generation batch (0.7): a
    typo/grammar review pass benefits from consistency, not creative variation.

    No caller-supplied version parameter: each unit carries its own version
    (unit.version, set from its source filename by review_units_from_file) —
    review_units_from_corpus(corpus_dir, lang) can return units from more than
    one Bible version for the same language (e.g. es/RVR1960 + es/NVI both match
    lang "es"), so a single version string here would mislabel every custom_id
    from every version but one.

    response_format (build_review_response_format(role)) is set on every
    record — confirmed against a real batch test 2026-08-15: 10/10 rows
    returned valid JSON matching the schema, 0 truncated, and quoted_text
    values were short specific fragments rather than the whole-paragraph
    echoes the old free-text format produced.
    """
    extra: dict[str, Any] = dict(provider.extra_record_fields or {})
    response_format = build_review_response_format(role)

    return [
        chat_request_record(
            custom_id_for(lang, unit.version, unit.entry_id, unit.field),
            [{"role": "user", "content": build_review_user(unit.text)}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            **extra,
        )
        for unit in units
    ]
