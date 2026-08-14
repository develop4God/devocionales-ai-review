"""
Build native_reader_batch review requests, as OpenAI-compatible batch records, for
every reflexion/oracion field in a devocionales-json corpus file.

Review-side counterpart to devotional_gen.py (which builds generation requests).
Reuses this project's own corpus-reading convention from domain/scan.py
(find_devotional_files, the data.{language}.{date}[i] shape) rather than
reimplementing corpus discovery here.

native_reader_batch (config/roles.yml) is a plain-text response format
("current text: / proposed text: / explanation:"), not the structured-output JSON
schema domain/flag.py uses for the live native_reader role — so unlike
devotional_gen.py's records, no response_format is set here; the role's own persona
dictates the expected shape, and domain/review_collect.py is responsible for parsing
it back out.
"""

from __future__ import annotations

import json
from typing import Any

from batch_common import BatchProviderConfig, chat_request_record

from content_batch_graph.domain.devotional_gen import language_name
from content_batch_graph.domain.roles import Role
from content_batch_graph.domain.scan import find_devotional_files

# Same fields domain/scan.py scans for corpus-wide pattern matches — kept as this
# module's own constant (not imported from scan.py, which keeps it private) so each
# module stays free to diverge later without coordinating a shared constant.
REVIEWED_FIELDS = ("reflexion", "oracion")


def build_review_system(lang: str, role: Role) -> str:
    """
    The one system prompt shared by every review record for this language.

    Contains only the role's persona with {language} substituted — no JSON output
    contract, unlike devotional_gen.py's generation prompt: native_reader_batch's
    own persona text already specifies the plain-text response format.
    """
    name = language_name(lang)
    return role["persona"].replace("{language}", name).strip()


def build_review_user(source_text: str) -> str:
    """The per-entry user prompt: the exact text to review, nothing else."""
    return f"Text to review:\n{source_text}"


def custom_id_for(lang: str, version: str, entry_id: str, field: str) -> str:
    """
    Deterministic per-record id: review_{lang}_{version}_{entry_id}_{field}.

    Distinct prefix from devotional_gen.py's "gen_..." ids, so a results collector
    can tell a review-batch result apart from a generation-batch result at a glance.
    """
    return f"review_{lang}_{version}_{entry_id}_{field}"


class ReviewUnit:
    """One (entry_id, field, text) unit pulled from a corpus file, pending review."""

    __slots__ = ("entry_id", "field", "text")

    def __init__(self, entry_id: str, field: str, text: str) -> None:
        self.entry_id = entry_id
        self.field = field
        self.text = text


def review_units_from_file(file_path: str, lang: str) -> list[ReviewUnit]:
    """
    Reads one devocionales-json file and returns a ReviewUnit for every non-empty
    reflexion/oracion field under data.{lang}, in file order.

    Mirrors domain/scan.py::scan_file_for_pattern's corpus-walking shape (same
    data.{language}.{date}[i] structure, same entry.get("id")) but collects text to
    review instead of scanning for a known pattern.
    """
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
                    units.append(ReviewUnit(entry_id, field, text))
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
    version: str,
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
    """
    extra: dict[str, Any] = dict(provider.extra_record_fields or {})

    return [
        chat_request_record(
            custom_id_for(lang, version, unit.entry_id, unit.field),
            [{"role": "user", "content": build_review_user(unit.text)}],
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
        for unit in units
    ]
