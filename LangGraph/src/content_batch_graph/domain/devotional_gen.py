"""
Build a full year of devotional-generation requests as Fireworks batch dataset
input lines.

Net-new generation logic (the rest of domain/ is review-side). The output schema is
deliberately minimal — each day is only {date, reflexion, oracion}: no verse
selection, no verse-of-the-day spine, no para_meditar/tags.

The system prompt is built once and shared by every day in the year, but is never
repeated per dataset line — it's passed as the batch job's own system_prompt
(Fireworks' documented shape for a dataset where every row shares one system
message, docs.fireworks.ai/guides/batch-inference's "Job-level system prompt"),
which is what keeps the shared prefix byte-identical and prompt-caching intact
across all 365 requests — which is also why all per-day variation is pushed into
the (short) user message.
"""

from __future__ import annotations

import calendar
from datetime import date

from batch_common import BatchProviderConfig, chat_request_record

from content_batch_graph.domain.roles import Role

_JSON_CONTRACT = (
    "Respond with a single JSON object and nothing else. It must have exactly two "
    "keys, both strings:\n"
    '  "reflexion" — the day\'s reflection.\n'
    '  "oracion"   — the short closing prayer.\n'
    "Do not add any other key. Do not wrap the JSON in markdown fences."
)


# Role personas are written with a {language} placeholder that reads as a language
# name ("a native {language} speaker"), so a bare ISO code would produce "a native es
# speaker". Codes seen in this corpus are mapped; anything else passes through as-is.
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "tl": "Tagalog",
    "zh": "Chinese",
}


def language_name(lang: str) -> str:
    """Human-readable language name for an ISO-ish code, or the input unchanged."""
    return _LANGUAGE_NAMES.get(lang.lower(), lang)


def build_generation_system(lang: str, version: str, role: Role) -> str:
    """
    The one system prompt shared by every day in a year's batch.

    Contains only invariants — persona, language, output contract — so the prefix
    is byte-identical across all records and stays cacheable.
    """
    name = language_name(lang)
    persona = role["persona"].replace("{language}", name)
    return (
        f"{persona.strip()}\n\n"
        f"Language: {name} ({lang}). Edition context: {version}.\n"
        f"Write everything — reflection and prayer — in {name}.\n\n"
        f"{_JSON_CONTRACT}"
    )


def build_generation_user(day: date) -> str:
    """
    The per-day user prompt: the date and its calendar framing, nothing else.

    There is no verse spine to carry, so this stays short on purpose — it is the
    only part of each request that differs from the shared cached prefix.
    """
    weekday = calendar.day_name[day.weekday()]
    month = calendar.month_name[day.month]
    return (
        f"Date: {day.isoformat()}\n"
        f"Day: {weekday}, {day.day} {month} {day.year}\n\n"
        f"Write today's devotional: the reflection (reflexion) and the closing "
        f"prayer (oracion). Return the JSON object described in your instructions."
    )


def year_specs(year: int) -> list[date]:
    """
    Every calendar day of `year`, in order.

    Uses calendar.isleap rather than a fixed 365-day range so leap years produce
    366 days and Feb 29 is never silently dropped or shifted.
    """
    total = 366 if calendar.isleap(year) else 365
    return [date.fromordinal(date(year, 1, 1).toordinal() + i) for i in range(total)]


def custom_id_for(lang: str, version: str, day: date) -> str:
    """
    Deterministic per-record id: gen_{lang}_{version}_{year}_{MM-DD}.

    Encoding the date in the id lets the collector recover it directly, with no
    side index file mapping custom_id back to a date.
    """
    return f"gen_{lang}_{version}_{day.year}_{day:%m-%d}"


def build_year_batch(
    year: int,
    lang: str,
    version: str,
    provider: BatchProviderConfig,
    role: Role,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> list[dict]:
    """
    One dataset input line per day of `year`, system message omitted.

    Pass build_generation_system(lang, version, role) as the batch job's own
    system_prompt at submit time — see this module's docstring for why it's
    never repeated per line here.
    """
    extra = dict(provider.extra_record_fields or {})

    return [
        chat_request_record(
            custom_id_for(lang, version, day),
            [{"role": "user", "content": build_generation_user(day)}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            **extra,
        )
        for day in year_specs(year)
    ]
