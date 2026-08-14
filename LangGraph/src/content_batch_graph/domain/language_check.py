"""
A local LanguageTool server, used to ground critic_pass's typo/grammar judgments
in a real rule-and-dictionary source instead of the model's unaided memory —
the same role domain/dictionary.py's KWF lookup plays for Filipino, generalized to
languages where LanguageTool has a module that actually matches this project's
content.

Scope: LanguageTool checks a real dictionary + grammar rules, deterministically —
no AI, no per-call cost. It has nothing useful to say about awkward_phrasing
(that's a naturalness judgment, not a rule violation) — never call this for that
category.

Deliberately excluded from _SUPPORTED_LANGS, not just "unsupported by LanguageTool":
  - Hindi — real content exists (see GEP/data/batch_input/batch_input_hi_*), but no
    LanguageTool module exists for it at all.
  - Korean — devotional_gen.py's language_name() table has a "ko" entry, but no
    Korean content exists anywhere in this repo or its sibling GEP project as of
    this writing; listed here defensively in case that changes, not because it's
    an active gap today.
  - Filipino — LanguageTool has no dedicated Filipino ("fil") module, only Tagalog
    ("tl"). Tagalog and Filipino are related but distinct (Filipino is the
    standardized national language, with its own vocabulary/spelling conventions
    Tagalog proper doesn't have) — this project already has a purpose-built,
    authoritative Filipino source (domain/dictionary.py's KWF lookup) and must not
    silently substitute a Tagalog checker for it.
  - Chinese — LanguageTool's only Chinese module is Simplified ("zh-CN"), with no
    Traditional Chinese variant, and even for Simplified it does grammar only, no
    spellcheck. This project's Chinese devotional content is Traditional script —
    running it through a Simplified-only checker would flag nearly everything as
    wrong, not just miss some things.

Keyed by the same human-readable language name run_critic_pass already receives
(devotional_gen.py's language_name() output, e.g. "Spanish") — not an ISO code —
since that's what's actually in scope at every call site that would use this.

Requires a running LanguageTool server (self-hosted Docker, e.g.
`docker run -p 8081:8010 erikvl87/languagetool`), reachable at LANGUAGETOOL_URL
(default http://localhost:8081/v2/check). This project never talks to the public
LanguageTool API — devotional text may be unpublished, and the public API is
rate-limited.
"""

from __future__ import annotations

import os
from typing import TypedDict

import httpx

_DEFAULT_URL = "http://localhost:8081/v2/check"
_TIMEOUT_S = 10.0

# LanguageTool language codes for every language name this project's flag/critic
# passes actually pass around (the human-readable form, e.g. "Spanish", not an ISO
# code — see devotional_gen.py's language_name()) where LanguageTool has a module
# that genuinely matches this project's content. See the module docstring for why
# Filipino and Chinese are deliberately absent despite LanguageTool nominally
# supporting a related language/script.
_SUPPORTED_LANGS = {
    "arabic": "ar",
    "german": "de-DE",
    "english": "en-US",
    "spanish": "es",
    "french": "fr",
    "japanese": "ja-JP",
    "portuguese": "pt-BR",
    "brazilian portuguese": "pt-BR",
}


class LanguageToolMatch(TypedDict):
    quoted_text: str  # the exact span LanguageTool flagged, verbatim from source
    message: str  # LanguageTool's explanation, already in the tool's own language
    replacements: list[str]  # suggested corrections, best first; may be empty
    rule_category: str  # LanguageTool's own rule category, e.g. "TYPOS", "GRAMMAR"


def is_supported(language: str) -> bool:
    """True if LanguageTool has a module for this human-readable language name."""
    return language.lower() in _SUPPORTED_LANGS


def check_text(text: str, language: str) -> list[LanguageToolMatch]:
    """
    Runs text through the local LanguageTool server and returns every match found
    (empty list if the server ran and found nothing — a real, meaningful answer).

    Returns [] without contacting the server if language has no LanguageTool module
    (see _SUPPORTED_LANGS) or text is empty. Raises httpx.HTTPError if the server
    itself is unreachable or errors — same discipline as dictionary.py's KWF
    lookup: a failure to check is not the same answer as "checked, found nothing,"
    so callers must not conflate the two the way a swallowed exception would force
    them to.
    """
    lt_lang = _SUPPORTED_LANGS.get(language.lower())
    if not lt_lang or not text.strip():
        return []

    url = os.environ.get("LANGUAGETOOL_URL", _DEFAULT_URL)
    response = httpx.post(
        url,
        data={"text": text, "language": lt_lang},
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()

    matches: list[LanguageToolMatch] = []
    for match in payload.get("matches", []):
        offset = match.get("offset", 0)
        length = match.get("length", 0)
        quoted = text[offset : offset + length]
        if not quoted:
            continue
        matches.append(
            LanguageToolMatch(
                quoted_text=quoted,
                message=match.get("message", ""),
                replacements=[
                    r["value"] for r in match.get("replacements", []) if r.get("value")
                ],
                rule_category=(match.get("rule") or {})
                .get("category", {})
                .get("id", ""),
            )
        )
    return matches


def find_match_for(
    quoted_text: str, text: str, language: str
) -> LanguageToolMatch | None:
    """
    Runs check_text over text and returns the match whose quoted_text equals
    quoted_text, if any — the specific corroboration critic_pass needs: "did
    LanguageTool independently flag this exact span too?"
    """
    for match in check_text(text, language):
        if match["quoted_text"] == quoted_text:
            return match
    return None
