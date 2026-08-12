"""
A live KWF (Komisyon sa Wikang Filipino) dictionary lookup, used to ground
critic_pass's typo-category judgments in a real authoritative source instead of the
model's unaided memory.

Scope: this is a headword-existence check ("is this a real Filipino word?"), which is
exactly the question a typo claim reduces to. It has nothing useful to say about
grammar/awkward_phrasing findings — those are about whether correctly-spelled words
read naturally in sequence, not whether any single word exists. Never call this for
non-typo categories.

Built after a real bug: critic_pass flagged "espirituwal" as a typo of "espiritwal",
which we initially (wrongly) reversed based on secondary-source claims about KWF
orthography. Querying kwfdiksiyonaryo.ph directly confirmed the original judgment was
right: "espiritwal" has a real entry, "espirituwal" has none.
"""

from __future__ import annotations

import re
from typing import TypedDict

import httpx

_KWF_BASE_URL = "https://kwfdiksiyonaryo.ph/"
_NOT_FOUND_MARKER = "Walang resulta"
_TIMEOUT_S = 10.0


class DictionaryLookup(TypedDict):
    word: str
    found: bool
    part_of_speech: str | None
    definition: str | None


def lookup_word(word: str) -> DictionaryLookup:
    """
    Queries kwfdiksiyonaryo.ph for word. Returns found=False (not an error) if the
    site has no entry for it — that's a real, meaningful answer, not a failure.
    Raises httpx.HTTPError if the request itself fails (network/5xx).
    """
    response = httpx.get(_KWF_BASE_URL, params={"query": word}, timeout=_TIMEOUT_S)
    response.raise_for_status()
    html = response.text

    if _NOT_FOUND_MARKER in html:
        return DictionaryLookup(
            word=word, found=False, part_of_speech=None, definition=None
        )

    pos_match = re.search(
        r"Bahagi ng Pananalita.*?col-md-8[^>]*>([^<]+)<", html, re.DOTALL
    )
    definition_match = re.search(
        r"Kahulugan.*?col-md-8[^>]*>\s*<p[^>]*>([^<]+)", html, re.DOTALL
    )

    return DictionaryLookup(
        word=word,
        found=True,
        part_of_speech=pos_match.group(1).strip() if pos_match else None,
        definition=definition_match.group(1).strip() if definition_match else None,
    )
