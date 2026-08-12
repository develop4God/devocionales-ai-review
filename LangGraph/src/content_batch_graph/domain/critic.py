"""
The critic pass: a second, independent AI judges each verified finding for
correctness and proposes an exact surgical replacement.

verify_pass only proves a finding's quoted_text is present, verbatim, in the source
— it says nothing about whether the finding is actually right. This is the layer
that closes that gap: one real model call per finding (not one call for all
findings at once), so each judgment is independent and isn't biased by the others.
"""

from __future__ import annotations

import httpx
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from content_batch_graph.domain.dictionary import lookup_word
from content_batch_graph.domain.providers import get_model
from content_batch_graph.state import CriticFinding, VerifiedFinding

# Unicode hyphen lookalikes observed being substituted by the model for a plain
# ASCII "-" when proposing a replacement (e.g. U+2011 NON-BREAKING HYPHEN in a
# Portuguese clitic pronoun fix, "amar-te" -> "amar‑te") — visually identical in a
# diff, but a real character-level corruption if applied to the corpus. Scoped to
# hyphen variants only, not quote-style normalization: curly quotes can be the
# correct orthographic form in some languages, so those are left to the critic's
# own judgment rather than force-normalized here.
_HYPHEN_LOOKALIKES = {
    "‐": "-",  # HYPHEN
    "‑": "-",  # NON-BREAKING HYPHEN
    "‒": "-",  # FIGURE DASH
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
}


def _normalize_replacement_text(text: str) -> str:
    for lookalike, ascii_equivalent in _HYPHEN_LOOKALIKES.items():
        text = text.replace(lookalike, ascii_equivalent)
    return text


_CRITIC_PERSONA = """\
You are an independent {language} language critic reviewing one specific claim \
about a piece of {language} text. You did not make the original claim — judge it \
fresh, on its own merits.

You will be given the full source text, one quoted span from it, and a claimed \
issue with that span. Decide:
1. Is this actually a real issue in correct {language}? If the quoted text is \
   already correct, or the claimed issue is wrong, say so.
2. If it is a real issue, give the exact replacement text for the quoted span —
   the minimal correction, changing nothing beyond what's needed to fix this \
   specific issue.

Judge only the quoted span and the claimed issue. Do not propose unrelated \
changes."""


def _check_dictionary_for_typo(quoted_text: str) -> bool | None:
    """
    Checks quoted_text against the real KWF dictionary as the source of truth for
    a typo claim. Returns:
      - False if the word IS found (a real word can't be a typo — dismiss the
        claim outright, no further review needed)
      - None if the word is NOT found, or the check can't run (multi-word span,
        dictionary-site failure) — this is NOT evidence of a typo (Filipino
        dictionaries only list root words, not every inflected/affixed form, so a
        miss is inconclusive) — falls through to the critic's own judgment
    """
    word = quoted_text.strip()
    if not word or " " in word:
        return None
    try:
        lookup = lookup_word(word)
    except httpx.HTTPError:
        # A dictionary-site failure (network, 5xx) should never block the pipeline
        # — fall back to the model's unaided judgment rather than erroring out.
        return None
    return False if lookup["found"] else None


class _CriticResponse(BaseModel):
    is_valid: bool = Field(
        description="True if the claimed issue is a real problem in correct "
        "{language}, false if the quoted text is already correct or the claim "
        "is wrong."
    )
    replacement_text: str = Field(
        default="",
        description="The exact corrected replacement for quoted_text. Empty if "
        "is_valid is false.",
    )
    reasoning: str = Field(
        description="Brief explanation of the judgment, for a human reviewer."
    )


def run_critic_pass(
    source_text: str,
    finding: VerifiedFinding,
    language: str,
    provider_id: str | None = None,
) -> CriticFinding:
    """
    Calls a real model, independent of the original flag_pass call, to judge one
    verified finding and propose its exact surgical replacement text.

    For Filipino typo findings, the real KWF dictionary is checked first as the
    source of truth: if the quoted word is a real dictionary entry, the typo claim
    is dismissed immediately, no model call needed — a real word can't be a typo.
    If the word isn't found, that's inconclusive (Filipino dictionaries only list
    root words, not every inflected form) and falls through to the critic's own
    judgment below, same as any other finding.
    """
    if language == "Filipino" and finding["category"] == "typo":
        dictionary_verdict = _check_dictionary_for_typo(finding["quoted_text"])
        if dictionary_verdict is False:
            return CriticFinding(
                quoted_text=finding["quoted_text"],
                issue=finding["issue"],
                category=finding["category"],
                verified=True,
                is_valid=False,
                replacement_text=None,
                critic_reasoning=(
                    f"{finding['quoted_text']!r} is a real word in the KWF "
                    "(Komisyon sa Wikang Filipino) dictionary — dismissed as a "
                    "typo claim without further review."
                ),
            )

    persona = _CRITIC_PERSONA.format(language=language)
    model = get_model(provider_id).with_structured_output(_CriticResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{persona}"),
            (
                "human",
                (
                    "Source text:\n{source_text}\n\n"
                    "Quoted span: {quoted_text!r}\n"
                    "Claimed issue: {issue}"
                ),
            ),
        ]
    )
    response = (prompt | model).invoke(
        {
            "persona": persona,
            "source_text": source_text,
            "quoted_text": finding["quoted_text"],
            "issue": finding["issue"],
        }
    )

    replacement_text = response.replacement_text or None
    if replacement_text:
        replacement_text = _normalize_replacement_text(replacement_text)

    return CriticFinding(
        quoted_text=finding["quoted_text"],
        issue=finding["issue"],
        category=finding["category"],
        verified=True,
        is_valid=response.is_valid,
        replacement_text=replacement_text,
        critic_reasoning=response.reasoning,
    )


def run_critic_pass_batch(
    source_text: str,
    findings: list[VerifiedFinding],
    language: str,
    provider_id: str | None = None,
) -> list[CriticFinding]:
    """Runs run_critic_pass independently for each finding, preserving order."""
    return [
        run_critic_pass(source_text, finding, language, provider_id)
        for finding in findings
    ]
