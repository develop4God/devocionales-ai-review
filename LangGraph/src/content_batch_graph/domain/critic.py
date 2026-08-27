"""
The critic pass: a second, independent AI judges each verified finding for
correctness and proposes an exact surgical replacement.

verify_pass only proves a finding's quoted_text is present, verbatim, in the source
— it says nothing about whether the finding is actually right. This is the layer
that closes that gap: one real model call per finding (not one call for all
findings at once), so each judgment is independent and isn't biased by the others.
"""

from __future__ import annotations

import re

import httpx
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from content_batch_graph.domain.dictionary import lookup_word
from content_batch_graph.domain.language_check import find_match_for, is_supported
from content_batch_graph.domain.providers import get_model
from content_batch_graph.domain.structured_call import invoke_structured
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


# Post-verdict dismiss rules: applied to the critic's OWN judgment after it comes
# back, never shown to the model beforehand. Priming the model with "here's the
# rule" before it judges risks it hallucinating compliance rather than actually
# reasoning about the text — observed directly (2026-08-12, pt_ARC devotional
# review): the critic flagged correct Brazilian Portuguese próclise ("me alegrar")
# as an error and proposed European-style ênclise ("alegrar-me") instead, because
# nothing told it which regional standard applies. Cheap, mechanical, Python-only
# checks — same shape as _check_dictionary_for_typo, but post-hoc instead of a
# before-the-call short-circuit, since this needs the critic's own quoted span and
# proposed replacement to detect the pattern.
_PROCLISE_TO_ENCLISE_RE = re.compile(r"^(me|te|se|nos|vos)\s+(\w+)$", re.IGNORECASE)


def _dismiss_known_false_positive(
    language: str, quoted_text: str, replacement_text: str | None
) -> str | None:
    """
    Returns a dismissal reason if this critic verdict matches a known false-positive
    pattern, else None. Only ever narrows an is_valid=True verdict to False — never
    the reverse, since a false negative here just falls back to normal human review.
    """
    if replacement_text is not None and replacement_text == quoted_text:
        # A no-op "fix" — observed directly (2026-08-12, Arabic devotional review):
        # the critic returned is_valid=True with replacement_text byte-for-byte
        # identical to quoted_text. Applying this would silently do nothing while
        # still consuming a fix_pass replace call and reporting a false "APPLIED"
        # to the reviewer — a quality gate every language needs, not just one.
        return (
            f"replacement_text is byte-for-byte identical to quoted_text "
            f"({quoted_text!r}) — dismissed as a no-op, not a real fix."
        )

    if language != "Brazilian Portuguese" or not replacement_text:
        return None

    match = _PROCLISE_TO_ENCLISE_RE.match(quoted_text.strip())
    if not match:
        return None
    pronoun, verb = match.group(1).lower(), match.group(2).lower()

    expected_enclise = f"{verb}-{pronoun}"
    if replacement_text.strip().lower() == expected_enclise:
        return (
            f"{quoted_text!r} is proclise (pronoun before verb), the standard, "
            "correct form in Brazilian Portuguese — dismissed. The proposed "
            f"replacement {replacement_text!r} is enclise, the European "
            "Portuguese norm, not an error in Brazilian usage."
        )
    return None


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

    For typo/grammar findings in any language a local LanguageTool server covers,
    LanguageTool is checked next: if it does NOT independently flag the same
    quoted span, that's meaningful — a real typo/grammar rule violation would
    normally trip LanguageTool's own rules too, so a miss dismisses the claim
    without a critic call. If LanguageTool also flags it, or has no module for
    this language, or the server is unreachable, that's inconclusive either way and
    falls through to the critic's own judgment, same as any other finding.
    """
    if language == "Filipino" and finding["category"] == "typo":
        dictionary_verdict = _check_dictionary_for_typo(finding["quoted_text"])
        if dictionary_verdict is False:
            return CriticFinding(
                quoted_text=finding["quoted_text"],
                issue=finding["issue"],
                category=finding["category"],
                proposed_text=finding.get("proposed_text"),
                verified=True,
                is_valid=False,
                replacement_text=None,
                critic_reasoning=(
                    f"{finding['quoted_text']!r} is a real word in the KWF "
                    "(Komisyon sa Wikang Filipino) dictionary — dismissed as a "
                    "typo claim without further review."
                ),
            )

    if finding["category"] in ("typo", "grammar") and is_supported(language):
        try:
            lt_no_match = (
                find_match_for(finding["quoted_text"], source_text, language) is None
            )
        except httpx.HTTPError:
            # The LanguageTool server itself is unreachable — inconclusive, not a
            # dismissal. Falls through to the model's unaided judgment below, same
            # as a KWF dictionary-site failure does above.
            lt_no_match = False
        if lt_no_match:
            return CriticFinding(
                quoted_text=finding["quoted_text"],
                issue=finding["issue"],
                category=finding["category"],
                proposed_text=finding.get("proposed_text"),
                verified=True,
                is_valid=False,
                replacement_text=None,
                critic_reasoning=(
                    f"LanguageTool ({language}) does not flag {finding['quoted_text']!r} "
                    "as a typo or grammar issue — dismissed without further review."
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
    response = invoke_structured(
        prompt | model,
        {
            "persona": persona,
            "source_text": source_text,
            "quoted_text": finding["quoted_text"],
            "issue": finding["issue"],
        },
    )

    replacement_text = response.replacement_text or None
    if replacement_text:
        replacement_text = _normalize_replacement_text(replacement_text)

    is_valid = response.is_valid
    reasoning = response.reasoning
    if is_valid:
        dismiss_reason = _dismiss_known_false_positive(
            language, finding["quoted_text"], replacement_text
        )
        if dismiss_reason:
            is_valid = False
            replacement_text = None
            reasoning = dismiss_reason

    return CriticFinding(
        quoted_text=finding["quoted_text"],
        issue=finding["issue"],
        category=finding["category"],
        proposed_text=finding.get("proposed_text"),
        verified=True,
        is_valid=is_valid,
        replacement_text=replacement_text,
        critic_reasoning=reasoning,
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
