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

_DICTIONARY_GROUNDING_TEMPLATE = """\

You have been given real dictionary lookup results for the KWF (Komisyon sa \
Wikang Filipino) — the official Filipino language authority — to ground your \
judgment. Trust this dictionary data over your own unaided memory of Filipino \
spelling, since a specific word's real vs. non-standard status is exactly what \
the dictionary settles.

{lookup_summary}"""


def _describe_lookup(label: str, lookup: dict) -> str:
    if lookup["found"]:
        return (
            f"- {label} {lookup['word']!r}: FOUND in the KWF dictionary "
            f"({lookup['part_of_speech']}) — {lookup['definition']}"
        )
    return f"- {label} {lookup['word']!r}: NOT FOUND in the KWF dictionary."


def _build_dictionary_grounding(quoted_text: str) -> str:
    """
    Looks up quoted_text (and, if it's a single word, the word alone) in the real
    KWF dictionary and formats the result as grounding context for the critic
    prompt. Only meaningful for single-word typo claims — returns "" for anything
    else, so multi-word spans fall back to the model's own judgment unchanged.
    """
    word = quoted_text.strip()
    if not word or " " in word:
        return ""
    try:
        lookup = lookup_word(word)
    except httpx.HTTPError:
        # A dictionary-site failure (network, 5xx) should never block the pipeline
        # — fall back to the model's unaided judgment rather than erroring out.
        return ""
    summary = _describe_lookup("Quoted word", lookup)
    return _DICTIONARY_GROUNDING_TEMPLATE.format(lookup_summary=summary)


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

    For Filipino typo findings, the prompt is grounded with a real KWF dictionary
    lookup of the quoted word — this is a headword-existence check, not something
    that applies to grammar/awkward_phrasing findings, which aren't about whether a
    single word exists.
    """
    persona = _CRITIC_PERSONA.format(language=language)
    if language == "Filipino" and finding["category"] == "typo":
        persona += _build_dictionary_grounding(finding["quoted_text"])
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

    return CriticFinding(
        quoted_text=finding["quoted_text"],
        issue=finding["issue"],
        category=finding["category"],
        verified=True,
        is_valid=response.is_valid,
        replacement_text=response.replacement_text or None,
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
