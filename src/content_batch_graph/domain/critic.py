"""
The critic pass: a second, independent AI judges each verified finding for
correctness and proposes an exact surgical replacement.

verify_pass only proves a finding's quoted_text is present, verbatim, in the source
— it says nothing about whether the finding is actually right. This is the layer
that closes that gap: one real model call per finding (not one call for all
findings at once), so each judgment is independent and isn't biased by the others.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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
    """
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
