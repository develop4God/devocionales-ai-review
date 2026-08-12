"""
The drift check pass: after a surgical fix is applied, a second independent AI call
re-reads each changed span in its new context to confirm the fix actually resolved
the issue and didn't introduce a new one nearby.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from content_batch_graph.domain.providers import get_model
from content_batch_graph.state import CriticFinding

_DRIFT_PERSONA = """\
You are an independent {language} language reviewer, re-checking a piece of text \
after a specific correction was applied to it.

You will be given the corrected text and the replacement text that was inserted. \
Read the replacement in its current context and decide:
1. Does the replacement read correctly, with no leftover trace of the original \
   issue?
2. Does the replacement introduce any new typo, grammar error, or awkward phrasing \
   in the surrounding text?

Report drift only if you find a real, concrete problem with the replacement or its \
immediate context. Do not flag anything outside the replacement's immediate \
context."""


class _DriftResponse(BaseModel):
    drift_detected: bool = Field(
        description="True if the replacement did not resolve the original issue, "
        "or introduced a new problem nearby."
    )
    notes: str = Field(description="Explanation, for a human reviewer.")


def run_drift_check(
    fixed_text: str,
    applied_findings: list[CriticFinding],
    language: str,
    provider_id: str | None = None,
) -> tuple[bool, str]:
    """
    Returns (drift_detected, notes). If no findings were applied, returns
    (False, note) without calling the model.
    """
    applicable = [
        f for f in applied_findings if f["is_valid"] and f["replacement_text"]
    ]
    if not applicable:
        return False, "No fixes were applied; nothing to check for drift."

    replacements_list = "\n".join(
        f"- {f['replacement_text']!r} (was fixing: {f['issue']})" for f in applicable
    )

    persona = _DRIFT_PERSONA.format(language=language)
    model = get_model(provider_id).with_structured_output(_DriftResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{persona}"),
            (
                "human",
                "Corrected text:\n{fixed_text}\n\nReplacements applied:\n{replacements_list}",
            ),
        ]
    )
    response = (prompt | model).invoke(
        {
            "persona": persona,
            "fixed_text": fixed_text,
            "replacements_list": replacements_list,
        }
    )

    return response.drift_detected, response.notes
