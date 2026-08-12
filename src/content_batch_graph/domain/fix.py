"""
The fix pass: apply a correction for each approved finding, produce the corrected
text plus a concise summary of what changed, for human review.

Calls a real LLM via domain/providers.get_model(), parsed into structured output —
same pattern as domain/flag.py. This is intentionally a single pass that rewrites the
whole file_text, not per-finding edits, so the model can keep the corrections
consistent with each other and with the surrounding text.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from content_batch_graph.domain.providers import get_model
from content_batch_graph.state import VerifiedFinding

_FIX_PERSONA = """\
You are a careful {language} editor. You will be given a piece of {language} text \
and a list of specific issues found in it. For each issue, apply the minimal \
correction needed to fix it — do not rewrite anything beyond what's needed to \
address the listed issues, and do not change formatting, structure, or any part \
of the text unrelated to the listed issues.

Return the full corrected text, and a concise summary (a few sentences) of what \
you changed and why, for a human reviewer who has not seen the original issues \
list."""


class _FixResponse(BaseModel):
    fixed_text: str = Field(
        description="The full text with each listed issue corrected, and "
        "nothing else changed."
    )
    summary: str = Field(
        description="Concise summary of what was changed and why, for a human reviewer."
    )


def run_fix_pass(
    file_text: str,
    findings: list[VerifiedFinding],
    language: str,
    provider_id: str | None = None,
) -> tuple[str, str]:
    """
    Returns (fixed_text, summary). If findings is empty, returns file_text unchanged
    with a summary noting nothing needed fixing.
    """
    if not findings:
        return file_text, "No approved findings to fix."

    issues_list = "\n".join(
        f"- ({f['category']}) {f['quoted_text']!r}: {f['issue']}" for f in findings
    )

    persona = _FIX_PERSONA.format(language=language)
    model = get_model(provider_id).with_structured_output(_FixResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{persona}"),
            (
                "human",
                "Text:\n{file_text}\n\nIssues to fix:\n{issues_list}",
            ),
        ]
    )
    response = (prompt | model).invoke(
        {"persona": persona, "file_text": file_text, "issues_list": issues_list}
    )

    return response.fixed_text, response.summary
