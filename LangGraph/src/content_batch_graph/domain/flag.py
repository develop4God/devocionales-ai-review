"""
The flag pass: read source text, produce raw (unverified) findings.

Calls a real LLM, resolved via domain/providers.get_model(), with the persona/
instructions for a role resolved via domain/roles.get_role(), and parses its
response into structured Finding objects via .with_structured_output() — no
prompted "please return JSON" and no text-unwrapping/regex parsing of the response.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, create_model

from content_batch_graph.domain.providers import get_model
from content_batch_graph.domain.roles import Role, get_role
from content_batch_graph.state import Finding


def _build_flag_response_schema(role: Role) -> type[BaseModel]:
    """
    Builds the structured-output schema for this role, with `category` constrained
    to a Literal of exactly the role's declared categories — schema-level enforcement
    instead of relying on a free-text description, which small models don't reliably
    follow.
    """
    category_type = Literal[tuple(role["categories"])]  # type: ignore[valid-type]
    finding_schema = create_model(
        "_FindingSchema",
        quoted_text=(
            str,
            Field(
                description="The exact problematic text, quoted verbatim from the source."
            ),
        ),
        issue=(
            str,
            Field(description="What's wrong with the quoted text, in English."),
        ),
        proposed_text=(
            str,
            Field(
                default="",
                description="The corrected replacement for quoted_text. Empty if "
                "no replacement is proposed.",
            ),
        ),
        category=(category_type, Field(description="The category of this finding.")),
    )
    return create_model(
        "_FlagResponse",
        findings=(
            list[finding_schema],
            Field(
                default_factory=list,
                description="Every issue found. Empty if the text has no issues.",
            ),
        ),
    )


def run_flag_pass(
    source_text: str,
    language: str,
    role_id: str | None = None,
    provider_id: str | None = None,
) -> list[Finding]:
    """
    Calls a real model, acting as the given role (default: native_reader), to review
    source_text written in `language` and return its findings as a verifiable
    list[Finding] — nothing in this list is trusted yet; verify_pass checks each
    quoted_text is actually present in the source before treating it as real.

    provider_id overrides providers.yml's default_provider for this call — used to
    compare candidate providers/models against the same role and content.
    """
    if not source_text:
        return []

    role = get_role(role_id)
    persona = role["persona"].format(language=language)
    flag_response_schema = _build_flag_response_schema(role)
    model = get_model(provider_id).with_structured_output(flag_response_schema)

    prompt = ChatPromptTemplate.from_messages(
        [("system", "{persona}"), ("human", "Text to review:\n{source_text}")]
    )
    response = (prompt | model).invoke({"persona": persona, "source_text": source_text})

    return [
        Finding(
            quoted_text=f.quoted_text,
            issue=f.issue,
            category=f.category,
            proposed_text=f.proposed_text or None,
        )
        for f in response.findings
    ]
