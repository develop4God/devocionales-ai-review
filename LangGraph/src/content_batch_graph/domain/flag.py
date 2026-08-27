"""
The flag pass: read source text, produce raw (unverified) findings.

Calls a real LLM, resolved via domain/providers.get_model(), with the persona/
instructions for a role resolved via domain/roles.get_role(), and parses its
response into structured Finding objects via .with_structured_output() — no
prompted "please return JSON" and no text-unwrapping/regex parsing of the response.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from content_batch_graph.domain.providers import get_model
from content_batch_graph.domain.roles import build_finding_schema, get_role
from content_batch_graph.domain.structured_call import invoke_structured
from content_batch_graph.state import Finding


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
    flag_response_schema = build_finding_schema(role)
    model = get_model(provider_id).with_structured_output(flag_response_schema)

    prompt = ChatPromptTemplate.from_messages(
        [("system", "{persona}"), ("human", "Text to review:\n{source_text}")]
    )
    chain = prompt | model
    inputs = {"persona": persona, "source_text": source_text}
    response = invoke_structured(chain, inputs)

    return [
        Finding(
            quoted_text=f.quoted_text,
            issue=f.issue,
            category=f.category,
            proposed_text=f.proposed_text or None,
        )
        for f in response.findings
    ]
