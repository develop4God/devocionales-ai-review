"""
The flag pass: read source text, produce raw (unverified) findings.

Calls a real LLM, resolved via domain/providers.get_model(), with the persona/
instructions for a role resolved via domain/roles.get_role(), and parses its
response into structured Finding objects via .with_structured_output() — no
prompted "please return JSON" and no text-unwrapping/regex parsing of the response.
"""

from __future__ import annotations

import openai
from langchain_core.prompts import ChatPromptTemplate

from content_batch_graph.domain.providers import get_model
from content_batch_graph.domain.roles import build_finding_schema, get_role
from content_batch_graph.state import Finding

# Groq's structured-output enforcement has been observed intermittently failing on
# an otherwise well-formed request — either emitting JSON that violates its own
# declared schema (json_validate_failed) or emitting text that isn't valid JSON at
# all (output_parse_failed, the model's raw reasoning text leaking into the content
# field instead of a schema-shaped response). Neither is a network/rate-limit
# error, so get_model()'s max_retries (which only covers the underlying SDK's
# retryable errors) never catches either. Retrying the identical call is a real-
# world-confirmed fix for both: reproduced against real content on 2026-08-27 —
# 1 retry cleared most failures but left ~1/10 calls still failing (both codes
# observed); 2 retries is the current setting.
_RETRYABLE_ERROR_CODES = {"json_validate_failed", "output_parse_failed"}
_MAX_STRUCTURED_OUTPUT_RETRIES = 2


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

    attempt = 0
    while True:
        try:
            response = chain.invoke(inputs)
            break
        except openai.BadRequestError as e:
            # e.body is the error object's own contents directly (e.g.
            # {"message": ..., "code": "json_validate_failed", ...}) — confirmed
            # against real Groq 400 responses on 2026-08-27. It is NOT nested under
            # an "error" key the way the raw HTTP JSON body is; assuming that nesting
            # here previously meant this check never matched and no retry ever fired.
            error_code = e.body.get("code") if isinstance(e.body, dict) else None
            if (
                error_code not in _RETRYABLE_ERROR_CODES
                or attempt >= _MAX_STRUCTURED_OUTPUT_RETRIES
            ):
                raise
            attempt += 1

    return [
        Finding(
            quoted_text=f.quoted_text,
            issue=f.issue,
            category=f.category,
            proposed_text=f.proposed_text or None,
        )
        for f in response.findings
    ]
