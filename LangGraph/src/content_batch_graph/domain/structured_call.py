"""
A shared retry wrapper around invoking a structured-output LangChain chain,
used by every domain module that calls .with_structured_output() (flag.py,
critic.py, drift.py) — one call site owning this logic, not three copies that
could drift apart.

Structured-output enforcement across providers has been observed intermittently
failing on an otherwise well-formed request, in ways get_model()'s max_retries
(which only covers the underlying SDK's retryable errors, e.g. a 429) never
catches:

- Groq's gpt-oss models: a 400 openai.BadRequestError with code
  json_validate_failed (the model's JSON violates its own declared schema) or
  output_parse_failed (the model's raw text isn't valid JSON at all) —
  confirmed against real content on 2026-08-27.
- Local Ollama models (confirmed with gemma4:26b on 2026-08-27, live validation
  run against the 2027 ES devotional file): langchain_core.exceptions.
  OutputParserException, when the model's raw output can't be coerced into the
  declared schema at all. Unlike the Groq case this killed the whole worker
  process outright (run_live_validation.py's top-level catch-all stopped the
  run), since nothing upstream of chain.invoke() was retrying it.

Retrying the identical call is a real-world-confirmed fix for the Groq case
(15/15 clean calls after retries were added, was ~5-7/10 before); the same
treatment is extended here to OutputParserException on the same reasoning --
a single call retried once or twice is cheap, and a structured-output failure
on well-formed input is characteristic of provider-side flakiness, not a
scenario where retrying is expected to make things worse.
"""

from __future__ import annotations

from typing import TypeVar

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable

_T = TypeVar("_T")

_RETRYABLE_GROQ_ERROR_CODES = {"json_validate_failed", "output_parse_failed"}
_MAX_STRUCTURED_OUTPUT_RETRIES = 2


def invoke_structured(chain: Runnable, inputs: dict) -> _T:
    """
    Calls chain.invoke(inputs), retrying up to _MAX_STRUCTURED_OUTPUT_RETRIES
    times on a known-flaky structured-output failure (see module docstring).
    Any other exception propagates immediately, unretried.
    """
    attempt = 0
    while True:
        try:
            return chain.invoke(inputs)
        except openai.BadRequestError as e:
            # e.body is the error object's own contents directly (e.g.
            # {"message": ..., "code": "json_validate_failed", ...}) — confirmed
            # against real Groq 400 responses on 2026-08-27. It is NOT nested
            # under an "error" key the way the raw HTTP JSON body is.
            error_code = e.body.get("code") if isinstance(e.body, dict) else None
            if (
                error_code not in _RETRYABLE_GROQ_ERROR_CODES
                or attempt >= _MAX_STRUCTURED_OUTPUT_RETRIES
            ):
                raise
            attempt += 1
        except OutputParserException:
            if attempt >= _MAX_STRUCTURED_OUTPUT_RETRIES:
                raise
            attempt += 1
