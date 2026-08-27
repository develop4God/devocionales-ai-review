"""
Tests for domain/structured_call.py's shared retry wrapper, used by flag.py,
critic.py, and drift.py -- one call site, tested once, instead of duplicated
retry logic tested (or not) in three places.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from langchain_core.exceptions import OutputParserException

from content_batch_graph.domain.structured_call import invoke_structured


def _fake_bad_request_error(code: str, message: str = "") -> openai.BadRequestError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(
        400, request=request, json={"code": code, "message": message}
    )
    return openai.BadRequestError(
        message=message, response=response, body={"code": code, "message": message}
    )


class _FakeChain:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.call_count = 0

    def invoke(self, inputs):
        self.call_count += 1
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


@pytest.mark.parametrize("error_code", ["json_validate_failed", "output_parse_failed"])
def test_invoke_structured_retries_on_retryable_groq_errors(error_code):
    chain = _FakeChain([_fake_bad_request_error(error_code), "ok"])
    result = invoke_structured(chain, {})
    assert result == "ok"
    assert chain.call_count == 2


def test_invoke_structured_retries_on_output_parser_exception():
    # Confirmed real failure mode: local Ollama (gemma4:26b) structured-output
    # calls raised langchain_core.exceptions.OutputParserException directly (not
    # an openai.BadRequestError, since Ollama isn't the openai package) during a
    # live validation run on 2026-08-27, killing the whole worker process because
    # nothing upstream retried it.
    chain = _FakeChain([OutputParserException("could not parse"), "ok"])
    result = invoke_structured(chain, {})
    assert result == "ok"
    assert chain.call_count == 2


def test_invoke_structured_does_not_retry_unrelated_bad_request_errors():
    chain = _FakeChain([_fake_bad_request_error("some_other_error")])
    with pytest.raises(openai.BadRequestError):
        invoke_structured(chain, {})
    assert chain.call_count == 1


def test_invoke_structured_raises_after_exhausting_retries():
    # 3 consecutive failures (1 initial + 2 retries) still propagates.
    chain = _FakeChain(
        [
            _fake_bad_request_error("json_validate_failed"),
            _fake_bad_request_error("json_validate_failed"),
            _fake_bad_request_error("json_validate_failed"),
        ]
    )
    with pytest.raises(openai.BadRequestError):
        invoke_structured(chain, {})
    assert chain.call_count == 3


def test_invoke_structured_raises_after_exhausting_output_parser_retries():
    chain = _FakeChain(
        [
            OutputParserException("1"),
            OutputParserException("2"),
            OutputParserException("3"),
        ]
    )
    with pytest.raises(OutputParserException):
        invoke_structured(chain, {})
    assert chain.call_count == 3


def test_invoke_structured_succeeds_immediately_with_no_retries_needed():
    chain = _FakeChain(["ok"])
    result = invoke_structured(chain, {})
    assert result == "ok"
    assert chain.call_count == 1


def test_invoke_structured_can_mix_error_types_across_retries():
    # A Groq structured-output failure followed by an Ollama-style one, then
    # success -- both error types share the same retry budget.
    chain = _FakeChain(
        [
            _fake_bad_request_error("json_validate_failed"),
            OutputParserException("could not parse"),
            "ok",
        ]
    )
    result = invoke_structured(chain, {})
    assert result == "ok"
    assert chain.call_count == 3
