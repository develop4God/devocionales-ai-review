"""
Tests for scripts/run_live_validation.py's pure logic: shard partitioning and
rate-limit error classification. Not covering main() itself -- that's an
integration entrypoint driving real graph.invoke() calls, exercised by the
manual live runs against real content, not by this unit suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_live_validation import (
    apply_shard,
    classify_rate_limit_error,
    parse_retry_after_seconds,
)


def _fake_bad_request_error(code: str, message: str = "") -> openai.BadRequestError:
    import httpx

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(
        429, request=request, json={"code": code, "message": message}
    )
    return openai.BadRequestError(
        message=message, response=response, body={"code": code, "message": message}
    )


def test_apply_shard_none_returns_everything():
    pending = [("a", "x"), ("b", "y"), ("c", "z")]
    assert apply_shard(pending, None) == pending


def test_apply_shard_partitions_deterministically():
    pending = [(str(i), "f") for i in range(10)]
    shard1 = apply_shard(pending, "1/3")
    shard2 = apply_shard(pending, "2/3")
    shard3 = apply_shard(pending, "3/3")

    # every item lands in exactly one shard
    combined = shard1 + shard2 + shard3
    assert sorted(combined) == sorted(pending)
    assert len(combined) == len(pending)
    assert not (set(shard1) & set(shard2))
    assert not (set(shard2) & set(shard3))


def test_apply_shard_rejects_out_of_range_index():
    pending = [("a", "x")]
    with pytest.raises(ValueError, match="1 <= i <= N"):
        apply_shard(pending, "3/2")
    with pytest.raises(ValueError, match="1 <= i <= N"):
        apply_shard(pending, "0/2")


def test_classify_rate_limit_error_per_minute():
    e = _fake_bad_request_error(
        "rate_limit_exceeded",
        "Rate limit reached ... on tokens per minute (TPM): ... try again in 2.5s.",
    )
    assert classify_rate_limit_error(e) == "per_minute"


def test_classify_rate_limit_error_daily_quota():
    e = _fake_bad_request_error(
        "rate_limit_exceeded",
        "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199369.",
    )
    assert classify_rate_limit_error(e) == "daily_quota"


def test_classify_rate_limit_error_none_for_unrelated_bad_request():
    e = _fake_bad_request_error("some_other_error", "unrelated failure")
    assert classify_rate_limit_error(e) is None


def test_classify_rate_limit_error_none_for_non_rate_limit_exception():
    assert classify_rate_limit_error(ValueError("not a rate limit")) is None


def test_parse_retry_after_seconds_extracts_and_adds_margin():
    e = _fake_bad_request_error(
        "rate_limit_exceeded", "Please try again in 4.2075s. Need more tokens?"
    )
    assert parse_retry_after_seconds(e) == pytest.approx(6.2075, abs=0.001)


def test_parse_retry_after_seconds_falls_back_to_default_when_unparseable():
    e = _fake_bad_request_error("rate_limit_exceeded", "no duration mentioned here")
    assert parse_retry_after_seconds(e) == 10.0
