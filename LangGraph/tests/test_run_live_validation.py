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
    auto_decide,
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


def _finding(category: str, is_valid: bool) -> dict:
    return {
        "quoted_text": "x",
        "issue": "y",
        "category": category,
        "is_valid": is_valid,
        "replacement_text": "z",
        "critic_reasoning": "because",
    }


def test_auto_decide_applies_valid_typo_findings():
    findings = [_finding("typo", True)]
    assert auto_decide(findings) == [0]


def test_auto_decide_excludes_grammar_even_when_valid():
    # 2026-08-27 audit: critic_pass's own proposed replacement can itself be wrong
    # or drop content unrelated to the claimed issue, undetected by any other
    # stage. grammar/awkward_phrasing are never auto-applied regardless of the
    # critic's is_valid verdict -- they must reach a human review pass instead.
    findings = [_finding("grammar", True)]
    assert auto_decide(findings) == []


def test_auto_decide_excludes_awkward_phrasing_even_when_valid():
    findings = [_finding("awkward_phrasing", True)]
    assert auto_decide(findings) == []


def test_auto_decide_excludes_invalid_typo():
    findings = [_finding("typo", False)]
    assert auto_decide(findings) == []


def test_auto_decide_mixed_categories_only_returns_valid_typo_indices():
    findings = [
        _finding("typo", True),  # 0: included
        _finding("grammar", True),  # 1: excluded (category)
        _finding("typo", False),  # 2: excluded (not valid)
        _finding("awkward_phrasing", True),  # 3: excluded (category)
        _finding("typo", True),  # 4: included
    ]
    assert auto_decide(findings) == [0, 4]


def test_apply_shard_none_returns_everything():
    items = [("a", "x"), ("b", "y"), ("c", "z")]
    assert apply_shard(items, None) == items


def test_apply_shard_partitions_deterministically():
    items = [(str(i), "f") for i in range(10)]
    shard1 = apply_shard(items, "1/3")
    shard2 = apply_shard(items, "2/3")
    shard3 = apply_shard(items, "3/3")

    # every item lands in exactly one shard
    combined = shard1 + shard2 + shard3
    assert sorted(combined) == sorted(items)
    assert len(combined) == len(items)
    assert not (set(shard1) & set(shard2))
    assert not (set(shard2) & set(shard3))


def test_apply_shard_rejects_out_of_range_index():
    items = [("a", "x")]
    with pytest.raises(ValueError, match="1 <= i <= N"):
        apply_shard(items, "3/2")
    with pytest.raises(ValueError, match="1 <= i <= N"):
        apply_shard(items, "0/2")


def test_apply_shard_membership_is_stable_as_items_are_completed():
    # Regression test: apply_shard must be called on the full, stable item list
    # (main()'s all_items) and only THEN filtered by what a shared ledger says is
    # done -- not the other way around. Sharding an already-filtered "pending"
    # list means a shard's membership shifts as items get done (by this worker or
    # any other worker sharing the same ledger), which is exactly how two workers
    # ended up both about to claim the same item in a real two-worker run on
    # 2026-08-27 (caught before either made an API call).
    all_items = [(str(i), "f") for i in range(20)]

    # Correct order: shard the full list, independent of what's been completed
    # elsewhere. Confirms it stays identical across two calls -- nothing about
    # ledger state can enter this computation.
    assert set(apply_shard(all_items, "1/2")) == set(apply_shard(all_items, "1/2"))

    # The bug this guards against: sharding an already-filtered ("pending") list
    # instead produces a DIFFERENT set once other items have been marked done --
    # demonstrating why apply_shard's docstring requires the full list as input,
    # not a ledger-filtered one.
    correct_shard = set(apply_shard(all_items, "1/2"))
    done_by_other_worker = {all_items[3], all_items[7], all_items[12]}
    remaining_after_other_worker = [
        item for item in all_items if item not in done_by_other_worker
    ]
    buggy_shard = set(apply_shard(remaining_after_other_worker, "1/2"))
    assert buggy_shard != correct_shard


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
