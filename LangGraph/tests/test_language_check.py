import httpx
import pytest

from content_batch_graph.domain.language_check import (
    check_text,
    find_match_for,
    is_supported,
)


def test_is_supported_true_for_covered_language():
    assert is_supported("Spanish") is True
    assert is_supported("spanish") is True  # case-insensitive


def test_is_supported_false_for_uncovered_language():
    # Hindi and Korean have no LanguageTool module — see language_check.py's
    # _SUPPORTED_LANGS comment.
    assert is_supported("Hindi") is False
    assert is_supported("Korean") is False


def test_check_text_returns_empty_for_unsupported_language_without_network_call(
    monkeypatch,
):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "should not contact the server for an unsupported language"
        )

    monkeypatch.setattr(httpx, "post", _fail_if_called)
    assert check_text("some text", "Hindi") == []


def test_check_text_returns_empty_for_empty_text_without_network_call(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not contact the server for empty text")

    monkeypatch.setattr(httpx, "post", _fail_if_called)
    assert check_text("", "Spanish") == []


def test_check_text_parses_matches_from_server_response(monkeypatch):
    def _fake_post(url, data, timeout):
        assert data["language"] == "es"
        return httpx.Response(
            200,
            json={
                "matches": [
                    {
                        "offset": 8,
                        "length": 3,
                        "message": "Possible typo.",
                        "replacements": [{"value": "los"}, {"value": "las"}],
                        "rule": {"category": {"id": "TYPOS"}},
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    matches = check_text("Este es lso mejores.", "Spanish")

    assert len(matches) == 1
    assert matches[0]["quoted_text"] == "lso"
    assert matches[0]["replacements"] == ["los", "las"]
    assert matches[0]["rule_category"] == "TYPOS"


def test_check_text_raises_on_server_failure(monkeypatch):
    def _fake_post(url, data, timeout):
        raise httpx.ConnectError("connection refused", request=None)

    monkeypatch.setattr(httpx, "post", _fake_post)
    with pytest.raises(httpx.HTTPError):
        check_text("some text", "Spanish")


def test_find_match_for_returns_match_when_quoted_text_present(monkeypatch):
    def _fake_post(url, data, timeout):
        return httpx.Response(
            200,
            json={
                "matches": [
                    {
                        "offset": 8,
                        "length": 3,
                        "message": "Possible typo.",
                        "replacements": [{"value": "los"}],
                        "rule": {"category": {"id": "TYPOS"}},
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    match = find_match_for("lso", "Este es lso mejores.", "Spanish")
    assert match is not None
    assert match["quoted_text"] == "lso"


def test_find_match_for_returns_none_when_quoted_text_not_flagged(monkeypatch):
    def _fake_post(url, data, timeout):
        return httpx.Response(
            200, json={"matches": []}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    assert find_match_for("lso", "Este es lso mejores.", "Spanish") is None
