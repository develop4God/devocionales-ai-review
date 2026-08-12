
import pytest

from content_batch_graph.domain.providers import get_model


def test_get_model_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_model("anthropic_default")


def test_get_model_raises_for_unknown_provider_id():
    with pytest.raises(ValueError, match="not found or not enabled"):
        get_model("nonexistent_provider")


def test_get_model_returns_chat_anthropic_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-construction-only")
    model = get_model("anthropic_default")
    assert type(model).__name__ == "ChatAnthropic"


def test_get_model_uses_default_provider_when_none_given(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-construction-only")
    model = get_model()
    assert type(model).__name__ == "ChatAnthropic"
