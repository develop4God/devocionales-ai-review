"""BatchProviderConfig defaults and env-var key resolution."""

from __future__ import annotations

import dataclasses

import pytest
from batch_common.config import BatchAPIError, BatchProviderConfig, api_key_from_env


def _cfg(**kw) -> BatchProviderConfig:
    base = {
        "provider_id": "p1",
        "base_url": "https://x.test/v1",
        "model": "m/1",
        "env_var": "P1_API_KEY",
    }
    base.update(kw)
    return BatchProviderConfig(**base)


def test_defaults():
    cfg = _cfg()
    assert cfg.endpoint == "/v1/chat/completions"
    assert cfg.completion_window == "24h"
    assert cfg.extra_record_fields is None


def test_api_key_from_env_returns_key(monkeypatch):
    monkeypatch.setenv("P1_API_KEY", "sk-123")
    assert api_key_from_env(_cfg()) == "sk-123"


def test_api_key_from_env_raises_naming_the_variable(monkeypatch):
    monkeypatch.delenv("P1_API_KEY", raising=False)
    with pytest.raises(BatchAPIError, match="P1_API_KEY"):
        api_key_from_env(_cfg())


def test_api_key_from_env_treats_empty_string_as_missing(monkeypatch):
    monkeypatch.setenv("P1_API_KEY", "")
    with pytest.raises(BatchAPIError, match="P1_API_KEY"):
        api_key_from_env(_cfg())


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _cfg().model = "other"  # type: ignore[misc]
