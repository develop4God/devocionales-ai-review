"""
Shared fixtures: a fake urllib layer so no test in this package touches the network.
"""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import dataclass, field

import pytest

from batch_common.config import BatchProviderConfig


@pytest.fixture
def cfg() -> BatchProviderConfig:
    return BatchProviderConfig(
        provider_id="fake_batch",
        base_url="https://api.example.test/inference/v1/",
        model="fake/model",
        env_var="FAKE_BATCH_API_KEY",
        account_id_env_var="FAKE_BATCH_ACCOUNT_ID",
    )


@pytest.fixture
def api_key(monkeypatch) -> str:
    monkeypatch.setenv("FAKE_BATCH_API_KEY", "test-key")
    return "test-key"


@pytest.fixture
def account_id(monkeypatch) -> str:
    monkeypatch.setenv("FAKE_BATCH_ACCOUNT_ID", "test-account")
    return "test-account"


class _FakeResponse(io.BytesIO):
    """Minimal context-manager stand-in for urlopen's return value."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@dataclass
class FakeHTTP:
    """
    Records every request and replays a queued list of responses.

    A queued entry is either a dict (JSON-encoded), bytes (raw body), or an
    HTTPError instance (raised).
    """

    responses: list = field(default_factory=list)
    requests: list = field(default_factory=list)

    def urlopen(self, req, timeout=None):
        self.requests.append(req)
        if not self.responses:
            raise AssertionError("FakeHTTP ran out of queued responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, urllib.error.HTTPError):
            raise nxt
        if isinstance(nxt, bytes):
            return _FakeResponse(nxt)
        return _FakeResponse(json.dumps(nxt).encode())


@pytest.fixture
def fake_http(monkeypatch) -> FakeHTTP:
    http = FakeHTTP()
    monkeypatch.setattr("batch_common.client.urllib.request.urlopen", http.urlopen)
    # poll() sleeps between non-terminal statuses — make that free.
    monkeypatch.setattr("batch_common.client.time.sleep", lambda _s: None)
    return http


def http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.example.test/x",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(body.encode()),
    )
