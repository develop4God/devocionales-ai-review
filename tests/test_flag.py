from types import SimpleNamespace

import pytest

from content_batch_graph.domain.flag import run_flag_pass
from content_batch_graph.domain.verify import verify_findings


def test_flag_pass_empty_input_returns_empty():
    assert run_flag_pass("", "English") == []


def test_flag_pass_returns_list_of_finding_shaped_dicts():
    # Real call against the local Ollama default provider (no API key needed) — see
    # config/providers.yml. Content is deliberately clean so a correct model should
    # report no findings, but we only assert the *shape* of the response: this is a
    # small model and its judgment isn't being graded here, verify_pass's job (real
    # source-text matching) is what makes any finding trustworthy downstream.
    findings = run_flag_pass("The sun rises in the east.", "English")
    assert isinstance(findings, list)
    for f in findings:
        assert set(f.keys()) == {"quoted_text", "issue", "category"}
        assert isinstance(f["quoted_text"], str)
        assert isinstance(f["issue"], str)
        assert isinstance(f["category"], str)


def test_flag_pass_findings_pass_through_verify_pass_safely():
    # Ties back to the project's core discipline (domain/verify.py): a real model's
    # raw claim is never trusted directly. With this local model (qwen2.5:0.5b), the
    # quoted_text for a caught typo sometimes comes back "corrected" rather than
    # verbatim — verify_findings must catch and reject that, not let it through as
    # verified. This is the exact case the verify-before-trust layer exists for, so
    # the test asserts the safety property (nothing rejected leaks into "verified"),
    # not that the small model always quotes perfectly.
    source = "This is the correct answer, but teh explanation is not clear enough."
    findings = run_flag_pass(source, "English")
    verified, rejected = verify_findings(findings, source)

    assert len(verified) + len(rejected) == len(findings)
    for vf in verified:
        assert vf["quoted_text"] in source


def test_flag_pass_raises_for_unknown_role_id():
    with pytest.raises(ValueError, match="not found"):
        run_flag_pass("some text", "English", role_id="nonexistent_role")


def test_flag_pass_raises_for_unknown_provider_id():
    with pytest.raises(ValueError, match="not found or not enabled"):
        run_flag_pass("some text", "English", provider_id="nonexistent_provider")


def test_flag_pass_provider_id_overrides_default(monkeypatch):
    # Confirms provider_id is actually threaded into get_model(), not ignored —
    # patches get_model to capture what it was called with instead of hitting a
    # real provider.
    import content_batch_graph.domain.flag as flag_module

    captured = {}

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            return SimpleNamespace(findings=[])

    def _fake_get_model(provider_id=None):
        captured["provider_id"] = provider_id
        return _FakeModel()

    monkeypatch.setattr(flag_module, "get_model", _fake_get_model)
    run_flag_pass("some text", "English", provider_id="cerebras_default")
    assert captured["provider_id"] == "cerebras_default"
