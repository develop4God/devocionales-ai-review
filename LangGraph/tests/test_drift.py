from types import SimpleNamespace

from content_batch_graph.domain.drift import run_drift_check
from content_batch_graph.state import CriticFinding


def _critic_finding(
    quoted_text: str, issue: str, replacement_text: str, is_valid: bool = True
) -> CriticFinding:
    return CriticFinding(
        quoted_text=quoted_text,
        issue=issue,
        category="typo",
        verified=True,
        is_valid=is_valid,
        replacement_text=replacement_text,
        critic_reasoning="test reasoning",
    )


def test_drift_check_returns_no_drift_note_when_no_findings_applied():
    findings = [_critic_finding("teh", "typo", None, is_valid=False)]  # type: ignore[arg-type]
    drift_detected, notes = run_drift_check("Some text.", findings, "English")

    assert drift_detected is False
    assert "No fixes were applied" in notes


def test_drift_check_real_call_on_a_clean_correction():
    # Real call against the local Ollama default provider. The replacement is a
    # correct, unambiguous fix in context, so a correct drift check should not flag
    # it — but this is a small model, so we only assert the response shape, matching
    # the project's existing pattern for real-model tests (see test_flag.py).
    fixed_text = "This is the correct answer, and it is quite clear."
    findings = [_critic_finding("teh", "Likely typo.", "the")]
    drift_detected, notes = run_drift_check(fixed_text, findings, "English")

    assert isinstance(drift_detected, bool)
    assert isinstance(notes, str)


def test_drift_check_provider_id_overrides_default(monkeypatch):
    import content_batch_graph.domain.drift as drift_module

    captured = {}

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            return SimpleNamespace(drift_detected=False, notes="fine")

    def _fake_get_model(provider_id=None):
        captured["provider_id"] = provider_id
        return _FakeModel()

    monkeypatch.setattr(drift_module, "get_model", _fake_get_model)
    findings = [_critic_finding("teh", "issue", "the")]
    run_drift_check(
        "some text with the", findings, "English", provider_id="cerebras_default"
    )
    assert captured["provider_id"] == "cerebras_default"
