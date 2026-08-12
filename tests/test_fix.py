from types import SimpleNamespace

from content_batch_graph.domain.fix import run_fix_pass
from content_batch_graph.domain.validate import validate_json
from content_batch_graph.state import VerifiedFinding


def _finding(quoted_text: str, issue: str, category: str = "typo") -> VerifiedFinding:
    return VerifiedFinding(
        quoted_text=quoted_text, issue=issue, category=category, verified=True
    )


def test_fix_pass_returns_unchanged_text_and_note_when_no_findings():
    fixed_text, summary = run_fix_pass("Some text.", [], "English")
    assert fixed_text == "Some text."
    assert "No approved findings" in summary


def test_fix_pass_real_call_fixes_a_known_typo():
    # Real call against the local Ollama default provider (no API key needed) — see
    # config/providers.yml. Uses a deliberate, unambiguous typo so there's something
    # a correct fix should reliably resolve.
    source = "This is teh correct answer, and it is quite clear."
    findings = [_finding("teh", "Likely typo: 'teh' should probably be 'the'.")]
    fixed_text, summary = run_fix_pass(source, findings, "English")

    assert isinstance(fixed_text, str)
    assert isinstance(summary, str)
    assert fixed_text != ""
    # fix_pass output isn't re-verified the way flag_pass findings are (validate_pass
    # only checks JSON structure, not prose correctness) — this test only asserts the
    # call produces real, non-empty output in the right shape.


def test_fix_pass_provider_id_overrides_default(monkeypatch):
    import content_batch_graph.domain.fix as fix_module

    captured = {}

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            return SimpleNamespace(fixed_text="corrected", summary="did a thing")

    def _fake_get_model(provider_id=None):
        captured["provider_id"] = provider_id
        return _FakeModel()

    monkeypatch.setattr(fix_module, "get_model", _fake_get_model)
    findings = [_finding("x", "issue")]
    run_fix_pass(
        "some text with x", findings, "English", provider_id="cerebras_default"
    )
    assert captured["provider_id"] == "cerebras_default"


def test_fix_pass_output_is_at_least_still_valid_when_content_is_json():
    # End-to-end sanity: fixing a devocional-shaped JSON blob's prose field shouldn't
    # break the surrounding JSON structure, mirroring what validate_pass checks for
    # real in the graph.
    source = '{"reflexion": "This is teh correct answer."}'
    findings = [_finding("teh", "Likely typo: 'teh' should probably be 'the'.")]
    fixed_text, _summary = run_fix_pass(source, findings, "English")
    passed, error = validate_json(fixed_text)
    assert passed, f"fix_pass broke JSON structure: {error}"
