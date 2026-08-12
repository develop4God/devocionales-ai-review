from types import SimpleNamespace

from content_batch_graph.domain.critic import (
    _build_dictionary_grounding,
    run_critic_pass,
    run_critic_pass_batch,
)
from content_batch_graph.state import VerifiedFinding


def _finding(quoted_text: str, issue: str, category: str = "typo") -> VerifiedFinding:
    return VerifiedFinding(
        quoted_text=quoted_text, issue=issue, category=category, verified=True
    )


def test_critic_pass_real_call_judges_a_known_typo():
    # Real call against the local Ollama default provider (no API key needed) — see
    # config/providers.yml. Uses a deliberate, unambiguous typo so a correct critic
    # should confirm it and propose "the" as the replacement.
    source = "This is teh correct answer, and it is quite clear."
    finding = _finding("teh", "Likely typo: 'teh' should probably be 'the'.")
    result = run_critic_pass(source, finding, "English")

    assert result["quoted_text"] == "teh"
    assert isinstance(result["is_valid"], bool)
    assert isinstance(result["critic_reasoning"], str)
    # Small local models aren't graded on judgment quality here — only that the call
    # returns the right shape. verify_pass-style trust-but-verify happens downstream
    # when replacement_text is actually applied by fix_pass (a plain string replace).


def test_critic_pass_batch_preserves_order_and_count(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    def _fake_run_critic_pass(source_text, finding, language, provider_id=None):
        return {
            "quoted_text": finding["quoted_text"],
            "issue": finding["issue"],
            "category": finding["category"],
            "verified": True,
            "is_valid": True,
            "replacement_text": finding["quoted_text"].upper(),
            "critic_reasoning": "stub",
        }

    monkeypatch.setattr(critic_module, "run_critic_pass", _fake_run_critic_pass)

    findings = [_finding("a", "issue a"), _finding("b", "issue b")]
    results = run_critic_pass_batch("source text a b", findings, "English")

    assert [r["quoted_text"] for r in results] == ["a", "b"]
    assert [r["replacement_text"] for r in results] == ["A", "B"]


def test_critic_pass_provider_id_overrides_default(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    captured = {}

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            return SimpleNamespace(
                is_valid=True, replacement_text="fixed", reasoning="why"
            )

    def _fake_get_model(provider_id=None):
        captured["provider_id"] = provider_id
        return _FakeModel()

    monkeypatch.setattr(critic_module, "get_model", _fake_get_model)
    run_critic_pass(
        "some text",
        _finding("x", "issue"),
        "English",
        provider_id="cerebras_default",
    )
    assert captured["provider_id"] == "cerebras_default"


def test_build_dictionary_grounding_reports_found_word(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    def _fake_lookup_word(word):
        return {
            "word": word,
            "found": True,
            "part_of_speech": "Pang-uri",
            "definition": "a real definition",
        }

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    grounding = _build_dictionary_grounding("espiritwal")

    assert "FOUND" in grounding
    assert "espiritwal" in grounding
    assert "a real definition" in grounding


def test_build_dictionary_grounding_reports_not_found_word(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    def _fake_lookup_word(word):
        return {
            "word": word,
            "found": False,
            "part_of_speech": None,
            "definition": None,
        }

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    grounding = _build_dictionary_grounding("espirituwal")

    assert "NOT FOUND" in grounding
    assert "espirituwal" in grounding


def test_build_dictionary_grounding_returns_empty_for_multi_word_span():
    # Dictionary grounding is a headword-existence check — meaningless for a
    # multi-word span, so it should skip the lookup entirely.
    grounding = _build_dictionary_grounding("isang patuloy na")
    assert grounding == ""


def test_build_dictionary_grounding_returns_empty_on_lookup_failure(monkeypatch):
    import httpx

    import content_batch_graph.domain.critic as critic_module

    def _raise(word):
        raise httpx.ConnectError("simulated dictionary site failure")

    monkeypatch.setattr(critic_module, "lookup_word", _raise)
    grounding = _build_dictionary_grounding("espiritwal")

    assert grounding == ""


def test_run_critic_pass_grounds_filipino_typo_findings(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    def _fake_lookup_word(word):
        return {
            "word": word,
            "found": False,
            "part_of_speech": None,
            "definition": None,
        }

    captured = {}

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, inputs):
            captured["system_message"] = inputs.messages[0].content
            return SimpleNamespace(
                is_valid=True, replacement_text="espiritwal", reasoning="why"
            )

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    monkeypatch.setattr(
        critic_module, "get_model", lambda provider_id=None: _FakeModel()
    )

    run_critic_pass(
        "some text with espirituwal in it",
        _finding("espirituwal", "possible typo"),
        "Filipino",
    )

    assert "NOT FOUND" in captured["system_message"]
    assert "espirituwal" in captured["system_message"]


def test_run_critic_pass_skips_grounding_for_non_filipino_language(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    called = {"count": 0}

    def _fake_lookup_word(word):
        called["count"] += 1
        return {"word": word, "found": True, "part_of_speech": "x", "definition": "y"}

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            return SimpleNamespace(
                is_valid=True, replacement_text="the", reasoning="why"
            )

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    monkeypatch.setattr(
        critic_module, "get_model", lambda provider_id=None: _FakeModel()
    )

    run_critic_pass("some text", _finding("teh", "typo"), "English")

    assert called["count"] == 0


def test_run_critic_pass_skips_grounding_for_non_typo_category(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    called = {"count": 0}

    def _fake_lookup_word(word):
        called["count"] += 1
        return {"word": word, "found": True, "part_of_speech": "x", "definition": "y"}

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            return SimpleNamespace(
                is_valid=True, replacement_text="fixed", reasoning="why"
            )

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    monkeypatch.setattr(
        critic_module, "get_model", lambda provider_id=None: _FakeModel()
    )

    run_critic_pass(
        "some text",
        _finding("dinisenyo", "awkward", category="awkward_phrasing"),
        "Filipino",
    )

    assert called["count"] == 0
