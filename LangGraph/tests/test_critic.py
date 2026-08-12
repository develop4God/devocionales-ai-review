from types import SimpleNamespace

from content_batch_graph.domain.critic import (
    _check_dictionary_for_typo,
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


def test_check_dictionary_for_typo_returns_false_when_word_found(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    def _fake_lookup_word(word):
        return {
            "word": word,
            "found": True,
            "part_of_speech": "Pang-uri",
            "definition": "a real definition",
        }

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    assert _check_dictionary_for_typo("espiritwal") is False


def test_check_dictionary_for_typo_returns_none_when_word_not_found(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    def _fake_lookup_word(word):
        return {
            "word": word,
            "found": False,
            "part_of_speech": None,
            "definition": None,
        }

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    # A miss is inconclusive (could be a typo, could be an inflected form the
    # dictionary doesn't list separately) — never treated as proof of a typo.
    assert _check_dictionary_for_typo("kalaking") is None


def test_check_dictionary_for_typo_returns_none_for_multi_word_span():
    # A headword-existence check is meaningless for a multi-word span.
    assert _check_dictionary_for_typo("isang patuloy na") is None


def test_check_dictionary_for_typo_returns_none_on_lookup_failure(monkeypatch):
    import httpx

    import content_batch_graph.domain.critic as critic_module

    def _raise(word):
        raise httpx.ConnectError("simulated dictionary site failure")

    monkeypatch.setattr(critic_module, "lookup_word", _raise)
    assert _check_dictionary_for_typo("espiritwal") is None


def test_run_critic_pass_dismisses_typo_claim_when_word_is_in_dictionary(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    def _fake_lookup_word(word):
        return {
            "word": word,
            "found": True,
            "part_of_speech": "Pang-uri",
            "definition": "a real definition",
        }

    model_called = {"count": 0}

    class _FakeModel:
        def with_structured_output(self, schema):
            model_called["count"] += 1
            return self

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    monkeypatch.setattr(
        critic_module, "get_model", lambda provider_id=None: _FakeModel()
    )

    result = run_critic_pass(
        "some text with espiritwal in it",
        _finding("espiritwal", "possible typo"),
        "Filipino",
    )

    assert result["is_valid"] is False
    assert result["replacement_text"] is None
    assert "KWF" in result["critic_reasoning"]
    # The whole point of the SOT check: a dictionary hit dismisses the claim
    # immediately, with no model call at all.
    assert model_called["count"] == 0


def test_run_critic_pass_falls_through_to_model_when_word_not_in_dictionary(
    monkeypatch,
):
    import content_batch_graph.domain.critic as critic_module

    def _fake_lookup_word(word):
        return {
            "word": word,
            "found": False,
            "part_of_speech": None,
            "definition": None,
        }

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            return SimpleNamespace(
                is_valid=True, replacement_text="katotohanan", reasoning="why"
            )

    monkeypatch.setattr(critic_module, "lookup_word", _fake_lookup_word)
    monkeypatch.setattr(
        critic_module, "get_model", lambda provider_id=None: _FakeModel()
    )

    result = run_critic_pass(
        "some text with katotohanen in it",
        _finding("katotohanen", "possible typo"),
        "Filipino",
    )

    assert result["is_valid"] is True
    assert result["replacement_text"] == "katotohanan"


def test_run_critic_pass_skips_dictionary_check_for_non_filipino_language(
    monkeypatch,
):
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


def test_run_critic_pass_normalizes_hyphen_lookalikes_in_replacement(monkeypatch):
    import content_batch_graph.domain.critic as critic_module

    class _FakeModel:
        def with_structured_output(self, schema):
            return self

        def __call__(self, _inputs):
            # U+2011 NON-BREAKING HYPHEN, as seen from a real Cerebras response —
            # visually identical to "-" but a different character.
            return SimpleNamespace(
                is_valid=True, replacement_text="amar‑te", reasoning="why"
            )

    monkeypatch.setattr(
        critic_module, "get_model", lambda provider_id=None: _FakeModel()
    )

    result = run_critic_pass(
        "some text with amar-Te in it",
        _finding("amar-Te", "capitalization issue", category="awkward_phrasing"),
        "Portuguese",
    )

    assert result["replacement_text"] == "amar-te"
    assert "‑" not in result["replacement_text"]


def test_run_critic_pass_skips_dictionary_check_for_non_typo_category(monkeypatch):
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
