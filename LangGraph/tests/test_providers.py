import pytest

import content_batch_graph.domain.providers as providers_module
from content_batch_graph.domain.providers import get_model


@pytest.fixture(autouse=True)
def _reset_providers_config_cache():
    # _config is a module-level cache -- a test that sets
    # CONTENT_BATCH_GRAPH_PROVIDERS_CONFIG must see a fresh load, and every other
    # test must not see a config left over from one that did.
    providers_module._config = None
    yield
    providers_module._config = None


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
    # No provider_id given -> resolves settings.default_provider from
    # config/providers.yml, whatever that's currently set to, rather than calling
    # get_model(explicit_id) directly.
    monkeypatch.setenv("CEREBRAS_API_KEY", "fake-key-for-construction-only")
    model = get_model()
    assert type(model).__name__ == type(get_model("cerebras_gpt_oss_120b")).__name__


def test_get_model_local_provider_needs_no_api_key():
    model = get_model("ollama_local")
    assert type(model).__name__ == "ChatOllama"


def test_get_model_passes_max_retries_from_config(monkeypatch):
    # settings.max_retries in providers.yml previously wasn't wired into the
    # constructed model at all — a 429 raised immediately instead of retrying with
    # backoff. Confirms it's now actually passed through.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-construction-only")
    model = get_model("anthropic_default")
    assert model.max_retries == 2


def test_get_model_passes_extra_body_from_config_for_openai_package(monkeypatch):
    # groq_gpt_oss_20b/120b declare extra_body: {reasoning_effort: low} in
    # providers.yml — gpt-oss's default reasoning_effort ("medium") can burn the
    # whole completion-token budget on chain-of-thought before emitting the
    # structured JSON answer, failing the call with json_validate_failed. Confirms
    # that config value actually reaches the constructed ChatOpenAI client.
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-construction-only")
    model = get_model("groq_gpt_oss_20b")
    assert model.extra_body == {"reasoning_effort": "low"}


def test_get_model_extra_body_defaults_to_none_when_not_configured(monkeypatch):
    # openai_default has no extra_body in providers.yml — confirms providers that
    # don't need it aren't forced to declare one, and get None rather than an error.
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-construction-only")
    model = get_model("openai_default")
    assert model.extra_body is None


def test_get_model_respects_config_path_env_override(monkeypatch, tmp_path):
    # scripts/run_live_validation.py's --config flag sets
    # CONTENT_BATCH_GRAPH_PROVIDERS_CONFIG so a parallel worker process can use its
    # own providers.yml (a different default_provider) without touching the shared
    # config file. Confirms get_model() actually reads the override, not just the
    # module's own default path.
    override_config = tmp_path / "custom_providers.yml"
    override_config.write_text(
        """
providers:
  - id: only_provider_here
    name: "Test/only-provider"
    priority: 1
    enabled: true
    client_type: api
    package: openai
    base_url: "https://example.invalid/v1"
    model: "test-model"
    env_var: OPENAI_API_KEY

settings:
  default_provider: only_provider_here
  max_retries: 1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTENT_BATCH_GRAPH_PROVIDERS_CONFIG", str(override_config))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-construction-only")

    model = get_model()  # no provider_id -> resolves default_provider from override
    assert model.model_name == "test-model"


def test_get_model_uses_default_config_path_when_env_not_set(monkeypatch):
    monkeypatch.delenv("CONTENT_BATCH_GRAPH_PROVIDERS_CONFIG", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-construction-only")
    # anthropic_default only exists in the real config/providers.yml -- resolving
    # it successfully confirms the override env var being unset falls back to the
    # module's real default path, not a broken/empty one.
    model = get_model("anthropic_default")
    assert type(model).__name__ == "ChatAnthropic"
