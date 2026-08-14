import pytest

from content_batch_graph.domain.batch_providers import get_batch_provider
from content_batch_graph.domain.providers import get_model

BATCH_ID = "fireworks_batch_devotional_gen"


def test_get_batch_provider_maps_yml_entry_to_config():
    cfg = get_batch_provider(BATCH_ID)
    assert cfg.provider_id == BATCH_ID
    assert cfg.base_url == "https://api.fireworks.ai/inference/v1"
    assert cfg.model == "accounts/fireworks/models/deepseek-v3p2"
    assert cfg.env_var == "FIREWORKS_API_KEY"
    assert cfg.endpoint == "/v1/chat/completions"
    assert cfg.completion_window == "24h"


def test_get_batch_provider_needs_no_api_key():
    # Resolving config must not touch the environment — build-year --dry-run has
    # to work with no key set at all.
    get_batch_provider(BATCH_ID)


def test_get_batch_provider_raises_for_unknown_id():
    with pytest.raises(ValueError, match="not found or not enabled"):
        get_batch_provider("nonexistent_provider")


def test_get_batch_provider_raises_for_non_batch_provider():
    with pytest.raises(ValueError, match="batch.supported"):
        get_batch_provider("anthropic_default")


def test_get_model_refuses_a_batch_only_provider():
    # A batch entry must never reach _build_model() and become a live ChatOpenAI.
    with pytest.raises(ValueError, match="batch-only"):
        get_model(BATCH_ID)


def test_get_model_still_works_for_normal_providers():
    assert type(get_model("ollama_local")).__name__ == "ChatOllama"
