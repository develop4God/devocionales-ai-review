"""
Resolve a configured provider (config/providers.yml) into a real LangChain chat model.

No provider is hardcoded here beyond the mapping from a provider's declared `package`
to the LangChain integration class that implements it — add a new provider entirely
in providers.yml; this module never needs a new branch for a new model, only for a
genuinely new integration package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "config" / "providers.yml"

load_dotenv(dotenv_path=_REPO_ROOT / ".env")

_config: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def _build_model(provider: dict[str, Any], max_retries: int) -> BaseChatModel:
    package = provider["package"]

    # Local providers run on-machine and need no API key or retry handling —
    # there's no shared rate limit to hit.
    if package == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=provider["model"],
            base_url=provider.get("base_url", "http://localhost:11434"),
        )

    api_key = os.environ.get(provider["env_var"])
    if not api_key:
        raise RuntimeError(
            f"Provider '{provider['id']}' requires {provider['env_var']} to be set."
        )

    if package == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=provider["model"], api_key=api_key, max_retries=max_retries
        )
    if package == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=provider["model"],
            api_key=api_key,
            base_url=provider.get("base_url"),
            max_retries=max_retries,
            extra_body=provider.get("extra_body"),
        )
    if package == "cerebras":
        from langchain_cerebras import ChatCerebras

        return ChatCerebras(
            model=provider["model"], api_key=api_key, max_retries=max_retries
        )
    raise ValueError(f"Unknown provider package: {package!r}")


def get_model(provider_id: str | None = None) -> BaseChatModel:
    """
    Returns a real LangChain chat model for the given provider id, or the configured
    default_provider if none is given. Raises if the required API key isn't set.

    The model is constructed with settings.max_retries from providers.yml, so a
    transient 429 (rate limit) is retried with backoff by the underlying SDK client
    instead of raising immediately.
    """
    config = _load_config()
    provider_id = provider_id or config["settings"]["default_provider"]
    providers = {p["id"]: p for p in config["providers"] if p.get("enabled", True)}
    if provider_id not in providers:
        raise ValueError(f"Provider '{provider_id}' not found or not enabled.")
    provider = providers[provider_id]
    # Batch-only entries describe an offline JSONL job, not a live chat model —
    # there is no synchronous client to construct for them. Caught here rather
    # than inside _build_model(), where it would surface as a confusing
    # ChatOpenAI misconfiguration against a batch-only endpoint.
    if provider.get("client_type") == "batch":
        raise ValueError(
            f"Provider '{provider_id}' is batch-only (client_type: batch) and has no "
            f"live chat model. Resolve it with "
            f"domain.batch_providers.get_batch_provider() instead."
        )
    max_retries = config["settings"].get("max_retries", 2)
    return _build_model(provider, max_retries)
