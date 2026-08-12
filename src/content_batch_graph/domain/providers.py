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
from langchain_core.language_models.chat_models import BaseChatModel

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "providers.yml"

_config: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def _build_model(provider: dict[str, Any]) -> BaseChatModel:
    api_key = os.environ.get(provider["env_var"])
    if not api_key:
        raise RuntimeError(
            f"Provider '{provider['id']}' requires {provider['env_var']} to be set."
        )

    package = provider["package"]
    if package == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=provider["model"], api_key=api_key)
    if package == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=provider["model"],
            api_key=api_key,
            base_url=provider.get("base_url"),
        )
    raise ValueError(f"Unknown provider package: {package!r}")


def get_model(provider_id: str | None = None) -> BaseChatModel:
    """
    Returns a real LangChain chat model for the given provider id, or the configured
    default_provider if none is given. Raises if the required API key isn't set.
    """
    config = _load_config()
    provider_id = provider_id or config["settings"]["default_provider"]
    providers = {p["id"]: p for p in config["providers"] if p.get("enabled", True)}
    if provider_id not in providers:
        raise ValueError(f"Provider '{provider_id}' not found or not enabled.")
    return _build_model(providers[provider_id])
