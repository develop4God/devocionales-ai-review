"""
Resolve a batch-capable provider (config/providers.yml) into a BatchProviderConfig.

The batch counterpart to domain/providers.get_model(): same single config source
(providers.yml, loaded through providers._load_config so there is exactly one yml
loader in this package), different output — a transport config for the offline
batch API rather than a live LangChain chat model.
"""

from __future__ import annotations

from typing import Any

from batch_common import BatchProviderConfig

from content_batch_graph.domain.providers import _load_config


def _batch_providers() -> dict[str, dict[str, Any]]:
    config = _load_config()
    return {p["id"]: p for p in config["providers"] if p.get("enabled", True)}


def get_batch_provider(provider_id: str) -> BatchProviderConfig:
    """
    Returns the batch transport config for the given provider id.

    Raises ValueError if the id isn't found/enabled, or if the entry doesn't
    declare batch.supported: true — an entry without it has no batch endpoint to
    submit against, and failing here beats failing after a 365-record upload.
    """
    providers = _batch_providers()
    if provider_id not in providers:
        raise ValueError(f"Provider '{provider_id}' not found or not enabled.")

    provider = providers[provider_id]
    batch = provider.get("batch") or {}
    if not batch.get("supported"):
        raise ValueError(
            f"Provider '{provider_id}' does not have batch.supported: true in "
            f"providers.yml and cannot be used for batch generation."
        )

    for required in ("base_url", "model", "env_var"):
        if not provider.get(required):
            raise ValueError(
                f"Batch provider '{provider_id}' is missing required field "
                f"'{required}' in providers.yml."
            )

    return BatchProviderConfig(
        provider_id=provider["id"],
        base_url=provider["base_url"],
        model=provider["model"],
        env_var=provider["env_var"],
        endpoint=batch.get("endpoint", "/v1/chat/completions"),
        completion_window=batch.get("completion_window", "24h"),
        extra_record_fields=batch.get("extra_record_fields"),
        account_id_env_var=provider.get("account_id_env_var"),
    )
