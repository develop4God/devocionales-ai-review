"""
Provider configuration for the OpenAI-compatible batch API.

The transport layer (client.py) takes one of these and nothing else — it never
reads a yml file, an env var name, or a project-specific config loader itself.
Each consuming project maps its own provider config format into a
BatchProviderConfig and hands it over, which is what keeps this package usable
from GEP (providers.yml via cloud_client) and LangGraph (providers.yml via
domain/providers) without either coupling leaking in here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BatchProviderConfig:
    """Everything the batch transport needs to talk to one provider."""

    provider_id: str
    base_url: str
    model: str
    env_var: str
    endpoint: str = "/v1/chat/completions"
    completion_window: str = "24h"
    # Provider-specific extras merged into each request body by the caller
    # (e.g. a reasoning/thinking toggle). Kept opaque here on purpose.
    extra_record_fields: dict | None = field(default=None)


class BatchAPIError(RuntimeError):
    """Raised on non-retryable batch API failures."""


def api_key_from_env(cfg: BatchProviderConfig) -> str:
    """
    Read the API key for this provider out of the environment.

    Raises BatchAPIError (not KeyError) with the variable name spelled out, so a
    missing key fails loudly at the CLI boundary instead of surfacing as an
    opaque 401 after an upload has already started.
    """
    key = os.environ.get(cfg.env_var, "")
    if not key:
        raise BatchAPIError(
            f"Environment variable {cfg.env_var!r} is not set "
            f"(required by provider {cfg.provider_id!r}).\n"
            f"Run: export {cfg.env_var}=<your_api_key>"
        )
    return key
