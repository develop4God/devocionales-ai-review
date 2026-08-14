"""
batch_common — shared OpenAI-compatible batch API transport.

Stdlib-only by design: both GEP (plain python3) and LangGraph (uv-managed) can
depend on this without inheriting the other's dependency tree.
"""

from batch_common.client import BatchClient
from batch_common.config import (
    BatchAPIError,
    BatchProviderConfig,
    account_id_from_env,
    api_key_from_env,
)
from batch_common.jsonl import chat_request_record, read_jsonl, write_jsonl
from batch_common.paths import BatchPaths

__all__ = [
    "BatchAPIError",
    "BatchClient",
    "BatchPaths",
    "BatchProviderConfig",
    "account_id_from_env",
    "api_key_from_env",
    "chat_request_record",
    "read_jsonl",
    "write_jsonl",
]
