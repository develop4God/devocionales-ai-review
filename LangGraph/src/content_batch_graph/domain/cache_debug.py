"""
Debug-level visibility into whether Ollama's prompt-prefix cache is firing for a
structured-output call. Every real model call in this graph shares the same
prompt shape (fixed persona/system message, only source_text varies) — a fast
prompt_eval_duration relative to prompt_eval_count on a repeat call is the signal
that the shared prefix was reused instead of reprocessed. Only meaningful for
local (Ollama) providers; other providers don't expose these fields.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_ollama_cache_stats(response_metadata: dict) -> None:
    """
    Logs prompt_eval_count/prompt_eval_duration/eval_count/eval_duration from an
    Ollama response's response_metadata, if present. No-op for any other
    provider's metadata shape (those fields are simply absent).
    """
    if response_metadata.get("model_provider") != "ollama":
        return

    prompt_eval_count = response_metadata.get("prompt_eval_count")
    prompt_eval_duration = response_metadata.get("prompt_eval_duration")
    if prompt_eval_count is None or prompt_eval_duration is None:
        return

    logger.debug(
        "ollama cache stats: model=%s prompt_eval_count=%s "
        "prompt_eval_duration=%.3fs eval_count=%s eval_duration=%.3fs",
        response_metadata.get("model_name"),
        prompt_eval_count,
        prompt_eval_duration / 1e9,
        response_metadata.get("eval_count"),
        (response_metadata.get("eval_duration") or 0) / 1e9,
    )
