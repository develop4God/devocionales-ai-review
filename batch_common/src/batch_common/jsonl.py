"""
JSONL read/write and OpenAI-compatible batch request-record shaping.

One place that knows the batch input record envelope
({custom_id, method, url, body}), so a consuming project builds only the body's
model/messages and never re-derives the envelope itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    """Write records one-JSON-object-per-line to path. Creates parents. Returns path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
    return path


def read_jsonl(path: Path) -> Iterator[dict]:
    """
    Yield one parsed dict per non-blank line.

    Malformed lines raise — a caller that wants to tolerate partial garbage
    (e.g. a batch results collector) should read the raw lines itself and decide
    per line, rather than having this silently drop data.
    """
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chat_request_record(
    custom_id: str,
    model: str,
    messages: list[dict],
    endpoint: str = "/v1/chat/completions",
    **body: object,
) -> dict:
    """
    Build one OpenAI-compatible batch input record.

    Extra keyword args (max_tokens, temperature, response_format, ...) are merged
    into the request body alongside model/messages.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": endpoint,
        "body": {"model": model, "messages": messages, **body},
    }
