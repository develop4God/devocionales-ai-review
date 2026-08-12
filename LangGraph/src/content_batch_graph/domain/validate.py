"""
Validate fixed content with real programmatic checks — no LLM involved.

Slice 3's minimum: confirm the fixed text is still valid JSON. This is the
structural guarantee GEP's old system never had — a fix is proven not to have
broken the file, not just self-assessed as correct by the model that wrote it.

Slice 3.6: when fixed_text is a single field's prose (not a whole JSON document
itself), field_path locates where that field lives in the real source file at
file_path, so the fix can be spliced back in and the *whole file's* structure
validated — not just the bare prose string, which would never parse as JSON.
"""

from __future__ import annotations

import json
from typing import Any


def validate_json(text: str) -> tuple[bool, str | None]:
    """
    Returns (True, None) if text parses as valid JSON, otherwise (False, error_msg).
    """
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return False, str(e)
    return True, None


def _resolve_path(data: Any, field_path: str) -> tuple[Any, str | int]:
    """
    Walks field_path (dot-separated keys/indices) to the parent container of the
    final segment. Returns (parent_container, final_key_or_index).
    """
    parts = field_path.split(".")
    node = data
    for part in parts[:-1]:
        key: str | int = int(part) if part.isdigit() else part
        node = node[key]
    last = parts[-1]
    final_key: str | int = int(last) if last.isdigit() else last
    return node, final_key


def validate_field_fix(
    file_path: str, field_path: str, fixed_text: str
) -> tuple[bool, str | None]:
    """
    Splices fixed_text into a copy of the real JSON document at file_path,
    at the location field_path points to, and confirms the result is still
    valid, well-formed JSON. Returns (True, None) on success, (False, error_msg)
    on any failure (file read, path resolution, or JSON validity).
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            document = json.load(f)
    except OSError as e:
        return False, f"Could not read file_path: {e}"
    except json.JSONDecodeError as e:
        return False, f"file_path is not valid JSON to begin with: {e}"

    try:
        parent, final_key = _resolve_path(document, field_path)
        parent[final_key] = fixed_text
    except (KeyError, IndexError, TypeError) as e:
        return False, f"field_path '{field_path}' did not resolve in file_path: {e}"

    try:
        json.dumps(document)
    except (TypeError, ValueError) as e:
        return False, f"Spliced document is not valid JSON: {e}"

    return True, None
