"""
Validate fixed content with real programmatic checks — no LLM involved.

Slice 3's minimum: confirm the fixed text is still valid JSON. This is the
structural guarantee GEP's old system never had — a fix is proven not to have
broken the file, not just self-assessed as correct by the model that wrote it.
"""

from __future__ import annotations

import json


def validate_json(text: str) -> tuple[bool, str | None]:
    """
    Returns (True, None) if text parses as valid JSON, otherwise (False, error_msg).
    """
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return False, str(e)
    return True, None
