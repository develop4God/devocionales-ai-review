"""The validate_pass node: real programmatic check that fix_pass didn't break the file."""

from __future__ import annotations

from content_batch_graph.domain.validate import validate_field_fix, validate_json
from content_batch_graph.state import BatchState


def validate_pass(state: BatchState) -> dict:
    field_path = state.get("field_path")
    if field_path:
        passed, error = validate_field_fix(
            state["file_path"], field_path, state["fixed_text"] or ""
        )
    else:
        passed, error = validate_json(state["fixed_text"] or "")
    return {"validation_passed": passed, "validation_error": error}
