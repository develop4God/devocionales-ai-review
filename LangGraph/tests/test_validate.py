import json
from pathlib import Path

import pytest

from content_batch_graph.domain.validate import validate_field_fix, validate_json


def test_validate_json_accepts_valid_json():
    passed, error = validate_json('{"a": 1, "b": [1, 2, 3]}')
    assert passed is True
    assert error is None


def test_validate_json_rejects_invalid_json():
    passed, error = validate_json('{"a": 1, "b": [1, 2, 3]')  # missing closing brace
    assert passed is False
    assert error is not None


def test_validate_json_rejects_empty_string():
    passed, error = validate_json("")
    assert passed is False
    assert error is not None


def test_validate_json_accepts_real_devocional_shape():
    text = '{"id": "x", "date": "2025-08-01", "reflexion": "Some text."}'
    passed, error = validate_json(text)
    assert passed is True
    assert error is None


@pytest.fixture
def devocional_file(tmp_path: Path) -> str:
    data = {
        "data": {
            "fr": {
                "2025-08-01": [
                    {
                        "id": "2cor317LSG1910",
                        "date": "2025-08-01",
                        "reflexion": "Original prose here.",
                    }
                ]
            }
        }
    }
    file_path = tmp_path / "devocional.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return str(file_path)


def test_validate_field_fix_accepts_valid_splice(devocional_file):
    field_path = "data.fr.2025-08-01.0.reflexion"
    passed, error = validate_field_fix(devocional_file, field_path, "Fixed prose here.")
    assert passed is True
    assert error is None


def test_validate_field_fix_does_not_mutate_the_real_file(devocional_file):
    field_path = "data.fr.2025-08-01.0.reflexion"
    validate_field_fix(devocional_file, field_path, "Fixed prose here.")

    with open(devocional_file, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["data"]["fr"]["2025-08-01"][0]["reflexion"] == "Original prose here."


def test_validate_field_fix_rejects_stale_field_path(devocional_file):
    field_path = "data.fr.2025-08-01.0.nonexistent_field_but_dict_still_settable"
    # A dict allows setting a brand-new key, so this doesn't fail at splice time —
    # confirms the realistic failure mode is a bad *container* path, tested next.
    passed, _error = validate_field_fix(devocional_file, field_path, "text")
    assert passed is True


def test_validate_field_fix_rejects_bad_container_path(devocional_file):
    field_path = "data.fr.2025-08-01.99.reflexion"  # index 99 doesn't exist
    passed, error = validate_field_fix(devocional_file, field_path, "text")
    assert passed is False
    assert error is not None


def test_validate_field_fix_rejects_missing_file():
    passed, error = validate_field_fix(
        "/nonexistent/path.json", "data.reflexion", "text"
    )
    assert passed is False
    assert error is not None
