from content_batch_graph.domain.validate import validate_json


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
