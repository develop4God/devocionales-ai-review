import pytest

from content_batch_graph.domain.roles import get_role


def test_get_role_uses_default_role_when_none_given():
    role = get_role()
    assert role["id"] == "native_reader"


def test_get_role_returns_role_by_id():
    role = get_role("native_reader")
    assert role["id"] == "native_reader"
    assert "typo" in role["categories"]


def test_get_role_persona_formats_with_language():
    role = get_role("native_reader")
    persona = role["persona"].format(language="Spanish")
    assert "native Spanish speaker" in persona


def test_get_role_raises_for_unknown_role_id():
    with pytest.raises(ValueError, match="not found"):
        get_role("nonexistent_role")
