import pydantic
import pytest

from content_batch_graph.domain.roles import build_finding_schema, get_role


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


def test_build_finding_schema_constrains_category_to_the_roles_categories():
    role = get_role("native_reader_batch")
    schema = build_finding_schema(role)

    valid = schema(
        findings=[
            {"quoted_text": "x", "issue": "y", "proposed_text": "", "category": "typo"}
        ]
    )
    assert valid.findings[0].category == "typo"

    with pytest.raises(pydantic.ValidationError):
        schema(
            findings=[
                {
                    "quoted_text": "x",
                    "issue": "y",
                    "proposed_text": "",
                    "category": "not_a_real_category",
                }
            ]
        )


def test_build_finding_schema_defaults_to_empty_findings():
    role = get_role("native_reader_batch")
    schema = build_finding_schema(role)
    assert schema().findings == []


def test_build_finding_schema_json_schema_has_required_fields():
    role = get_role("native_reader_batch")
    json_schema = build_finding_schema(role).model_json_schema()
    finding_props = next(iter(json_schema["$defs"].values()))["properties"]
    assert set(finding_props) == {"quoted_text", "issue", "proposed_text", "category"}


def test_build_finding_schema_requires_proposed_text():
    # proposed_text has no Python default (unlike earlier), so it's listed in the
    # JSON schema's "required" array, and omitting it raises. This is deliberate:
    # Groq's structured-output enforcement has been observed dropping an optional
    # field entirely rather than emitting the schema's own default, which fails the
    # call with json_validate_failed even though the field was never required by us.
    # Forcing the model to always emit it (empty string if none) avoids that.
    role = get_role("native_reader_batch")
    schema = build_finding_schema(role)
    finding_schema = next(iter(schema.model_json_schema()["$defs"].values()))
    assert "proposed_text" in finding_schema["required"]

    with pytest.raises(pydantic.ValidationError):
        schema(findings=[{"quoted_text": "x", "issue": "y", "category": "typo"}])
