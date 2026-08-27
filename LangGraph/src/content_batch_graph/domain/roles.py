"""
Resolve a configured role (config/roles.yml) into its persona/instructions data.

No role is hardcoded here — the flag pass is a role-agnostic engine fed whatever
role data this module resolves. Adding a new role means adding an entry to
roles.yml, not writing a new function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from pydantic import BaseModel, Field, create_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "config" / "roles.yml"

_config: dict[str, Any] | None = None


class Role(TypedDict):
    id: str
    name: str
    persona: str
    categories: list[str]


def _load_config() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def get_role(role_id: str | None = None) -> Role:
    """
    Returns the role data for the given role id, or the configured default_role if
    none is given. Raises if the role id isn't found.
    """
    config = _load_config()
    role_id = role_id or config["settings"]["default_role"]
    roles = {r["id"]: r for r in config["roles"]}
    if role_id not in roles:
        raise ValueError(f"Role '{role_id}' not found.")
    return roles[role_id]


def build_finding_schema(role: Role) -> type[BaseModel]:
    """
    The structured-output schema for a role's findings, with `category`
    constrained to a Literal of exactly the role's declared categories —
    schema-level enforcement instead of relying on a free-text description,
    which small models don't reliably follow.

    One definition shared by both call sites that need a Finding-shaped
    response: domain/flag.py's live path (.with_structured_output(), a
    LangChain mechanism) and domain/review_gen.py's batch path
    (.model_json_schema(), passed as response_format to the batch API) — both
    need the exact same field set (quoted_text, issue, proposed_text, category)
    and the same verbatim-quote/empty-if-none framing, so one schema is
    defined here rather than two copies that could drift apart.
    """
    category_type = Literal[tuple(role["categories"])]  # type: ignore[valid-type]
    finding_schema = create_model(
        "_FindingSchema",
        quoted_text=(
            str,
            Field(
                description="The exact problematic text, quoted verbatim from the source."
            ),
        ),
        issue=(
            str,
            Field(description="What's wrong with the quoted text, in English."),
        ),
        proposed_text=(
            str,
            # No default -> required in the emitted JSON schema. Groq's structured-
            # output enforcement has been observed omitting an optional field
            # entirely rather than emitting the schema's own default, which trips
            # json_validate_failed even though our schema never required it. Forcing
            # the model to always emit the field (empty string if none) sidesteps
            # that failure mode instead of relying on the provider's default-value
            # handling.
            Field(
                description="The corrected replacement for quoted_text. Empty "
                'string ("") if no replacement is proposed.',
            ),
        ),
        category=(category_type, Field(description="The category of this finding.")),
    )
    return create_model(
        "_FlagResponse",
        findings=(
            list[finding_schema],
            Field(
                default_factory=list,
                description="Every issue found. Empty if the text has no issues.",
            ),
        ),
    )
